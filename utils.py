#==========================================
# Import libraries
#==========================================

# Standard library
import argparse
import glob
import os
import re
import shutil
import subprocess
# Scientific computing
import numpy as np
import numpy.ma as ma
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import curve_fit

# Visualization
import matplotlib.pyplot as plt

# Astronomy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS
import astropy.visualization as vis

# Machine learning
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Utilities
from tqdm import tqdm
from typing import List, Tuple, Sequence

# Deactivate warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#=============================================================
#
# Functions for masking the images
#
#=============================================================

def masks_maker_total_image(dir):
    """
    Run Gnuastro NoiseChisel and Segment on all FITS images in a directory,
    creating masks for each image and saving them into dedicated subfolders.

    Parameters
    ----------
    dir : str
        Directory containing the input FITS files.

    Returns
    -------
    None
        Creates output FITS masks on disk, no explicit return value.

    Workflow
    --------
    1. Ensure the output directories exist:
       - ./Process_data/Mask_data/Mask_noisechisel
       - ./Process_data/Mask_data/Mask_segment
    2. Collect all `.fits` files from the input directory.
    3. For each FITS image:
       - Run `astnoisechisel` to detect faint/extended structures.
       - Run `astsegment` on the NoiseChisel output to label objects.
       - Save results into the respective subfolders.
    """

    # Step 1: Create required directories for masks
    dir_process_data = "./Process_data"
    mask_dir = os.path.join(dir_process_data, "Mask_data")
    dir_noisechisel = os.path.join(mask_dir, "Mask_noisechisel")
    dir_segment = os.path.join(mask_dir, "Mask_segment")

    if not os.path.isdir(dir_process_data):
        os.makedirs(dir_process_data)
    if not os.path.isdir(mask_dir):
        os.makedirs(mask_dir)
    if not os.path.isdir(dir_noisechisel):
        os.makedirs(dir_noisechisel)
    if not os.path.isdir(dir_segment):
        os.makedirs(dir_segment)

    # Step 2: Gather all FITS files in the input directory
    fits_files = [file for file in os.listdir(dir) if file.endswith(".fits")]
    fits_files = np.sort(fits_files)

    # Step 3: Loop over FITS files and run NoiseChisel + Segment
    for name in fits_files:
        print("\n\n" + "=" * 60)
        print(f"\n Masking {name}")
        print("\n" + "=" * 60)

        path = os.path.join(dir, name)

        # Define output paths
        output_noisechisel = os.path.join(dir_noisechisel, name.replace(".fits", "_noisechisel.fits"))
        output_segment     = os.path.join(dir_segment, name.replace(".fits", "_segment.fits"))

        # Run NoiseChisel to detect faint structures
        os.system(
            f"astnoisechisel {path} --tilesize=20,20 --interpnumngb=5 --dthresh=0.05 "
            f"--snminarea=2 --rawoutput --quiet --output={output_noisechisel}"
        )

        # Run Segment to create labeled object masks
        os.system(
            f"astsegment {output_noisechisel} --tilesize=10,10 --interpnumngb=1 "
            f"--gthresh=-10 --objbordersn=0 --quiet --minnumfalse=1 --output={output_segment}"
        )

    return

#=============================================================
#
# FUNCTIONS FOR SELECTING STARS AND PROCESSING RADIAL PROFILES
#
#=============================================================

def fits_to_dataframe(fits_path):
    """
    Load a FITS table and convert it into a pandas DataFrame.

    Parameters
    ----------
    fits_path : str
        Path to the FITS file.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from extension 1 (BinTableHDU).
    """
    with fits.open(fits_path) as hdul:
        data = hdul[1].data  # Assumes the relevant data is stored in extension 1
    return pd.DataFrame(data)


def dataframe_to_fits(df, fits_path):
    """
    Save a pandas DataFrame as a FITS file.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to be saved.
    fits_path : str
        Output path for the FITS file.

    Notes
    -----
    - Converts the DataFrame into a NumPy array and stores it in the primary HDU.
    - This creates a valid but minimal FITS file: column names, types, and units are not preserved.
      For richer metadata, consider using `astropy.io.fits.BinTableHDU.from_columns`.
    """
    hdu = fits.PrimaryHDU(df.to_numpy())
    hdul = fits.HDUList([hdu])
    hdul.writeto(fits_path, overwrite=True)


def extract_number(filename):
    """
    Extract an integer from a filename with the pattern '_<num>.fits'.

    Parameters
    ----------
    filename : str
        Filename to parse (e.g., "image_12.fits").

    Returns
    -------
    int
        Extracted number. Returns 0 if no match is found.

    Examples
    --------
    >>> extract_number("frame_25.fits")
    25
    >>> extract_number("noframe.fits")
    0
    """
    match = re.search(r'_(\d*)\.fits$', filename)
    return int(match.group(1)) if match and match.group(1) else 0


def masks_maker(name, profile):
    """
    Generate a binary mask from a FITS image using Gnuastro tools (NoiseChisel, Segment),
    apply the mask to the original image, and compute a radial profile.

    Parameters
    ----------
    name : str
        Path to the input FITS image (will be overwritten with the masked image).
    profile : str
        Output path for the radial profile generated by `astscript-radial-profile`.

    Returns
    -------
    np.ndarray
        Binary mask array (1 = object, 0 = signal).
    """
    # Print a header for visual separation in logs
    print("\n\n" + "=" * 60)
    print(f"\n Making inner mask and profile of {name}")
    print("\n" + "=" * 60)

    # Define intermediate output filenames
    output_astarith    = name.replace(".fits", "_astarithmetic.fits")
    output_noisechisel = name.replace(".fits", "_noisechisel.fits")
    output_segment     = name.replace(".fits", "_segment.fits")

    # Step 1: Preprocess image with `astarithmetic` (set zeros to NaN)
    os.system(
        f"astarithmetic {name} {name} --quiet --output={output_astarith} 0.0 eq nan where -g1"
    )

    # Step 2: Run NoiseChisel to detect low-surface-brightness features
    os.system(
        f"astnoisechisel {output_astarith} --tilesize=10,10 --quiet --interpnumngb=1 "
        f"--qthresh=0.5 --minnumfalse=1 --rawoutput --output={output_noisechisel}"
    )

    # Step 3: Segment the detections with `astsegment`
    os.system(
        f"astsegment {output_noisechisel} --tilesize=10,10 --interpnumngb=1 --quiet "
        f"--gthresh=-10 --objbordersn=0 --minnumfalse=1 --output={output_segment}"
    )

    # Step 4: Load segmentation map (HDU 3 usually contains object labels)
    data = fits.open(output_segment)
    objects = data[3].data

    # Identify the label of the central pixel
    number = stats.mode(
        objects[int(len(objects) / 2), int(len(objects) / 2)],
        axis=None,
        nan_policy="omit",
    ).mode

    # Convert segmentation to binary mask
    objects[objects == number] = 0
    objects[objects != 0] = 1

    # Step 5: Apply mask to the original image (set object pixels to NaN)
    imagen = fits.getdata(name)
    imagen[objects != 0] = np.nan

    # Overwrite the original FITS with the masked version
    hdu = fits.PrimaryHDU(imagen)
    hdu.writeto(name, overwrite=True)

    # Step 6: Remove intermediate files
    os.system("rm " + output_astarith)
    os.system("rm " + output_noisechisel)
    os.system("rm " + output_segment)

    # Step 7: Compute radial profile and save output
    os.system(
        f"astscript-radial-profile {name} --hdu=0 --quiet --measure=mean,std,area,semi-major "
        f"--rmax=2500 -o {profile}"
    )
    return objects
 

def prepare_star_selection(filter: str, name: str, dir: str,
                           mag_inf_lim: float, mag_sup_lim: float, min_dist: float):
    """
    Prepare working directories, run `astscript-psf-select-stars` to select stars,
    and normalize the output structure for further processing.

    Parameters
    ----------
    filter : str
        Photometric filter name (e.g., "g" or "r").
    name : str
        Input FITS filename.
    dir : str
        Path to the directory containing the input FITS files.
    mag_inf_lim : float
        Lower limit of the magnitude range for star selection.
    mag_sup_lim : float
        Upper limit of the magnitude range for star selection.
    min_dist : float
        Minimum angular distance (in degrees) allowed between stars.

    Returns
    -------
    tuple[Path, Path, Path, Path]
        Paths needed for downstream processing:
        - ruta_segment : Path to the segmentation FITS file.
        - ruta_star    : Path to the selected-star FITS file.
        - ruta_out_norm: Path to the normalized output (base name).
        - ruta_gal     : Path to the galaxy FITS file in the cut directory.
    """
    # Base directory for star subtraction products
    base = Path("./Process_data/Subtract_stars")

    # Root directory for "Quit_stars_<filter>"
    quit_stars_root = base / f"Quit_stars_{filter}"
    quit_stars_root.mkdir(parents=True, exist_ok=True)

    # Derive galaxy identifier (first part of filename before "_")
    name_2 = name.split("_")[0]

    # Directory to store stars selected for this galaxy/filter
    stars_dir = quit_stars_root / f"Stars_{name_2}_{filter}"

    # Clean and recreate directory to ensure a fresh start
    if stars_dir.exists():
        shutil.rmtree(stars_dir)
    stars_dir.mkdir(parents=True, exist_ok=True)

    # Full path to input galaxy FITS
    ruta_completa = Path(dir) / name
    # Path for output FITS with stars removed
    ruta_out_quit = stars_dir / name.replace(".fits", "_quit.fits")

    # Run Gnuastro tool to select PSF stars based on magnitude and distance limits
    subprocess.run(
        [
            "astscript-psf-select-stars",
            str(ruta_completa),
            "--quiet",
            f"--magnituderange={mag_inf_lim},{mag_sup_lim}",
            f"--mindistdeg={min_dist}",
            f"--output={ruta_out_quit}",
        ],
        check=True,
    )

    # Root directory for normalized radii results
    norm_root = base / "Norm_radii" / f"Quit_stars_{filter}"
    norm_root.mkdir(parents=True, exist_ok=True)

    # Print header for log visibility
    print("\n\n" + "=" * 60)
    print(f"\n PREPARING STAR SELECTION OF {name}")
    print("\n" + "=" * 60)

    # Directory for normalized star products
    norm_dir = norm_root / f"Stars_{name_2}_{filter}"
    # Directory for "close sources" (temporary storage of nearby sources)
    close_sources_dir = norm_root / f"Stars_{name_2}_{filter}_close_sources"
    close_sources_dir.mkdir(parents=True, exist_ok=True)

    # Clean and recreate normalized directory
    if norm_dir.exists():
        shutil.rmtree(norm_dir)
    norm_dir.mkdir(parents=True, exist_ok=True)

    # Collect .fits files from close_sources_dir that match the filter name
    files_moves = sorted(
        f for f in os.listdir(close_sources_dir)
        if re.search(fr"{re.escape(filter)}.*\.fits$", f)
    )

    # Move them into the normalized directory
    for file in files_moves:
        shutil.move(str(close_sources_dir / file), str(norm_dir))

    # Define final output paths for subsequent steps
    ruta_star = stars_dir / name.replace(".fits", "_quit.fits")
    ruta_out_norm = norm_dir / name.replace(".fits", "")
    ruta_gal = Path(dir) / name

    return ruta_star, ruta_out_norm, ruta_gal, norm_dir, name_2, close_sources_dir

def process_selected_stars(
    ruta_star: Path,
    ruta_gal: Path,
    ruta_out_norm_base: Path,
    norm_dir: Path,
    filter: str,
    name_2: str,
    masks_maker,           # function: (path: str|Path, radial_out: str|Path) -> np.ndarray (or None)
    extract_number,        # function: (filename: str) -> int
    crop_size_pix: Tuple[int, int] = (1000, 1000),
) -> Tuple[List[List[float]], List[List[float]], np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Create per-star crops with astcrop, build masks and radial profiles (masks_maker),
    and collect radii/counts arrays. Returns also RA/DEC/mag arrays and the indices to skip.

    Returns
    -------
    radius : list[list[float]]
    counts : list[list[float]]
    ra_stars : np.ndarray
    dec_stars : np.ndarray
    mags_stars : np.ndarray
    quitar : list[int]     # indices of problematic stars that were removed
    """
    norm_dir.mkdir(parents=True, exist_ok=True)

    # Read star table to get RA/DEC/MAG
    table = fits.getdata(ruta_star)
    ra_stars = np.asarray(table["ra"])
    dec_stars = np.asarray(table["dec"])
    mags_stars = np.asarray(table["phot_g_mean_mag"])

    # Generate per-star crops with astcrop
    for idx, (ra, dec) in enumerate(zip(ra_stars, dec_stars), start=0):
        out_stamp = norm_dir / f"{ruta_out_norm_base.name}_{idx}.fits"
        subprocess.run(
            [
                "astcrop",
                str(ruta_gal),
                "--mode=wcs",
                f"--center={ra},{dec}",
                "--widthinpix",
                "--quiet",
                f"--width={crop_size_pix[0]},{crop_size_pix[1]}",
                f"--output={out_stamp}",
            ],
            check=True,
        )

    # Gather generated stamps and sort by the trailing number
    files_stars = sorted(
        (f for f in os.listdir(norm_dir) if re.search(r".*\.fits$", f)),
        key=extract_number
    )

    # Prepare/refresh radial-profiles directory
    directorio_radial_profiles = Path(
        f"./Process_data/Subtract_stars/Norm_radii/Quit_stars_{filter}/Radial_profiles_{name_2}_{filter}"
    )
    if directorio_radial_profiles.exists():
        shutil.rmtree(directorio_radial_profiles)
    directorio_radial_profiles.mkdir(parents=True, exist_ok=True)

    radius: List[List[float]] = []
    counts: List[List[float]] = []
    quitar: List[int] = []
    contador = 0

    for file in files_stars:
        try:
            path = norm_dir / file
            radial = directorio_radial_profiles / file.replace(".fits", "_radial_profile.fits")

            # Build mask and radial profile
            masks_maker(str(path), str(radial))

            # Read radial profile: assume HDU=1 with 2 cols (radius, counts)
            with fits.open(radial) as hdul:
                data = hdul[1].data
                x = [data[i][0] for i in range(len(data))]
                y = [data[i][1] for i in range(len(data))]

            radius.append(x)
            counts.append(y)
            contador += 1

        except Exception as e:
            print("\n PROBLEMS WITH THIS, PROBABLY NANS")
            print(e)
            print(contador)

            # Clean temporary/failed products like the original code
            try:
                os.remove(str(path).replace(".fits", "_astarithmetic.fits"))
            except Exception:
                pass
            try:
                os.remove(str(path))
            except Exception:
                pass

            quitar.append(contador)
            contador += 1
            continue

    return np.array(radius), np.array(counts), ra_stars, dec_stars, mags_stars, quitar

def find_flat_points(radius: Sequence[Sequence[float]], counts: Sequence[Sequence[float]]) -> List[float]:
    """
    Find the last radius where each radial profile becomes flat, based on the slope.

    Parameters
    ----------
    radius : sequence of sequences
        Radii for each profile (list of arrays/lists).
    counts : sequence of sequences
        Count values for each profile (list of arrays/lists).

    Returns
    -------
    flat_points : list of float
        List with the radius value where each profile becomes flat.
        Returns 0 if no flat region is found.
    """
    flat_points: List[float] = []

    for x, y in zip(radius, counts):
        # Numerical derivative of log10(y) vs log10(x)
        dy_dx = np.gradient(np.log10(y), np.log10(x), edge_order=2)

        # Keep only valid values
        valid = ~np.isnan(dy_dx)
        dy_dx = np.asarray(dy_dx)[valid]
        x_valid = np.asarray(x)[valid]

        # Find indices where slope is close to zero
        flat_index = np.where(np.abs(dy_dx) < 2.5)[0]

        if len(flat_index) > 0:
            # Default: last flat point in the contiguous region
            flat_point = x_valid[flat_index[-1]]

            # Adjust if the flat region is broken (non-contiguous)
            for i in range(1, len(flat_index)):
                if flat_index[i] - flat_index[i - 1] > 1:
                    flat_point = x_valid[flat_index[i - 1] + 2]
                    break
        else:
            # If no flat region is found
            flat_point = 0

        flat_points.append(flat_point)

    return flat_points

def filter_and_clip_stars(
    norm_dir: str,
    filter: str,
    mags_stars: np.ndarray,
    ra_stars: np.ndarray,
    dec_stars: np.ndarray,
    flat_points: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter star files to keep only consistent indices and apply sigma clipping to flat points.

    Parameters
    ----------
    norm_dir : str
        Directory where normalized star FITS files are stored.
    filter : str
        Photometric filter (used in regex to identify files).
    mags_stars : np.ndarray
        Magnitudes of the stars.
    ra_stars : np.ndarray
        Right ascension of the stars.
    dec_stars : np.ndarray
        Declination of the stars.
    flat_points : list of float
        Radii where each profile becomes flat.

    Returns
    -------
    mags_stars : np.ndarray
        Filtered magnitudes.
    ra_stars : np.ndarray
        Filtered RA.
    dec_stars : np.ndarray
        Filtered DEC.
    flat_points : np.ndarray
        Filtered flat points.
    clip_flats : np.ndarray
        Sigma-clipped version of flat_points.
    """
    # Collect normalized FITS files matching the filter
    fits_files = [
        file for file in os.listdir(norm_dir)
        if re.search(fr"{re.escape(filter)}.*\.fits$", file)
    ]
    files = np.sort(fits_files)

    preserve_index: List[int] = []

    # Extract trailing numeric index from filename
    for file_name in files:
        last_component = file_name.split('_')[-1].split('.')[0]
        
        if last_component == '':
            index = 0
            preserve_index.append(int(index))
        else: preserve_index.append(int(last_component))

    # Indices to remove (difference between all and conserved)
    quitar = np.setdiff1d(np.arange(0,len(mags_stars)), np.array(preserve_index))

    # Filter arrays accordingly
    mags_stars = np.delete(mags_stars, quitar)
    ra_stars = np.delete(ra_stars, quitar)
    dec_stars = np.delete(dec_stars, quitar)
    flat_points = np.array(flat_points)

    return mags_stars, ra_stars, dec_stars, flat_points, preserve_index

def fit_ransac_and_build_rings(
    mags_stars: np.ndarray,
    flat_points: np.ndarray,
    px_scale: float,
    ruta_gal: str,
    ra_stars: np.ndarray,
    dec_stars: np.ndarray,
):
    """
    Fit RANSAC regression on (mag, log10(flat_points * px_scale)),
    compute saturation radii and ring sizes, and convert RA/DEC to pixel coordinates.
    """
    # Prepare data
    x = np.array(mags_stars, dtype=float)
    y = np.array(flat_points, dtype=float) * px_scale

    # Remove invalid values
    valid = (y > 0) & np.isfinite(np.log10(y))
    x = x[valid]
    y = y[valid]

    # RANSAC regression: log10(y) ~ slope * x + intercept
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=0.5,
        residual_threshold=0.25,
        random_state=0,
    )
    x_reshaped = x.reshape(-1, 1)
    ransac.fit(x_reshaped, np.log10(y))

    # Range for prediction
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_pred_ransac = ransac.predict(x_range)

    # Extract fit parameters
    slope = ransac.estimator_.coef_[0]
    intercept = ransac.estimator_.intercept_
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)

    # Saturation radius as a function of magnitude
    r_sat_sat = 10 ** (slope * mags_stars + intercept)

    # Define inner/outer radii of rings (in pixels)
    r_min_ring = 1.5 * r_sat_sat / px_scale
    r_max_ring = 4.0 * r_sat_sat / px_scale

    # Get WCS from galaxy FITS (HDU=1) and convert RA/DEC to pixel coords
    with fits.open(ruta_gal) as hdu:
        wcs = WCS(hdu[1].header)

    coords_sky = SkyCoord(ra=ra_stars, dec=dec_stars, unit="deg", frame="icrs")
    x_stars, y_stars = wcs.world_to_pixel(coords_sky)

    return (r_min_ring, r_max_ring, x_stars, y_stars, mags_stars)

def finalize_sources_and_write_fits(
    x_positions: np.ndarray,
    y_positions: np.ndarray,
    r_min_ring: np.ndarray,
    r_max_ring: np.ndarray,
    magnitudes: np.ndarray,
    preserve_index: np.ndarray,
    norm_dir: str,
    filter: str,
    close_sources_dir: str,
    ruta_star: str,
    ra_stars: np.ndarray,
    dec_stars: np.ndarray,
    mags_stars: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[int], np.ndarray]:
    # Distancia euclidiana
    def euclidean_distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # Selección por solapamiento usando r_max_ring y magnitudes
    indices_to_keep: List[int] = []
    n = len(x_positions)
    for i in range(n):
        within_radius = []
        for j in range(n):
            if i != j:
                d = euclidean_distance(x_positions[i], y_positions[i], x_positions[j], y_positions[j])
                if d <= r_max_ring[j]:
                    within_radius.append(j)
        if within_radius:
            min_magnitude_index = min(within_radius + [i], key=lambda idx: magnitudes[idx])
            if min_magnitude_index == i:
                indices_to_keep.append(i)
        else:
            indices_to_keep.append(i)

    # Filtrados finales
    x_stars = x_positions[indices_to_keep]
    y_stars = y_positions[indices_to_keep]
    r_min_ring = r_min_ring[indices_to_keep]
    r_max_ring = r_max_ring[indices_to_keep]
    indices_finales = np.array(np.sort(preserve_index))[indices_to_keep]
    magnitudes_filtered = magnitudes[indices_to_keep]
    close_sources = np.setdiff1d(np.array(np.sort(preserve_index)), np.sort(indices_finales))

    # Mover timesteps/recortes de fuentes descartadas a close_sources_dir
    source_directory = Path(norm_dir)
    close_sources_dir = Path(close_sources_dir)
    close_sources_dir.mkdir(parents=True, exist_ok=True)

    fit_files = [
        file for file in os.listdir(source_directory)
        if re.search(fr"{re.escape(filter)}.*\.fits$", file)
    ]
    files = np.sort(fit_files)

    for file_name in files:
        if file_name.endswith(".fits"):
            last_component = file_name.split("_")[-1].split(".")[0]
            if last_component == "":
                index = 0
            else:
                try:
                    index = int(last_component)
                except ValueError:
                    continue

            if index in close_sources:
                shutil.move(str(source_directory / file_name), str(close_sources_dir / file_name))
                print(f"Movido: {file_name}")
            else:
                print(f"Conservado: {file_name}")

    # Escribir FITS revisado con tabla actualizada
    hdul = fits.open(ruta_star)
    ra_final = ra_stars[indices_to_keep]
    dec_final = dec_stars[indices_to_keep]
    phot_g_final = mags_stars[indices_to_keep]

    data_combined = np.zeros(
        len(indices_finales),
        dtype=[
            ("ra", ">f4"),
            ("dec", ">f4"),
            ("phot_g_mean_mag", ">f4"),
            ("rmin_norm", ">i4"),
            ("rmax_norm", ">i4"),
            ("image_index", ">i4"),
        ],
    )
    data_combined["ra"] = ra_final
    data_combined["dec"] = dec_final
    data_combined["phot_g_mean_mag"] = phot_g_final
    data_combined["rmin_norm"] = np.round(r_min_ring).astype(int)
    data_combined["rmax_norm"] = (np.round(r_max_ring).astype(int) + 1)
    data_combined["image_index"] = indices_finales

    hdul[1].header["NAXIS2"] = int(len(indices_finales))
    new_hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=hdul[0].header),
            fits.BinTableHDU(data_combined, header=hdul[1].header),
        ]
    )
    new_hdul.writeto(ruta_star.replace(".fits", "_revised.fits"), overwrite=True)

    return()

#===============================================================
#
# FUNCTIONS FOR SUBTRACTING STARS AND BUILD SCATTER LIGHT MODELS
#
#===============================================================

def setup_subtractor_paths(
    filter: str,
    name: str,
    directorio_beard_cut: str,
    dir_psf: str,
):
    """
    Prepare I/O paths and directories for the star subtraction step.
    Cleans the stamps dir and returns all relevant paths and file lists.
    """

    # Roots
    direct_copy_root = Path(f"./Process_data/Subtract_stars/Subtrac_copy_{filter}")
    scatter_root     = Path(f"./Process_data/Subtract_stars/Scatter_field_{filter}")
    direct_copy_root.mkdir(parents=True, exist_ok=True)
    scatter_root.mkdir(parents=True, exist_ok=True)

    # List input FITS in cut directory matching the filter
    cut_dir = Path(directorio_beard_cut)
    fit_files = sorted(
        f.name for f in cut_dir.iterdir()
        if f.is_file() and re.search(fr"{re.escape(filter)}_.+\.fits$", f.name)
    )
    fit_files = np.array(fit_files)

    # Galaxy id from filename
    name_2 = name.split("_")[0]

    # PSF pathing
    path_to_psf = Path(dir_psf) / f"psf_{name_2}_{filter}.fits"

    # Copy targets
    direct_copy       = direct_copy_root / f"{name_2}_{filter}"
    direct_copy_model = scatter_root / f"{name_2}_{filter}"
    direct_copy.mkdir(parents=True, exist_ok=True)
    direct_copy_model.mkdir(parents=True, exist_ok=True)

    # Stamps and star selection dirs
    dir_quit_stars = Path(f"./Process_data/Subtract_stars/Quit_stars_{filter}") / f"Stars_{name_2}_{filter}"
    dir_stamps = Path(f"./Process_data/Subtract_stars/Stamps/Stamps_{name_2}_{filter}")
    if dir_stamps.exists():
        shutil.rmtree(dir_stamps)
    dir_stamps.mkdir(parents=True, exist_ok=True)

    # Image & copies
    path_image       = cut_dir / name
    path_image_copy = direct_copy / name

    # Segment and temp paths
    name_seg  = name.replace(".fits", "_segment.fits")
    path_complete = Path("./Process_data/Mask_data/Mask_segment") / name_seg
    name_temp = name.replace(".fits", "_temp.fits")
    path_temp = Path("./Process_data/Mask_data/Mask_segment") / name_temp

    return (
        fit_files,  # np.ndarray of filenames
        name_2,
        path_to_psf,
        direct_copy,
        direct_copy_model,
        dir_quit_stars,
        dir_stamps,
        path_image,
        path_image_copy,
        path_complete,
        path_temp,
    )