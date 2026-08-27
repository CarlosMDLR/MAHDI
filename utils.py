#==========================================
# Import libraries
#==========================================

# Standard library
import argparse
import glob
import hashlib
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
from astropy.table import Table
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# Visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
# Astronomy
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.wcs import WCS
import astropy.visualization as vis
from scipy.interpolate import RegularGridInterpolator
import astropy.units as u
from astropy.table import Table
# Machine learning
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Utilities
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Sequence, Union

# Deactivate warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#=============================================================
#
# Functions for masking the images
#
#=============================================================

def masks_maker_total_image(dir, hdu, noisechisel_params=None, segment_params=None):
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
        output_noisechisel = os.path.join(dir_noisechisel, name.replace(".fits", "_noisechisel.fits"))
        output_segment = os.path.join(dir_segment, name.replace(".fits", "_segment.fits"))
        
        # Default parameters if none provided
        if noisechisel_params is None:
            noisechisel_params = "--tilesize=20,20 --interpnumngb=5 --dthresh=0.05 --snminarea=2 --rawoutput"
        if segment_params is None:
            segment_params = "--tilesize=10,10 --interpnumngb=1 --gthresh=-10 --objbordersn=0 --minnumfalse=1"

        # Run astnoisechisel
        cmd_noisechisel = f"astnoisechisel {path} --hdu={str(hdu)} {noisechisel_params} --quiet --output={output_noisechisel} 2>/dev/null"
        os.system(cmd_noisechisel)

        # Run astsegment
        cmd_segment = f"astsegment {output_noisechisel} {segment_params} --quiet --output={output_segment} 2>/dev/null"
        os.system(cmd_segment)

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


def dataframe_to_fits(df, fits_path, hdu_index=0):
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
    data = df.to_numpy()
    
    if hdu_index == 0:
        # Standard behavior: Data is in the primary slot
        hdu = fits.PrimaryHDU(data)
        hdul = fits.HDUList([hdu])
    else:
        # Create an empty PrimaryHDU and put data in an ImageHDU extension
        primary_hdu = fits.PrimaryHDU() 
        extension_hdu = fits.ImageHDU(data)
        hdul = fits.HDUList([primary_hdu, extension_hdu])
        
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


def create_blank_mask(image_path, mask_path, profile_path):
    """
    Create an all-zero mask and masked image (no pixels removed).
    """
    print("⚠️ Creating blank mask (no masking applied).")

    imagen = fits.getdata(image_path)
    blank_mask = np.zeros_like(imagen)

    fits.PrimaryHDU(blank_mask).writeto(mask_path, overwrite=True)
    fits.PrimaryHDU(imagen).writeto(image_path, overwrite=True)

    # Save dummy radial profile file
    with open(profile_path.replace(".fits", "_warning.txt"), "w") as f:
        f.write("NoiseChisel/Segment failed: blank mask used.\n")

    print(f"✅ Blank mask saved: {mask_path}")

def masks_maker(name, profile, noisechisel_params=None, segment_params=None):
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

    # Define intermediate output filenames
    output_astarith    = name.replace(".fits", "_astarithmetic.fits")
    output_noisechisel = name.replace(".fits", "_noisechisel.fits")
    output_segment     = name.replace(".fits", "_segment.fits")

    # --- Default parameters if not provided
    if noisechisel_params is None:
        noisechisel_params = "--tilesize=10,10 --outliernumngb=5 --interpnumngb=1 --qthresh=0.5 --minnumfalse=1"
    if segment_params is None:
        segment_params = "--tilesize=10,10 --interpnumngb=1 --gthresh=-10 --objbordersn=0 --minnumfalse=1"

    # Step 1: Preprocess image with `astarithmetic` (set zeros to NaN)
    os.system(
        f"astarithmetic {name} {name} --quiet --output={output_astarith} 0.0 eq nan where -g1 2>/dev/null"
    )

    # Step 2: Run NoiseChisel to detect low-surface-brightness features
    os.system(
        f"astnoisechisel {output_astarith} {noisechisel_params} "
        f"--quiet --output={output_noisechisel} 2>/dev/null"
    )

    # Step 3: Segment the detections with `astsegment`
    os.system(
        f"astsegment {output_noisechisel} {segment_params} "
        f"--quiet --output={output_segment} 2>/dev/null"
    )

    # Step 4: Load segmentation map (HDU 3 usually contains object labels)
    data = fits.open(output_segment)
    objects = data[3].data
    clumps = data[2].data

    # Identify the label of the central pixel
    number = stats.mode(
        objects[int(len(objects) / 2), int(len(objects) / 2)],
        axis=None,
        nan_policy="omit",
    ).mode
    number_clump = stats.mode(
        clumps[int(len(clumps) / 2), int(len(clumps) / 2)],
        axis=None,
        nan_policy="omit",
    ).mode
    # Convert segmentation to binary mask
    objects[(objects == number) & ((clumps == number_clump) | (clumps == -1))] = 0
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
 



def _get_table_column_name(table: Table, requested_name: str) -> str:
    """Return the actual table column name using a case-insensitive lookup."""
    mapping = {name.lower(): name for name in table.colnames}
    try:
        return mapping[requested_name.lower()]
    except KeyError as exc:
        raise KeyError(
            f"Required column '{requested_name}' was not found. "
            f"Available columns: {table.colnames}"
        ) from exc


def load_gaia_ids_not_subtract(filename: Optional[str]) -> Set[int]:
    """
    Read Gaia DR3 source IDs that must be protected from subtraction.

    The file may contain one or several IDs per line, separated by whitespace,
    commas or semicolons. Text following ``#`` is treated as a comment.
    """
    if filename is None:
        return set()

    path = Path(filename).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Gaia exclusion file not found: {path}")

    protected_ids: Set[int] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        content = raw_line.split("#", 1)[0].strip()
        if not content:
            continue

        for token in re.split(r"[\s,;]+", content):
            if not token:
                continue
            try:
                protected_ids.add(int(token))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid Gaia source_id '{token}' in {path}, line {line_number}."
                ) from exc

    return protected_ids


def _selected_stars_hash(ra: np.ndarray, dec: np.ndarray, magnitude: np.ndarray) -> str:
    """Build a reproducible hash identifying a selected-star catalog."""
    values = np.column_stack(
        (
            np.asarray(ra, dtype=np.float64),
            np.asarray(dec, dtype=np.float64),
            np.asarray(magnitude, dtype=np.float64),
        )
    )
    return hashlib.sha256(values.tobytes()).hexdigest()


def add_gaia_ids_and_filter_protected_stars(
    ruta_star: Union[str, Path],
    name_2: str,
    filter: str,
    gaia_ids_not_subtract: Optional[str] = None,
    gaia_match_radius_arcsec: float = 1.0,
    gaia_match_max_mag_diff: float = 0.1,
    gaia_cache_dir: Union[str, Path] = "./Process_data/Gaia_catalogs",
) -> None:
    """
    Crossmatch the selected-star FITS table with Gaia DR3 and protect requested IDs.

    A compact table containing the selected coordinates is uploaded to Gaia TAP+.
    Gaia candidates inside ``gaia_match_radius_arcsec`` are returned, and the best
    candidate for every selected star is chosen locally using angular separation
    and compatibility in Gaia G magnitude. The resulting ``source_id`` and
    ``gaia_match_separation_arcsec`` columns are written into ``ruta_star``.

    Stars whose ``source_id`` occurs in ``gaia_ids_not_subtract`` are removed from
    the selection table before stamps, radial profiles or PSF subtraction are made.
    """
    if gaia_match_radius_arcsec <= 0:
        raise ValueError("gaia_match_radius_arcsec must be greater than zero.")
    if gaia_match_max_mag_diff < 0:
        raise ValueError("gaia_match_max_mag_diff cannot be negative.")

    ruta_star = Path(ruta_star)
    selected = Table.read(ruta_star, hdu=1)

    ra_name = _get_table_column_name(selected, "ra")
    dec_name = _get_table_column_name(selected, "dec")
    mag_name = _get_table_column_name(selected, "phot_g_mean_mag")

    ra = np.asarray(selected[ra_name], dtype=float)
    dec = np.asarray(selected[dec_name], dtype=float)
    selected_mag = np.asarray(selected[mag_name], dtype=float)
    n_selected = len(selected)

    if n_selected == 0:
        selected["source_id"] = np.array([], dtype=np.int64)
        selected["gaia_match_separation_arcsec"] = np.array([], dtype=np.float64)
        selected.write(ruta_star, overwrite=True)
        print(f"Gaia crossmatch for {name_2}_{filter}: no selected stars.")
        return
    ###We do this just if he have to remove stars from subtraction
    if gaia_ids_not_subtract is not None:
        upload_table = Table()
        upload_table["selected_index"] = np.arange(n_selected, dtype=np.int64)
        upload_table["selected_ra"] = ra
        upload_table["selected_dec"] = dec
        upload_table["selected_mag"] = selected_mag

        catalog_hash = _selected_stars_hash(ra, dec, selected_mag)
        cache_dir = Path(gaia_cache_dir).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{name_2}_{filter}_gaia_dr3_candidates.fits"

        candidates = None
        if cache_path.is_file():
            try:
                cached = Table.read(cache_path)
                cached_hash = cached.meta.get("SELHASH")
                cached_radius = float(cached.meta.get("RADASEC", np.nan))
                if cached_hash == catalog_hash and np.isclose(
                    cached_radius, gaia_match_radius_arcsec
                ):
                    candidates = cached
                    print(f"Using cached Gaia DR3 candidates: {cache_path}")
            except Exception as exc:
                print(f"Ignoring invalid Gaia cache {cache_path}: {exc}")

        if candidates is None:
            try:
                from astroquery.gaia import Gaia
            except ImportError as exc:
                raise ImportError(
                    "The Gaia crossmatch requires astroquery. Install it with "
                    "'python -m pip install astroquery'."
                ) from exc

            radius_deg = gaia_match_radius_arcsec / 3600.0
            query = f"""
                SELECT
                    u.selected_index,
                    u.selected_ra,
                    u.selected_dec,
                    u.selected_mag,
                    g.source_id,
                    g.ra AS gaia_ra,
                    g.dec AS gaia_dec,
                    g.phot_g_mean_mag AS gaia_phot_g_mean_mag,
                    DISTANCE(
                        POINT('ICRS', u.selected_ra, u.selected_dec),
                        POINT('ICRS', g.ra, g.dec)
                    ) * 3600.0 AS separation_arcsec
                FROM TAP_UPLOAD.selected_stars AS u
                JOIN gaiadr3.gaia_source AS g
                ON 1 = CONTAINS(
                    POINT('ICRS', g.ra, g.dec),
                    CIRCLE('ICRS', u.selected_ra, u.selected_dec, {radius_deg:.15f})
                )
            """

            print(
                f"Querying Gaia DR3 for {n_selected} selected stars "
                f"({name_2}_{filter})..."
            )
            job = Gaia.launch_job_async(
                query=query,
                upload_resource=upload_table,
                upload_table_name="selected_stars",
                verbose=False,
            )
            candidates = job.get_results()
            candidates.meta["SELHASH"] = catalog_hash
            candidates.meta["RADASEC"] = float(gaia_match_radius_arcsec)
            candidates.write(cache_path, overwrite=True)

        source_ids = np.full(n_selected, -1, dtype=np.int64)
        match_separations = np.full(n_selected, np.nan, dtype=np.float64)

        if len(candidates) > 0:
            candidate_index_name = _get_table_column_name(candidates, "selected_index")
            candidate_source_name = _get_table_column_name(candidates, "source_id")
            candidate_mag_name = _get_table_column_name(candidates, "gaia_phot_g_mean_mag")
            candidate_sep_name = _get_table_column_name(candidates, "separation_arcsec")

            candidate_indices = np.asarray(candidates[candidate_index_name], dtype=int)
            candidate_sources = np.asarray(candidates[candidate_source_name], dtype=np.int64)
            candidate_mags = np.asarray(candidates[candidate_mag_name], dtype=float)
            candidate_seps = np.asarray(candidates[candidate_sep_name], dtype=float)

            for selected_index in range(n_selected):
                rows = np.flatnonzero(candidate_indices == selected_index)
                if rows.size == 0:
                    continue

                magnitude_differences = np.abs(
                    candidate_mags[rows] - selected_mag[selected_index]
                )
                valid = np.isfinite(candidate_seps[rows]) & (
                    candidate_seps[rows] <= gaia_match_radius_arcsec
                )

                if np.isfinite(selected_mag[selected_index]):
                    valid &= np.isfinite(magnitude_differences)
                    valid &= magnitude_differences <= gaia_match_max_mag_diff

                valid_rows = rows[valid]
                if valid_rows.size == 0:
                    continue

                valid_mag_diff = np.abs(
                    candidate_mags[valid_rows] - selected_mag[selected_index]
                )
                valid_mag_diff[~np.isfinite(valid_mag_diff)] = np.inf
                order = np.lexsort((valid_mag_diff, candidate_seps[valid_rows]))
                best_row = valid_rows[order[0]]

                source_ids[selected_index] = candidate_sources[best_row]
                match_separations[selected_index] = candidate_seps[best_row]

        if "source_id" in selected.colnames:
            selected.remove_column("source_id")
        if "gaia_match_separation_arcsec" in selected.colnames:
            selected.remove_column("gaia_match_separation_arcsec")
        selected["source_id"] = source_ids
        selected["gaia_match_separation_arcsec"] = match_separations

        protected_ids = load_gaia_ids_not_subtract(gaia_ids_not_subtract)
        protected_mask = np.isin(source_ids, np.fromiter(protected_ids, dtype=np.int64)) \
            if protected_ids else np.zeros(n_selected, dtype=bool)

        n_matched = int(np.count_nonzero(source_ids >= 0))
        n_unmatched = n_selected - n_matched
        n_protected = int(np.count_nonzero(protected_mask))

        selected = selected[~protected_mask]
        selected.write(ruta_star, overwrite=True)

        print(f"Gaia crossmatch for {name_2}_{filter}:")
        print(f"  selected stars: {n_selected}")
        print(f"  matched stars: {n_matched}")
        print(f"  unmatched stars: {n_unmatched}")
        print(f"  protected stars removed: {n_protected}")
        print(f"  stars retained for subtraction: {len(selected)}")
    else:
        print(f"Gaia crossmatch for {name_2}_{filter}: no stars were protected from subtraction.")
        source_ids = np.full(n_selected, -1, dtype=np.int64)
        match_separations = np.full(n_selected, np.nan, dtype=np.float64)
        selected["source_id"] = source_ids
        selected["gaia_match_separation_arcsec"] = match_separations
        selected.write(ruta_star, overwrite=True)

def prepare_star_selection(
    filter: str,
    name: str,
    dir: str,
    hdu: int,
    mag_inf_lim: float,
    mag_sup_lim: float,
    min_dist: float,
    gaia_ids_not_subtract: Optional[str] = None,
    gaia_match_radius_arcsec: float = 1.0,
    gaia_match_max_mag_diff: float = 0.1,
    gaia_cache_dir: Union[str, Path] = "./Process_data/Gaia_catalogs",
):
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
            f"--hdu={str(hdu)}",
            "--quiet",
            f"--magnituderange={mag_inf_lim},{mag_sup_lim}",
            f"--mindistdeg={min_dist}",
            f"--output={ruta_out_quit}",
        ],
        check=True,
    )

    # Enrich the selected catalog with Gaia DR3 source IDs and remove protected stars.
    add_gaia_ids_and_filter_protected_stars(
        ruta_star=ruta_out_quit,
        name_2=name_2,
        filter=filter,
        gaia_ids_not_subtract=gaia_ids_not_subtract,
        gaia_match_radius_arcsec=gaia_match_radius_arcsec,
        gaia_match_max_mag_diff=gaia_match_max_mag_diff,
        gaia_cache_dir=gaia_cache_dir,
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
    hdu: int,
    ruta_out_norm_base: Path,
    norm_dir: Path,
    filter: str,
    name_2: str,
    masks_maker,           # function: (path: str|Path, radial_out: str|Path) -> np.ndarray (or None)
    extract_number,        # function: (filename: str) -> int
    crop_size_pix: Tuple[int, int] = (500, 500),
    noisechisel_params: str = None,
    segment_params: str = None
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
    source_ids = np.asarray(table["source_id"], dtype=np.int64)
    gaia_match_separations = np.asarray(
        table["gaia_match_separation_arcsec"], dtype=float
    )

    # Generate per-star crops with astcrop
    for idx, (ra, dec) in enumerate(zip(ra_stars, dec_stars), start=0):
        out_stamp = norm_dir / f"{ruta_out_norm_base.name}_{idx}.fits"
        subprocess.run(
            [
                "astcrop",
                str(ruta_gal),
                f"--hdu={str(hdu)}",
                "--mode=wcs",
                f"--center={ra},{dec}",
                "--widthinpix",
                "--quiet",
                f"--width={crop_size_pix[0]},{crop_size_pix[1]}",
                f"--output={out_stamp}",
            ],
            check=True,
            stderr=subprocess.DEVNULL,
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

    for file in tqdm(files_stars, desc="Making inner mask and profiles of the stars"):
        try:
            path = norm_dir / file
            radial = directorio_radial_profiles / file.replace(".fits", "_radial_profile.fits")

            # Build mask and radial profile
            masks_maker(str(path), str(radial), noisechisel_params, segment_params)
            
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
                os.remove(str(path).replace(".fits", "_noisechisel.fits"))
                os.remove(str(path).replace(".fits", "_segment.fits"))
            except Exception:
                pass
            try:
                os.remove(str(path))
            except Exception:
                pass

            quitar.append(contador)
            contador += 1
            continue

    return (
        np.array(radius),
        np.array(counts),
        ra_stars,
        dec_stars,
        mags_stars,
        source_ids,
        gaia_match_separations,
        quitar,
    )

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
    source_ids: np.ndarray,
    gaia_match_separations: np.ndarray,
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
    source_ids = np.delete(source_ids, quitar)
    gaia_match_separations = np.delete(gaia_match_separations, quitar)
    flat_points = np.array(flat_points)

    return (
        mags_stars,
        ra_stars,
        dec_stars,
        source_ids,
        gaia_match_separations,
        flat_points,
        preserve_index,
    )

def loadTableForGettingMaskRadius(filename):
    mag_grid = None
    sb_grid = None

    with open(filename) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if "# mag_grid" in line:
            mag_grid = np.array(list(map(float, lines[i+1][2:].split())))
        
        elif "# sb_grid" in line:
            sb_grid = np.array(list(map(float, lines[i+1][2:].split())))
            
        if mag_grid is not None and sb_grid is not None:
            break

    grid = np.loadtxt(filename)
    return grid, mag_grid, sb_grid
###Functions for getting magRadioRelation
def get_psfprofile_filename(dir_psf, name, filter):
    name_2 = name.split("_")[0]
    path_to_psfProfile = Path(dir_psf) / f"psf_profile_{name_2}_{filter}.fits"
    return path_to_psfProfile

def readProfile(model):
    radius = []
    profile = []
    with open(model) as f:
        for line in f:
            splittedLine = line.split()
            if (not np.isnan(float(splittedLine[1]))):
                radius.append(float(splittedLine[0]))
                profile.append(float(splittedLine[1]))
    return(np.array(radius), np.array(profile))

def readProfileFits(filename):
    data = Table.read(filename)
    return data['RADIUS'], data['MEAN']

def truncateProfile(radius,profile,truncatePx):
    index = -1
    for i, value in enumerate(radius):
        if (value > truncatePx):
            index = i
            break
    return(radius[:index], profile[:index])

def integrateProfile2D(R_psf,I_psf,integrationLimit=None):
    R_psf = np.asarray(R_psf)
    I_psf = np.asarray(I_psf)

    #Remove NaN/Inf
    mask = np.isfinite(R_psf) & np.isfinite(I_psf)
    R_psf = R_psf[mask]
    I_psf = I_psf[mask]

    #Sort by radius
    idx = np.argsort(R_psf)
    R_psf = R_psf[idx]
    I_psf = I_psf[idx]

    #Apply integration limit
    if integrationLimit is not None:
        mask = R_psf <= integrationLimit
        R_psf = R_psf[mask]
        I_psf = I_psf[mask]
    
    integrand = I_psf * R_psf
    I_tot = 2*np.pi*cumulative_trapezoid(integrand, R_psf, initial=0)

    return I_tot[-1]

# NOTE
# 1.- We precompute the inverse PSF profile r(I) to efficiently get the radius at some I. Typical approach
# would be having I(r) and looking for the r that I(r) - X ~=, doing it directly. Inverting is faster.

# Since stellar magnitude only linearly scales the PSF flux, instead of scaling the
# PSF for each star we equivalently scale the surface brightness threshold. This lets
# us solve I_PSF(r) = I_threshold / s(m) using the original PSF shape, avoiding repeated
# PSF scaling and interpolation. The result is a fast lookup of radius as a function
# of magnitude and SB threshold.


def build_inverse_psf(radius, profile):
    mask = profile > 0
    r = radius[mask]
    I = profile[mask]

    # Interpolate radius as function of intensity. It is reversed because interp1d 
    # expects the 'x' dimension to always increase
    return interp1d(I[::-1], r[::-1], bounds_error=False, fill_value=np.nan)

def build_radius_table(radius, profile, mag_grid, sb_grid, zp, pixelScale):
    inv_psf = build_inverse_psf(radius, profile) # We invert the psf once. We get r(I)
    flux_model = integrateProfile2D(radius, profile)

    table = np.zeros((len(mag_grid), len(sb_grid)))

    for i, m in enumerate(mag_grid):
        s = magnitudeToScaleFactor(m, flux_model, zp) # We get the scale factor for the desired magnitude

        for j, sb in enumerate(sb_grid):
            thresholdCounts = sb_to_counts(sb, zp, pixelScale)
            table[i, j] = inv_psf(thresholdCounts / s) # We escale the threshold (instead of scaling the psf) and get the corresponding r
    return table

def magnitudeToScaleFactor(mag, modelFlux, zp=22.5):
    targetFlux = 10**(-0.4 * (mag - zp))
    return targetFlux / modelFlux

def sb_to_counts(sb, zp, pixelScale):
    return 10**(-0.4 * (sb - zp)) * (pixelScale**2)

def save_radius_table(filename, grid, mag_grid, sb_grid):
    with open(filename, "w") as f:

        f.write("# mag_grid\n")
        f.write("# " + " ".join(map(str, mag_grid)) + "\n")

        f.write("# sb_grid\n")
        f.write("# " + " ".join(map(str, sb_grid)) + "\n")

        np.savetxt(f, grid)

def generateSbMagTable(
    dir_psf: str,
    name: str,
    filter: str,
    output_filename: str,
    px_scale: float,
    zp: float,
):
    """
    Generate a table of surface brightness vs. radius for a given PSF model.
    The table is saved to `output_filename`.
    
    For masking the stars in our images, we need a radius for each star. This radius has to be related
    to the magnitude of the star.
    This relation is obtained via the PSF. We take the psf, we scale it to the magnitude of the star and
    we define a surface brightness up to which we mask the stars.
    This function receibes a PSF Model and explores a set of magnitudes for getting a relation (thus not having
    to escalate and integrate the model when masking each of the stars).
    """
    cutProfile_px = 3000 #Hardcoded
    psf_filename = get_psfprofile_filename(dir_psf, name, filter)
    radius,profile = readProfileFits(psf_filename)
    radius, profile = truncateProfile(radius, profile, cutProfile_px)
    originalModelFlux = integrateProfile2D(radius, profile)
    mag_grid = np.linspace(5,18,200)
    sb_grid = np.linspace(20,26,200)
    grid = build_radius_table(radius, profile, mag_grid, sb_grid, zp, px_scale)
    save_radius_table(output_filename, grid, mag_grid, sb_grid)

def fit_ransac_and_build_rings(
    mags_stars: np.ndarray,
    flat_points: np.ndarray,
    px_scale: float,
    ruta_gal: str,
    ra_stars: np.ndarray,
    dec_stars: np.ndarray,
    hdu_ind: int,
    name: str,
    filter: str,
    dir_psf: str,
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
    # r_min_ring = 2.0 * r_sat_sat / px_scale
    # r_max_ring = 4.0 * r_sat_sat / px_scale

    # Define and read the table with the relation between surface brightness and radius for the used psf
    magnitudeSbRadiusTableFile=f"./Process_data/Mask_data/magnitude_sb_radius_table_{name.split('_')[0]}_{filter}.dat"
    generateSbMagTable(dir_psf, name, filter, magnitudeSbRadiusTableFile, px_scale, zp=22.5)
    grid, mag_grid, sb_grid = loadTableForGettingMaskRadius(magnitudeSbRadiusTableFile)
    calculateMaskRadius = RegularGridInterpolator((mag_grid, sb_grid), grid, bounds_error=False, fill_value=np.nan)

    innerSurfaceBrightnessForMatching = 20.5
    outerSurfaceBrightnessForMatching = 23
    pts_bright = np.column_stack((mags_stars, np.full_like(mags_stars, innerSurfaceBrightnessForMatching)))
    pts_faint = np.column_stack((mags_stars, np.full_like(mags_stars, outerSurfaceBrightnessForMatching)))
    r_min_ring = calculateMaskRadius(pts_bright)
    r_max_ring = calculateMaskRadius(pts_faint)


    # Get WCS from galaxy FITS (HDU) and convert RA/DEC to pixel coords
    with fits.open(ruta_gal) as hdu:
        wcs = WCS(hdu[int(hdu_ind)].header)

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
    source_ids: np.ndarray,
    gaia_match_separations: np.ndarray,
) -> None:
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
                print(f"Rejected: {file_name}")
            else:
                print(f"Preserved: {file_name}")

    # Escribir FITS revisado con tabla actualizada
    with fits.open(ruta_star) as hdul:
        primary_header = hdul[0].header.copy()
        table_extname = hdul[1].header.get("EXTNAME")

    ra_final = ra_stars[indices_to_keep]
    dec_final = dec_stars[indices_to_keep]
    phot_g_final = mags_stars[indices_to_keep]
    source_id_final = source_ids[indices_to_keep]
    gaia_separation_final = gaia_match_separations[indices_to_keep]

    data_combined = np.zeros(
        len(indices_finales),
        dtype=[
            ("ra", ">f8"),
            ("dec", ">f8"),
            ("phot_g_mean_mag", ">f4"),
            ("source_id", ">i8"),
            ("gaia_match_separation_arcsec", ">f8"),
            ("rmin_norm", ">i4"),
            ("rmax_norm", ">i4"),
            ("image_index", ">i4"),
        ],
    )
    data_combined["ra"] = ra_final
    data_combined["dec"] = dec_final
    data_combined["phot_g_mean_mag"] = phot_g_final
    data_combined["source_id"] = source_id_final
    data_combined["gaia_match_separation_arcsec"] = gaia_separation_final
    data_combined["rmin_norm"] = np.round(r_min_ring).astype(int)
    data_combined["rmax_norm"] = (np.round(r_max_ring).astype(int) + 1)
    data_combined["image_index"] = indices_finales

    table_hdu = fits.BinTableHDU(data_combined)
    if table_extname:
        table_hdu.name = table_extname

    new_hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=primary_header),
            table_hdu,
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
