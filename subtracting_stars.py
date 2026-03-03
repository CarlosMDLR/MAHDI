#==========================================
# Import libraries
#==========================================

# Standard library
import os
import re
import numpy as np
import pandas as pd

# Utilities
import utils as ut
from astropy.io import fits


def select_stars(filter: str, name: str, dir: str, hdu:int, mag_inf_lim: float, mag_sup_lim: float, min_dist: float, px_scale: float, crop_size_pix, noisechisel_params, segment_params, dir_psf) -> None:
    """
    High-level pipeline for selecting stars around a galaxy, building radial profiles,
    fitting normalization rings, and producing a final revised FITS catalog.

    Parameters
    ----------
    filter : str
        Photometric filter (e.g., "g" or "r").
    name : str
        Name of the galaxy FITS file to process.
    dir : str
        Directory containing the galaxy cutouts.
    mag_inf_lim : float
        Lower limit for star magnitude selection.
    mag_sup_lim : float
        Upper limit for star magnitude selection.
    min_dist : float
        Minimum angular separation (in degrees) between stars.
    px_scale : float
        Pixel scale (arcsec/pixel), used for radius conversion and surface brightness.
    dir_psf : str
        Directory containing the PSF models, expected to have files named like:
        `psf_(gal_name)_(filter).fits`
        `psf_profile_(gal_name)_(filter).fits`
    Returns
    -------
    None
        The function writes updated FITS files to disk, no explicit return value.

    Workflow
    --------
    1. Prepare directories and run `astscript-psf-select-stars` to generate
       a FITS catalog of selected stars.
    2. Crop star stamps, build masks with NoiseChisel/Segment, and extract
       radial profiles.
    3. Determine the radii where profiles flatten (`flat_points`).
    4. Filter stars to ensure index consistency and apply sigma clipping.
    5. Fit a RANSAC regression between star magnitudes and flattening radii
       to derive normalization rings (`r_min_ring`, `r_max_ring`).
    6. Refine the star catalog, removing overlapping sources, and write out
       a revised FITS table with final positions, magnitudes, and ring sizes.
    """
    
    # Step 1: Prepare star-selection environment (directories, output paths)
    ruta_star, ruta_out_norm, ruta_gal, norm_dir, name_2, close_sources_dir = ut.prepare_star_selection(
        filter, name, dir, hdu, mag_inf_lim, mag_sup_lim, min_dist
    )
    
    # Step 2: Process stars → crop stamps, build masks, measure radial profiles
    radius, counts, ra_stars, dec_stars, mags_stars, _ = ut.process_selected_stars(
        ruta_star=ruta_star,
        ruta_gal=ruta_gal,
        hdu=hdu,
        ruta_out_norm_base=ruta_out_norm,
        norm_dir=norm_dir,
        filter=filter,
        name_2=name_2,
        masks_maker=ut.masks_maker,
        extract_number=ut.extract_number,
        crop_size_pix=crop_size_pix,
        noisechisel_params=noisechisel_params,
        segment_params=segment_params,
    )
    
    # Step 3: Find flattening radii for each profile
    flat_points = ut.find_flat_points(radius, counts)
    
    # Step 4: Filter stars (keep consistent indices) and clip flat radii
    mags_stars, ra_stars, dec_stars, flat_points, preserve_index = ut.filter_and_clip_stars(
        norm_dir=norm_dir,
        filter=filter,
        mags_stars=mags_stars,
        ra_stars=ra_stars,
        dec_stars=dec_stars,
        flat_points=flat_points,
    )
    
    # Step 5: Fit magnitude–radius relation with RANSAC, compute ring sizes
    r_min_ring, r_max_ring, x_positions, y_positions, magnitudes = ut.fit_ransac_and_build_rings(
        mags_stars, flat_points, px_scale, str(ruta_gal), ra_stars, dec_stars, hdu, name, filter, dir_psf
    )
    
    # Step 6: Finalize star catalog, move discarded sources, write revised FITS
    ut.finalize_sources_and_write_fits(
        x_positions=x_positions,
        y_positions=y_positions,
        r_min_ring=r_min_ring,
        r_max_ring=r_max_ring,
        magnitudes=magnitudes,
        preserve_index=preserve_index, 
        norm_dir=str(norm_dir),
        filter=filter,
        close_sources_dir=str(close_sources_dir),
        ruta_star=str(ruta_star),
        ra_stars=ra_stars,
        dec_stars=dec_stars,
        mags_stars=mags_stars,
    )
    return


def subtractor( filter: str, name: str, dir: str, dir_psf:str, hdu:int, psf_hdu:int, model_scatter_light, save_individual_scatter, px_scale: float, zp: float):
    """
    Perform PSF-based star subtraction from a galaxy FITS image using Gnuastro tools.
    Optionally also model and subtract the scattered light field.

    Parameters
    ----------
    filter : str
        Photometric filter (e.g., "g", "r").
    name : str
        Filename of the galaxy FITS image.
    directorio_beard_cut : str
        Directory containing the galaxy cutouts.
    model_scatter_light : bool
        Whether to model the extended scattered light field from stars.
    px_scale : float
        Pixel scale (arcsec/pixel), used for surface brightness calibration.
    zp : float
        Photometric zero point.

    Notes
    -----
    - Relies heavily on external Gnuastro tools (`asttable`, `astscript-psf-scale-factor`,
      `astscript-psf-subtract`, `astarithmetic`).
    - Temporary files are created and overwritten.
    - If `model_scatter_light=True`, also generates a combined scatter field map
      and its surface brightness equivalent.
    """
    
    # Print log header
    print("\n\n" + "=" * 60)
    print(f"\n Subtracting stars from {name}")
    print("\n" + "=" * 60)

    # Prepare all relevant paths and directories (helper function)
    archivos_fits_cut, name_2, path_to_psf, direct_copy, direct_copy_model, \
    dir_quit_stars, dir_stamps, path_image, path_image_copy, path_complete, path_temp \
    = ut.setup_subtractor_paths(filter, name, dir, dir_psf)

    # Make a working copy of the original image
    os.system(f'cp {path_image} {path_image_copy}')
    if hdu == 0:
        with fits.open(path_image_copy) as hdul:
            
            # Buscar el HDU que contiene datos reales
            data_hdu_index = None
            for i, hdu in enumerate(hdul):
                if hdu.data is not None:
                    data_hdu_index = i
                    break

            if data_hdu_index is None:
                raise ValueError("Error: Not found any HDU with data in the FITS file.")


            data   = hdul[data_hdu_index].data
            header = hdul[data_hdu_index].header
        primary_hdu = fits.PrimaryHDU()
        image_hdu = fits.ImageHDU(data=data, header=header)
        hdul_new = fits.HDUList([primary_hdu, image_hdu])
        hdul_new.writeto(path_image_copy, overwrite=True)
    # Path to the table with selected/revised stars
    path_stars = os.path.join(dir_quit_stars, name.replace('.fits', '_quit_revised.fits'))

    # Shell variables for position (center) and normalization radii
    center = "center=$(echo $ra $dec | awk '{printf \"%s,%s\", $1, $2}')"
    norm_factor = "normi=$(echo $rmin_norm $rmax_norm | awk '{printf \"%s,%s\", $1, $2}')" 

    # Temporary file for storing normalization factors per star
    normalization_dir = f"./Process_data/Subtract_stars/Normalization_factors/{name_2}_{filter}"
    if not os.path.isdir(normalization_dir):
        os.makedirs(normalization_dir)
    temp_file = normalization_dir + f"/{name_2}_{filter}_normalization_factors.txt"

    if model_scatter_light:
        # Directory for storing full scatter maps 
        directorio_full_scatter = f"./Process_data/Subtract_stars/Full_scatter_maps_{filter}/{name_2}_{filter}"
        if not os.path.isdir(directorio_full_scatter):
            os.makedirs(directorio_full_scatter)

        with fits.open(path_image_copy) as hdul:
            shape = hdul[1].shape

        # Initialisation of scatter map in ADUs
        ruta_full_scatter    = f"{directorio_full_scatter}/{name_2}_{filter}_full_scatter_field_{filter}.fits"
        ruta_full_scatterTmp = f"{directorio_full_scatter}/{name_2}_{filter}_full_scatter_field_{filter}_tmp.fits"
        df1 = pd.DataFrame(np.zeros(shape))
        ut.dataframe_to_fits(df1, ruta_full_scatter, hdu_index=1)
        os.system(f"cp {ruta_full_scatter} {ruta_full_scatterTmp}")

        # Loop over stars, compute PSF scale factor, subtract both model-only scatter image
        # and the full PSF from the galaxy image, accumulating results in a scatter map
        os.system(f"""
            asttable {path_stars} -cra,dec,rmin_norm,rmax_norm,image_index --sort phot_g_mean_mag | \\
            while read -r ra dec rmin_norm rmax_norm image_index phot_g_mean_mag; do
                echo "Subtracting star $ra $dec"; 
                currentScatterMap="./Process_data/Mask_data/Mask_segment/{name_2}-{filter}-scatter-img-$image_index-$ra-$dec.fits";
                {center};{norm_factor};
                scale=$(astscript-psf-scale-factor {path_image_copy} \
                    --mode=wcs --quiet\
                    --hdu=1 \
                    --psf={path_to_psf}  \
                    --psfhdu={str(psf_hdu)}  \
                    --quiet  \
                    --center=$ra,$dec \
                    --tmpdir=./Trash  \
                    --keeptmp  \
                    --nocentering  \
                    --sigmaclip=2,0.2 \
                    --normradii=$normi \
                    --segment={path_complete});
                scale_reduce=$(astarithmetic $scale 1 x --quiet);
                echo \"$image_index $scale_reduce $rmin_norm $rmax_norm $phot_g_mean_mag\" >> {temp_file};
                astscript-psf-subtract {path_image_copy}  \
                    --mode=wcs  \
                    --quiet  \
                    --hdu=1 \
                    --psf={path_to_psf} \
                    --psfhdu={str(psf_hdu)} \
                    --scale=$scale_reduce \
                    --center=$ra,$dec \
                    --modelonly \
                    --output=$currentScatterMap;
                astscript-psf-subtract {path_image_copy}  \
                    --mode=wcs \
                    --quiet  \
                    --hdu=1  \
                    --psf={path_to_psf}  \
                    --psfhdu={str(psf_hdu)}  \
                    --scale=$scale_reduce  \
                    --center=$ra,$dec  \
                    --output={path_temp};

                mv {path_temp} {path_image_copy};

                # Acummulate full scatter map
                astarithmetic $currentScatterMap {ruta_full_scatter} + -g1 --output {ruta_full_scatterTmp};
                mv {ruta_full_scatterTmp} {ruta_full_scatter} ;
                
                # Keep or remove the individual scatter map
                if [ "{str(save_individual_scatter).lower()}" = "true" ]; then 
                    mv $currentScatterMap {direct_copy_model};
                else
                    rm $currentScatterMap;
                fi
            done""")

        # Path for scatter map in Surface Brightness
        ruta_full_scatter_sb= f"{directorio_full_scatter}/{name_2}_{filter}_full_scatter_field_{filter}_sb.fits"

        # Convert scatter map to DataFrame, mask NaNs using the original galaxy,
        # and save both raw and surface-brightness calibrated versions
        df1 = ut.fits_to_dataframe(ruta_full_scatter)
        df2 = ut.fits_to_dataframe(str(path_image_copy))
        output_df = df1.copy()
        output_df[(df2.isna())] = np.nan
        ut.dataframe_to_fits(output_df, ruta_full_scatter)   
        output_df_sb = -2.5*np.log10(output_df)+zp+2.5*np.log10(px_scale**2)
        ut.dataframe_to_fits(output_df_sb, ruta_full_scatter_sb)  

        
    elif not model_scatter_light:
        # Same loop as above but subtract only the PSF (no scatter-field modeling)
        os.system(f"asttable {path_stars} -cra,dec,rmin_norm,rmax_norm,image_index --sort phot_g_mean_mag \
            | while read -r ra dec rmin_norm rmax_norm image_index phot_g_mean_mag; do\
                {center};{norm_factor};\
                scale=$(astscript-psf-scale-factor {path_image_copy} \
                    --mode=wcs --quiet\
                    --hdu=1 \
                    --psf={path_to_psf} \
                    --psfhdu={str(psf_hdu)} \
                    --quiet \
                    --center=$ra,$dec \
                    --tmpdir=./Trash \
                    --keeptmp \
                    --nocentering \
                    --sigmaclip=2,0.2 \
                    --normradii=$normi \
                    --segment={path_complete});\
                scale_reduce=$(astarithmetic $scale 1 x --quiet);\
                echo \"$image_index $scale_reduce $rmin_norm $rmax_norm $phot_g_mean_mag\" >> {temp_file};\
                astscript-psf-subtract {path_image_copy} \
                    --mode=wcs \
                    --quiet \
                    --hdu=1 \
                    --psf={path_to_psf} \
                    --psfhdu={str(psf_hdu)} \
                    --scale=$scale_reduce \
                    --center=$ra,$dec \
                    --output={path_temp};\
                mv {path_temp} {path_image_copy};\
            done")          

    return()


class SubtractingStars:
    def __init__(self, filter_list,dir, dir_psf, hdu, psf_hdu, mag_inf_lim, mag_sup_lim, min_dist,model_scatter, save_individual_scatter_maps, px_scale=0.33, crop_size_pix=(100,100), zp=22.5, noisechisel_params=None, segment_params=None):
        self.filter_list = filter_list
        self.dir = dir
        self.dir_psf = dir_psf
        self.hdu = hdu
        self.psf_hdu = psf_hdu
        self.mag_inf_lim = mag_inf_lim
        self.mag_sup_lim = mag_sup_lim
        self.model_scatter = model_scatter
        self.save_individual_scatter_maps = save_individual_scatter_maps
        self.min_dist = min_dist
        self.px_scale = px_scale
        self.zp = zp
        self.crop_size_pix=crop_size_pix
        self.noisechisel_params = noisechisel_params 
        self.segment_params = segment_params
    def selector(self):
        for filter in self.filter_list:    
            fits_files=[file for file in os.listdir(self.dir) if re.search(filter+'.*\.fits', file)]
            fits_files = np.sort(fits_files)
            for name in fits_files:
                try:
                    select_stars(filter, name, self.dir, self.hdu, self.mag_inf_lim, self.mag_sup_lim, self.min_dist, self.px_scale, self.crop_size_pix, self.noisechisel_params, self.segment_params,self.dir_psf)
                except Exception as e:
                    print("\n ############################################")
                    print(f"\n Failure in galaxy {name} {filter}: {e}")
                    print("\n ############################################")
                    continue
    def subtractor(self):

        for filter in self.filter_list:    
            fits_files=[file for file in os.listdir(self.dir) if re.search(filter+'.*\.fits', file)]
            fits_files = np.sort(fits_files)
            for name in fits_files:
                try:    
                    subtractor(filter, name, self.dir, self.dir_psf, self.hdu, self.psf_hdu, self.model_scatter, self.save_individual_scatter_maps, self.px_scale, self.zp)
                except Exception as e:
                    print("\n ############################################")
                    print(f"\n Failure in galaxy {name} {filter}: {e}")
                    print("\n ############################################")
                    continue
        