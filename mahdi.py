#==========================================
# Import libraries
#==========================================

import argparse
from utils import masks_maker_total_image
from subtracting_stars import SubtractingStars
from typing import Tuple

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline: create masks and subtract stars using PSF modeling."
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing the FITS galaxy cutouts.",
    )
    parser.add_argument(
        "--dir-psf",
        required=True,
        help="Directory containing the PSF models, the path have to be like: dir_psf /psf_(gal_name)_(filter).fits",
    )
    parser.add_argument(
        "--hdu",
        type=int,
        default=0,
        help="HDU index for FITS files. Default: 0.",
    )
    parser.add_argument(
        "--psf-hdu",
        type=int,
        default=0,
        help="HDU index for PSF FITS files. Default: 0.",
    )
    parser.add_argument(
        "--filters",
        required=True,
        help="Comma-separated list of filters, e.g. 'g,r,i'.",
    )
    parser.add_argument(
        "--mag-inf-sub",
        type=float,
        required=True,
        help="Lower magnitude limit for star selection.",
    )
    parser.add_argument(
        "--mag-sup-sub",
        type=float,
        required=True,
        help="Upper magnitude limit for star selection.",
    )
    parser.add_argument(
        "--min-dist-sub",
        type=float,
        required=True,
        help="Minimum angular distance (in degrees) between stars.",
    )
    parser.add_argument(
        "--model-scatter",
        action="store_true",
        help="Enable scattered light field modeling. Default: False.",
    )
    parser.add_argument(
        "--px-scale",
        type=float,
        default=0.33,
        help="Pixel scale (arcsec/pixel). Default: 0.33.",
    )
    parser.add_argument(
        "--crop-size-pix", 
        type=Tuple[int,int], 
        default=(500,500), 
        help="Crop size around stars (pixels). Default: 500."
        )
    parser.add_argument(
        "--zp",
        type=float,
        default=22.5,
        help="Photometric zero point. Default: 22.5.",
    )
    parser.add_argument(
        "--noisechisel-params-full-image",
        type=str,
        default=None,
        help="Extra parameters for astnoisechisel for the full image of the galaxy, e.g. '--tilesize=30,30 --snminarea=3'."
    )
    parser.add_argument(
        "--segment-params-full-image",
        type=str,
        default=None,
        help="Extra parameters for astsegment for the full image of the galaxy, e.g. '--gthresh=-5 --minnumfalse=2'."
    )
    parser.add_argument(
        "--noisechisel-params",
        type=str,
        default=None,
        help="Extra parameters for astnoisechisel for the crops of the stars, e.g. '--tilesize=30,30 --snminarea=3'."
    )
    parser.add_argument(
        "--segment-params",
        type=str,
        default=None,
        help="Extra parameters for astsegment for the crops of the stars, e.g. '--gthresh=-5 --minnumfalse=2'."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 1) Create masks for all FITS images in the input directory
    masks_maker_total_image(args.dir, 
                            args.hdu,
                            noisechisel_params=args.noisechisel_params_full_image,
                            segment_params=args.segment_params_full_image,
                            )

    # 2) Run star selection
    SubtractingStars(
        filter_list=args.filters.split(","),
        dir=args.dir,
        dir_psf= args.dir_psf,
        hdu=args.hdu,
        psf_hdu=args.psf_hdu,
        mag_inf_lim=args.mag_inf_sub,
        mag_sup_lim=args.mag_sup_sub,
        min_dist=args.min_dist_sub,
        model_scatter=args.model_scatter,
        px_scale=args.px_scale,
        crop_size_pix=args.crop_size_pix,
        zp=args.zp,
        noisechisel_params=args.noisechisel_params,
        segment_params=args.segment_params,
    ).selector()

    # 3) Run star subtraction
    SubtractingStars(
        filter_list=args.filters.split(","),
        dir=args.dir,
        dir_psf= args.dir_psf,
        hdu=args.hdu,
        psf_hdu=args.psf_hdu,
        mag_inf_lim=args.mag_inf_sub,
        mag_sup_lim=args.mag_sup_sub,
        min_dist=args.min_dist_sub,
        model_scatter=args.model_scatter,
        px_scale=args.px_scale,
        crop_size_pix=args.crop_size_pix,
        zp=args.zp,
        noisechisel_params=args.noisechisel_params,
        segment_params=args.segment_params,
    ).subtractor()


if __name__ == "__main__":
    main()
