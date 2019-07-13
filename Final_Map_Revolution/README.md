Hello,
The main program is found in compare_images.py.

The manual page:
  usage: compare_images.py [-h] --synth_img SYNTH_IMG --real_img REAL_IMG
                           [--max_iters MAX_ITERS]
                           [--ransac_reproj_threshold RANSAC_REPROJ_THRESHOLD]
                           [--step_size STEP_SIZE] --out_dir OUT_DIR
                           [--more_details] [--color_the_change]

  This program takes two images of a rearranged scene.
  It returns a mask (over the real_img) deciding what were the changes.

  optional arguments:
    -h, --help            show this help message and exit
    --synth_img SYNTH_IMG
                          A synthetic image from the mesh.
    --real_img REAL_IMG   A real image from the same scene.
    --max_iters MAX_ITERS
                          The maximal number of iterations Ransac will do,
                          maximum 2000
    --ransac_reproj_threshold RANSAC_REPROJ_THRESHOLD
                          Ransac reprojection error threshold, rangeR 0-10
    --step_size STEP_SIZE
                          The length of the comparison squares
    --out_dir OUT_DIR     Path to a directory to put the result in.
    --more_details        Generates the aligned synthetic image and the colored
                          real image
    --color_the_change    Color the changed portion on the real image.
                          Otherwise, color the unchanged.

Command line example:
   python compare_images.py --synth_img frame30synth.png  --real_img frame30real.png --out_dir my_out_dir/  
