# scan20-docker
Docker for the BraTS 2020 scan20 submission

This is an adjusted and updated version of the Docker currently available for the BraTS 2020 scan20 submission.

The original docker can be found on the Brats docker page: <https://hub.docker.com/r/brats/scan-20>

The original docker is used in the BraTS-Toolkit: <https://github.com/neuronflow/BraTS-Toolkit>

The initial version of the adjusted Docker can be found here: <https://github.com/Svdvoort/scan20-docker>

## Changes

- Packages updated to Pytorch 2.4.1 and cuda 11.8 to work with newer GPUs.
- Option to provide an input path where the files are located instead of a fixed path.
- Option to run different modes of the pipeline, i.e., full mode or lite mode with and without data augmentation. 
- Only the code related to performing predictions with the pre-trained models was kept. 
- The code was documented. 
- The input files are renamed:

| Scan type  | Original filename  |  New filename |
|---|---|---|
| pre-contrast T1  | t1.nii.gz  | T1.nii.gz  |
| post-contrast T1  | t1ce.nii.gz  |  T1GD.nii.gz  |
| T2 | t2.nii.gz | T2.nii.gz |
| T2w-FLAIR | flair.nii.gz | FLAIR.nii.gz |