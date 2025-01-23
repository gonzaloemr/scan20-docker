import numpy as np
from scipy.ndimage import label


def make_brats_segmentation(target_probs, threshold=50):
    """
    Generate a segmentation mask for brain tumor regions based on probability maps.

    This function is specifically designed for the BraTS dataset and processes
    probability maps for different tumor regions to create a segmentation mask with
    labeled tumor classes.

    Args:
        target_probs (numpy.ndarray): A 4D array (e.g., `[3, H, W, D]`) where each channel
            contains probability maps for different tumor regions:
                - Channel 0: Enhancing tumor (ET) [4]
                - Channel 1: Tumor core (TC) [1,4]
                - Channel 2: Whole tumor (WT) [1,2,4]
        threshold (int, optional): A threshold value (default is 50) to classify regions
            as part of a tumor. Probabilities above this value (interpreted as percentages,
            e.g., 50% = 50) are included in the segmentation.

    Returns:
        numpy.ndarray: A 3D array of the same spatial dimensions as the input probability maps
        containing the segmentation mask with the BraTS labels:
                - 0: Background (no tumor)
                - 1: Necrotic core
                - 2: Peritumoral edema
                - 4: Enhancing tumor
    """

    # Position 0: [4] -> 4
    # Position 1: [1,4] -> 1
    # Position 2: [1,2,4] -> 2
    seg = (target_probs[2] > threshold).astype(np.uint8) * 2
    seg[target_probs[1] > threshold] = 1
    seg[target_probs[0] > threshold] = 4

    return seg


def postprocess_brats_segmentation(
    seg,
    ensemble,
    flair,
    t1,
    low_core_1=75,
    low_core_2=55,
    low_enhancing=80,
    low_whole=90,
    size=100,
    bleed_size=None,
):
    """
    Postprocess a BraTS segmentation to refine and validate tumor regions.

    This function refines a segmentation mask produced for the BraTS dataset by
    applying additional logic and thresholds to improve segmentation quality,
    validate tumor compartments, and handle edge cases like missing tumor or core
    regions.

    Args:
        seg (numpy.ndarray): Initial segmentation mask.
        ensemble (numpy.ndarray): Probability maps for tumor regions, with channels:
            - Channel 0: Enhancing tumor (ET)
            - Channel 1: Tumor core (TC)
            - Channel 2: Whole tumor (WT)
        flair (numpy.ndarray): FLAIR MRI image for tissue contrast.
        t1 (numpy.ndarray): T1-weighted MRI image for tissue contrast.
        low_core_1 (int, optional): Threshold for low tumor core certainty. Default is 75.
        low_core_2 (int, optional): Threshold for very low tumor core certainty. Default is 55.
        low_enhancing (int, optional): Threshold for low enhancing tumor certainty. Default is 80.
        low_whole (int, optional): Threshold for low whole tumor certainty. Default is 90.
        size (int, optional): Minimum size of enhancing tumor regions to retain (in voxels). Default is 100.
        bleed_size (int, optional): Minimum size for potential bleed regions. Default is None.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            - Refined segmentation mask with labels:
                - 0: Background (no tumor)
                - 1: Necrotic core
                - 2: Peritumoral edema
                - 4: Enhancing tumor
            - Uncertainty map indicating regions with low confidence.
    """

    new_seg = np.copy(seg)

    bleed_threshold = np.percentile(flair[flair > 0], 10)
    tissue_threshold = np.percentile(t1[t1 > 0], 5)
    potential_bleed = np.logical_and(flair < bleed_threshold, t1 > tissue_threshold)
    potential_bleed_seg = np.logical_and(
        np.logical_and(ensemble[:, :, :, 2] < 50, ensemble[:, :, :, 2] > 5),
        potential_bleed,
    )
    if bleed_size is not None:
        if np.sum(potential_bleed_seg) > bleed_size:
            print("potential missed bleed")
            new_seg[
                np.logical_and(
                    np.logical_and(potential_bleed, ensemble[:, :, :, 2] > 25),
                    new_seg == 0,
                )
            ] = 2
            new_seg[
                np.logical_and(
                    np.logical_and(potential_bleed, ensemble[:, :, :, 1] > 25),
                    new_seg != 4,
                )
            ] = 1

    tumor_in_this_case = np.logical_or(
        ensemble[:, :, :, 2] > 50, ensemble[:, :, :, 2] > 50
    )
    mean_tumor_certainty = np.nan_to_num(
        np.mean(ensemble[:, :, :, 2][tumor_in_this_case]), 0
    )

    adjusted_tumor_threshold = 50
    adjusted_core_threshold = 50
    adjusted_enhancing_threshold = 50

    if mean_tumor_certainty < low_whole:
        print("low tumor certainty")
        adjusted_tumor_threshold = 15
        new_seg[
            np.logical_and(
                ensemble[:, :, :, 2] > adjusted_tumor_threshold, new_seg == 0
            )
        ] = 2
        print(f"ill-segmented tumor, adjusting threshold {adjusted_tumor_threshold}")

    labelmap, num_labels = label(new_seg > 0)

    core_in_this_case = np.logical_or(
        ensemble[:, :, :, 1] > 50, ensemble[:, :, :, 0] > 50
    )
    mean_core_certainty = np.nan_to_num(
        np.mean(ensemble[:, :, :, 1][core_in_this_case]), 0
    )

    enhancing_in_this_case = np.logical_or(
        ensemble[:, :, :, 0] > 50, ensemble[:, :, :, 0] > 50
    )
    mean_enhancing_uncertainty = np.nan_to_num(
        np.mean(ensemble[:, :, :, 1][core_in_this_case]), 1
    )
    # print(mean_core_certainty)

    if np.sum(new_seg == 4) < size:
        new_seg[new_seg == 4] = 1
        print(f"enhancing less than {size} mm3: deleting compartment")
        enhancing_present = False
    else:
        enhancing_present = True

    if mean_tumor_certainty < low_whole:
        tumor_thresh = 50
    else:
        tumor_thresh = 65
    for y in range(num_labels):
        if (
            np.median(ensemble[:, :, :, 2][labelmap == y + 1]) < tumor_thresh
        ) and num_labels > 1:
            new_seg[labelmap == y + 1] = 0
        else:
            if mean_enhancing_uncertainty < low_enhancing and enhancing_present:
                print("low enhancing certainty")
                adjusted_enhancing_threshold = 5
                new_seg[
                    np.logical_and(
                        ensemble[:, :, :, 0] > adjusted_enhancing_threshold,
                        labelmap == y + 1,
                    )
                ] = 4
                print(
                    f"ill-segmented enhancement, adjusting threshold {adjusted_enhancing_threshold}"
                )

            labelmap_enhancing, num_labels_enhancing = label(
                (new_seg * (labelmap == y + 1).astype(np.uint8)) == 4
            )
            for w in range(num_labels_enhancing):
                if np.sum(labelmap_enhancing == w + 1) < 10:
                    new_seg[labelmap_enhancing == w + 1] = 1
                    # print(f'enhancing tumor, deleted {np.sum(labelmap_enhancing == w+1)} voxels (too small)')

            labelmap_core, num_labels_core = label(
                np.logical_or(
                    new_seg * (labelmap == y + 1).astype(np.uint8) == 1,
                    new_seg * (labelmap == y + 1).astype(np.uint8) == 4,
                )
            )
            # print(np.sum(ensemble[:,:,:,1][labelmap == y+1]>80), np.sum(ensemble[:,:,:,1][labelmap == y+1]>10) - np.sum(ensemble[:,:,:,1][labelmap == y+1]>80))
            if mean_core_certainty >= low_core_1:
                for z in range(num_labels_core):
                    if (
                        np.median(ensemble[:, :, :, 1][labelmap_core == z + 1]) < 60
                        and np.sum(new_seg[labelmap_core == z + 1] == 4) == 0
                    ):
                        new_seg[labelmap_core == z + 1] = 2
                        # print(f'core tumor, deleted {np.sum(labelmap_core == z+1)} voxels (too uncertain, no enhancing)')

            if mean_core_certainty < low_core_1:
                print("low core certainty")
                adjusted_core_threshold = 5
                if mean_core_certainty < low_core_2:
                    adjusted_core_threshold = 1
                # if 2*np.sum(ensemble[:,:,:,1][labelmap == y+1]>50)< np.sum(ensemble[:,:,:,1][labelmap == y+1]>25):
                new_seg[
                    np.logical_and(
                        np.logical_and(
                            ensemble[:, :, :, 1] > adjusted_core_threshold,
                            labelmap == y + 1,
                        ),
                        new_seg != 4,
                    )
                ] = 1
                print(
                    f"ill-segmented core, adjusting threshold {adjusted_core_threshold}"
                )

    # Was the whole tumor removed?  something went wrong with the postprocessing, reset!

    if np.sum(new_seg > 0) < 100:
        print("tumor missing, resetting segmentation")
        new_seg = np.copy(seg)

    # Is there still no segmentation?  the we missing the tumor!  Look for it

    if np.sum(new_seg > 0) < 100:
        for emergencythreshold in [45, 40, 35, 30, 25, 20, 15, 10, 5]:
            if np.sum(ensemble[:, :, :, 2] > emergencythreshold) > 1000:
                print(f"tumor found at threshold {emergencythreshold}")
                adjusted_tumor_threshold = emergencythreshold
                new_seg[(ensemble[:, :, :, 2] > emergencythreshold)] = 1
                break

    # Is there still no core? core must be the whole tumor

    if np.sum(np.logical_or(new_seg == 1, new_seg == 4)) < 10:
        print("core missing, setting to whole tumor")
        new_seg[(new_seg == 2)] = 1

    labelmap, num_labels = label(new_seg > 0)

    labelmap_core, num_labels_core = label(np.logical_or(new_seg == 1, new_seg == 4))

    print(f"{num_labels} tumor compartments")

    print(f"{num_labels_core} core compartments")

    in_tumor = new_seg > 0
    out_of_tumor = np.logical_not(in_tumor)
    if adjusted_tumor_threshold != 50:
        ensemble[:, :, :, 2][in_tumor] = np.clip(
            ensemble[:, :, :, 2][in_tumor] + (50 - adjusted_tumor_threshold), 0, 100
        )
        ensemble[:, :, :, 2][out_of_tumor] = np.clip(
            ensemble[:, :, :, 2][out_of_tumor] * 50 / adjusted_tumor_threshold, 0, 100
        )

    in_core = np.logical_or(new_seg == 1, new_seg == 4)
    out_of_core = np.logical_not(in_core)
    if adjusted_core_threshold != 50:
        ensemble[:, :, :, 1][in_core] = np.clip(
            ensemble[:, :, :, 1][in_core] + (50 - adjusted_core_threshold), 0, 100
        )
        ensemble[:, :, :, 1][out_of_core] = np.clip(
            ensemble[:, :, :, 1][out_of_core] * 50 / adjusted_core_threshold, 0, 100
        )

    in_enhancing = new_seg == 4
    out_of_enhancing = np.logical_not(in_enhancing)
    if adjusted_enhancing_threshold != 50:
        ensemble[:, :, :, 0][in_enhancing] = np.clip(
            ensemble[:, :, :, 0][in_enhancing] + (50 - adjusted_enhancing_threshold),
            0,
            100,
        )
        ensemble[:, :, :, 0][out_of_enhancing] = np.clip(
            ensemble[:, :, :, 0][out_of_enhancing] * 50 / adjusted_enhancing_threshold,
            0,
            100,
        )

    uncertainty = np.zeros_like(ensemble)

    uncertainty[:, :, :, 0][out_of_enhancing] = (
        ensemble[:, :, :, 0][out_of_enhancing] * 2
    )
    uncertainty[:, :, :, 0][in_enhancing] = 0

    uncertainty[:, :, :, 1][out_of_core] = ensemble[:, :, :, 1][out_of_core] * 2
    uncertainty[:, :, :, 1][in_core] = 0

    uncertainty[:, :, :, 2][out_of_tumor] = ensemble[:, :, :, 2][out_of_tumor] * 2
    uncertainty[:, :, :, 2][in_tumor] = 0

    return new_seg, np.clip(uncertainty, 0, 100)
