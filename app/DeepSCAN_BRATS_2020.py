# Libraries to be imported
import torch
import os
import cv2
import numpy as np
import nibabel as nib
import torch.nn as nn
from torch.autograd import Function
from batchgenerators.augmentations.utils import pad_nd_image
import argparse
from loss_functions import BCE_from_logits_focal, DiceLoss, Heteroscedastic_loss
from torch.nn import ModuleList
from collections import OrderedDict
from postprocess_brats_seg import (
    make_brats_segmentation,
    postprocess_brats_segmentation,
)

# Device to use to run the code
device = torch.device("cuda")

# device = torch.cuda.current_device()
print(f"Device is {device}")

# General configuration of the pipeline

# Network type
NETWORK_TYPE = "3D-to-2D"
# Set this to True, if you want all the 3D image files to be loaded into memory
IN_MEMORY = True
# Set this above zero to limit the number of training cases
NUM_TRAINING_CASES = 0
# Set this above zero to limit the number of validation cases (to make epochs faster)
NUM_VALID_CASES = 0
# Data loader/augmentation setting
NUM_THREADS = 4
ENSEMBLE_AXES = ["axial", "sagittal", "coronal"]
BG_ZERO = True
# Training settings
BATCHES_PER_EPOCH = 20000
PATCH_SIZE = (5, 194, 194)
BATCH_SIZE = 2
GRADIENT_MASK_ZERO_VOXELS = True
HETEROSCEDASTIC_ENTROPY_TERM = True

# Target labels (Brats)
# Enhancing tumor ( ET - label 4)
# Peritumoral Edema ( ED - label 2)
# Necrotic and non-enhancing tumor core (NCR/NET - label 1)

# Target label sets
# Tumor core : Necrotic core + enhancing tumor
# Whole tumor: Necrotic core + enhancing tumor + edema
TARGET_LABEL_SETS = [[4], [1, 4], [1, 2, 4]]
TARGET_LABEL_NAMES = ["enhancing", "tumor_core", "whole_tumor"]
USE_ATTENTION = True

# Arguments to run the script
parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--pad",
    default=194,
    type=int,
    help="Size to pad to before applying classfier.  Should be larger than the input and divisible by 2 :-)",
)

parser.add_argument("-l", default=False, action="store_true", dest="lite_mode")
parser.add_argument("-l_na", default=False, action="store_true", dest="lite_mode_na")
parser.add_argument("-f_na", default=False, action="store_true", dest="full_mode_na")
parser.add_argument("--indir", type=str)

my_args = parser.parse_args()

INPUT_DIRS = my_args.indir
OUTPUT_DIR = os.path.join(my_args.indir, "results")


PATCH_SIZE = (5, my_args.pad, my_args.pad)

# Whether to activate the lite mode
if my_args.lite_mode:
    print("Running pipeline in lite mode...")
elif my_args.lite_mode_na:
    print("Running pipeline in lite mode without data augmentation")
elif my_args.full_mode_na:
    print("Running pipeline in full mode without data augmentation...")
else:
    print("Running pipeline in full mode...")

# Create output directory if it does not exist
if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


def load_case(subject):
    """
    Loads and stacks MRI images for a given subject.

    This function loads FLAIR, T1, T2, T1GD, and ground truth (gt) MRI images
    from the specified subject directory, stacks them into a single numpy array,
    and converts the data type to int16.

    Args:
        subject (str): The file path to the subject's directory containing the MRI images.

    Returns:
        np.ndarray: A 5-channel numpy array containing the stacked MRI images, with dtype int16.
    """
    flair = nib.load(f"{subject}/FLAIR.nii.gz").get_data()
    t1 = nib.load(f"{subject}/T1.nii.gz").get_data()
    t2 = nib.load(f"{subject}/T2.nii.gz").get_data()
    t1ce = nib.load(f"{subject}/T1GD.nii.gz").get_data()
    gt = nib.load(f"{subject}/gt.nii.gz").get_data()

    return np.stack([flair, t1, t2, t1ce, gt]).astype(np.int16)


# Functions to load cases both in the training and validation sets.
def load_training_case(subject):
    return load_case(subject)


def load_validation_case(subject):
    return load_case(subject)


def load_subject(subject, axis="axial"):
    """
    Loads the subject data and computes metadata.
    Args:
        subject (Union[np.ndarray, str]): The subject data as a numpy array or a key to retrieve the data from "subject_volumes".
        axis (str, optional): The axis along which to load the data. Must be one of ['axial', 'sagittal', 'coronal']. Defaults to 'axial'.
    Returns:
        Tuple[np.ndarray, dict]: A tuple containing the loaded data and a metadata dictionary with means and standard deviations for each modality.
    Raises:
        AssertionError: If the provided axis is not one of ['axial', 'sagittal', 'coronal'].
    """

    assert axis in ["axial", "sagittal", "coronal"]

    if isinstance(subject, np.ndarray):
        data = subject
    # else:
    #     data = subject_volumes[subject]

    if axis == "coronal":
        data = np.swapaxes(data, 1, 2)

    if axis == "axial":
        data = np.swapaxes(data, 1, 3)

    metadata = {}

    metadata["means"] = []

    metadata["sds"] = []

    for modality in range(data.shape[0] - 1):

        metadata["means"].append(np.mean(data[modality][data[modality] > 0]))
        metadata["sds"].append(np.std(data[modality][data[modality] > 0]))

    metadata["means"] = np.array(metadata["means"])
    metadata["sds"] = np.array(metadata["sds"])

    # metadata = load_pickle(subject + ".pkl")

    return data, metadata


def load_subject_and_preprocess(subject, axis, bg_zero=True):
    """
    Loads and preprocesses the subject data.

    This function loads the subject data and its metadata, normalizes the image data by subtracting the mean and dividing by the standard deviation, and optionally sets the background to zero.

    Args:
        subject (str): The identifier for the subject to load.
        axis (int): The axis along which to load the subject data.
        bg_zero (bool, optional): If True, sets the background to zero. Defaults to True.

    Returns:
        np.ndarray: The preprocessed image data.
    """
    all_data, subject_metadata = load_subject(subject, axis=axis)
    image_data = all_data[:-1]
    zero_mask = (image_data != 0).astype(np.uint8)
    image_data = (
        image_data
        - np.array(subject_metadata["means"])[:, np.newaxis, np.newaxis, np.newaxis]
    )
    sds_expanded = np.array(subject_metadata["sds"])[
        :, np.newaxis, np.newaxis, np.newaxis
    ]
    image_data = np.divide(
        image_data, sds_expanded, out=np.zeros_like(image_data), where=sds_expanded != 0
    )

    if bg_zero:
        image_data = image_data * zero_mask
        image_data = image_data.astype(float)
        # image_data = (image_data * zero_mask).astype(float)

    return image_data


def get_gradient_mask(volumes):
    """
    Generates a gradient mask for the given volumes.

    If the global variable GRADIENT_MASK_ZERO_VOXELS is set to True, the function creates a mask where non-zero voxels
    across all volumes are marked as 1. Otherwise, it returns a mask of ones with the same shape as the first volume.

    Args:
        volumes (numpy.ndarray): A 4D numpy array representing multiple volumes.

    Returns:
        numpy.ndarray: A 3D numpy array representing the gradient mask.
    """
    if GRADIENT_MASK_ZERO_VOXELS:
        nonzero_mask_all = np.any(volumes > 0, axis=0)
        return (nonzero_mask_all).astype(float)
    else:
        return np.ones_like(volumes[0]).astype(float)


def make_target(gt, train=True, network_type=NETWORK_TYPE):
    """
    Generates a target array by concatenating binary masks for each label set.

    Args:
        gt (numpy.ndarray): Ground truth array.
        train (bool, optional): Flag indicating if the function is being used for training. Defaults to True.
        network_type (str, optional): Type of network, affects the shape of the target if '3D-to-2D'. Defaults to NETWORK_TYPE.

    Returns:
        numpy.ndarray: Target array with concatenated binary masks.
    """
    target = np.concatenate(
        [
            np.isin(gt, np.array(labelset)).astype(float)
            for labelset in TARGET_LABEL_SETS
        ],
        axis=1,
    )
    if train and network_type == "3D-to-2D":
        target = target[:, :, PATCH_SIZE[0] // 2]
    return target


criteria = [
    BCE_from_logits_focal(2, indices=[0, 1, 2]),
    DiceLoss(indices=[0, 1, 2]),
    Heteroscedastic_loss(
        2,
        0,
        target_indices=[0, 1, 2],
        flip_indices=[3, 4, 5],
        use_entropy=HETEROSCEDASTIC_ENTROPY_TERM,
    ),
]


def rotateImage(image, angle, interp=cv2.INTER_NEAREST):
    """
    Rotates an image by a specified angle around its center.

    This function uses OpenCV to rotate an image around its center point by the given angle.
    The rotation is performed using the specified interpolation method.

    Args:
        image (numpy.ndarray): The input image to be rotated. It should be a 2D or 3D NumPy array
            with shape `(height, width, ...)`.
        angle (float): The angle of rotation in degrees. Positive values rotate the image
            counterclockwise, and negative values rotate it clockwise.
        interp (int, optional): The interpolation method to be used during the rotation.
            Defaults to "cv2.INTER_NEAREST".
    """

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=interp)
    return result


def rotate_image_on_axis(image, angle, rot_axis):
    """
    Rotates a 3D image around a specified axis.

    Args:
        image (numpy.ndarray): The input 3D image represented as a NumPy array.
        angle (float): The angle by which to rotate the image, in degrees.
        rot_axis (int): The axis along which the image should be rotated (e.g., 0, 1, or 2).

    Returns:
        numpy.ndarray: The rotated 3D image.
    """

    return np.swapaxes(
        rotateImage(np.swapaxes(image, 2, rot_axis), angle, cv2.INTER_LINEAR),
        2,
        rot_axis,
    )


def rotate_stack(stack, angle, rot_axis):
    """
    Rotates a stack of 3D images around a specified axis.

    Args:
        stack (numpy.ndarray): A stack of 3D images represented as a 4D NumPy array
            with shape (N, H, W, D), where N is the number of images.
        angle (float): The angle by which to rotate each image, in degrees.
        rot_axis (int): The axis along which the images should be rotated (e.g., 0, 1, or 2).

    Returns:
        numpy.ndarray: A 4D NumPy array containing the rotated stack of 3D images,
        with the same shape as the input stack.
    """

    images = []
    for idx in range(stack.shape[0]):
        images.append(rotate_image_on_axis(stack[idx], angle, rot_axis))
    return np.stack(images, axis=0)


def list_transformations(models, axes, do_mirroring, rot_angles, rot_axes):
    """
    Generates a list of transformation parameter combinations for a set of models.

    Args:
        models (list): A list of models (e.g., strings or identifiers) for which transformations are generated.
        axes (list): A list of spatial axes (e.g., [0, 1, 2]) to be considered for transformations.
        do_mirroring (list): A list of boolean values indicating whether mirroring should be applied (e.g., [True, False]).
        rot_angles (list): A list of rotation angles (in degrees) to be applied to the models.
        rot_axes (list): A list of rotation axes (e.g., [0, 1, 2]) to be considered for transformations.

    Returns:
        dict: A dictionary where each key is a model from "models", and the value is a list of tuples,
        each representing a transformation. Each tuple contains:
        - axis (int): The spatial axis.
        - angle (float): The rotation angle.
        - rot_axis (int): The rotation axis.
        - mirror (bool): Whether mirroring is applied.
    """

    parameter_combinations = [
        (model, axis, mirror, angle, rot_axis)
        for model in models
        for axis in axes
        for mirror in do_mirroring
        for angle in rot_angles
        for rot_axis in rot_axes
    ]

    transformations = {
        model: [
            (axis, angle, rot_axis, mirror)
            for m, axis, mirror, angle, rot_axis in parameter_combinations
            if m == model
        ]
        for model in models
    }

    return transformations


def apply_to_case_uncert(
    models,
    subject,
    do_mirroring=[False],
    rot_angles=[-15, 0, 15],
    rot_axes=[0],
    patch_size=PATCH_SIZE,
    axes=["axial"],
    bg_zero=True,
):
    """
    Applies a set of models to a subject to generate predictions and uncertainty estimates
    using rotation, mirroring, and axis-specific augmentations depending on the running mode of the pipeline.

    Args:
        models (list): A list of trained models to be applied.
        subject (str): Identifier for the subject data to be loaded and processed.
        do_mirroring (list of bool, optional): Whether to apply mirroring. Defaults to [False].
        rot_angles (list of int, optional): Angles (in degrees) for data rotation. Defaults to [-15, 0, 15].
        rot_axes (list of int, optional): Axes indices for rotation. Defaults to [0].
        patch_size (tuple, optional): Dimensions of the input patch for the models. Defaults to PATCH_SIZE.
        axes (list of str, optional): Axes for applying model predictions. Defaults to ["axial"].
        bg_zero (bool, optional): Whether to zero-out the background in the input data. Defaults to True.

    Returns:
        np.ndarray: Combined predictions and uncertainty estimates across all axes, rotations,
        and models, including logits and flip probabilities.

    """

    print(
        f"applying {len(models)} model(s) over {len(axes)} axes rotating through {len(rot_angles)} angle(s)"
    )
    with torch.no_grad():
        ensemble_logits = []
        ensemble_flips = []
        slice_masks = []
        case_data = load_subject_and_preprocess(
            subject, axis="sagittal", bg_zero=bg_zero
        )

        if do_mirroring == False:
            do_mirroring = [False]
        if do_mirroring == True:
            do_mirroring = [True, False]

        for model in models:

            model.eval()

            if my_args.lite_mode_na or my_args.full_mode_na:

                image_data = case_data

                if NETWORK_TYPE == "3D-to-2D":

                    input, slicer = pad_nd_image(
                        image_data,
                        (0, patch_size[1], patch_size[2]),
                        return_slicer=True,
                    )

                if NETWORK_TYPE == "3D":

                    input, slicer = pad_nd_image(
                        image_data, patch_size, return_slicer=True
                    )

                # Performs the prediction with the model

                output = model.predict_3D(
                    torch.from_numpy(input).float().cuda(),
                    patch_size=patch_size,
                )

                # Adds a slicer to increase the padding to cover the model output.
                slicer[0] = slice(0, len(TARGET_LABEL_SETS) * 2, None)

                # Data is sliced to remove the padding and keep only the relevant part of the output
                output = output[tuple(slicer)]

                # Slice sums are computed to determine which slices contain relevant data: number of non-zero pixels in each slice accross width and height
                slice_sums = np.sum(np.any(image_data > 0, 0), (1, 2))

                # Checks whether a slice contains more than 2500 non-zero pixels.
                slice_mask = np.stack(
                    [np.stack([slice_sums > 2500] * image_data.shape[2], -1)]
                    * image_data.shape[3],
                    -1,
                )

                slice_mask = np.stack([slice_mask] * len(TARGET_LABEL_SETS)).astype(
                    np.uint8
                )

                output[: len(TARGET_LABEL_NAMES)][np.logical_not(slice_mask)] = np.nan

                ensemble_logits.append(output[: len(TARGET_LABEL_NAMES)])

                flip = (
                    (
                        -torch.from_numpy(output[len(TARGET_LABEL_NAMES) :]).exp()
                    ).sigmoid()
                ).numpy()

                flip[np.logical_not(slice_mask)] = np.nan
                ensemble_flips.append(flip)

                slice_masks.append(slice_mask)

            else:

                for axis in axes:

                    for mirror in do_mirroring:

                        for angle in rot_angles:

                            for rot_axis in rot_axes:

                                if angle != 0:
                                    image_data = rotate_stack(
                                        case_data, angle, rot_axis
                                    )

                                else:
                                    image_data = case_data

                                if mirror:
                                    image_data = image_data[:, ::-1]

                                if axis == "coronal":
                                    image_data = np.swapaxes(image_data, 1, 2)

                                if axis == "axial":
                                    image_data = np.swapaxes(image_data, 1, 3)

                                if NETWORK_TYPE == "3D-to-2D":

                                    input, slicer = pad_nd_image(
                                        image_data,
                                        (0, patch_size[1], patch_size[2]),
                                        return_slicer=True,
                                    )

                                if NETWORK_TYPE == "3D":

                                    input, slicer = pad_nd_image(
                                        image_data, patch_size, return_slicer=True
                                    )

                                # Performs the prediction with the model

                                output = model.predict_3D(
                                    torch.from_numpy(input).float().cuda(),
                                    patch_size=patch_size,
                                )

                                # Adds a slicer to increase the padding to cover the model output.
                                slicer[0] = slice(0, len(TARGET_LABEL_SETS) * 2, None)

                                # Data is sliced to remove the padding and keep only the relevant part of the output
                                output = output[tuple(slicer)]

                                # Slice sums are computed to determine which slices contain relevant data: number of non-zero pixels in each slice accross width and height
                                slice_sums = np.sum(np.any(image_data > 0, 0), (1, 2))

                                # Checks whether a slice contains more than 2500 non-zero pixels.
                                slice_mask = np.stack(
                                    [
                                        np.stack(
                                            [slice_sums > 2500] * image_data.shape[2],
                                            -1,
                                        )
                                    ]
                                    * image_data.shape[3],
                                    -1,
                                )

                                slice_mask = np.stack(
                                    [slice_mask] * len(TARGET_LABEL_SETS)
                                ).astype(np.uint8)

                                if axis == "coronal":
                                    output = np.swapaxes(output, 1, 2)
                                    image_data = np.swapaxes(image_data, 1, 2)
                                    slice_mask = np.swapaxes(slice_mask, 1, 2)

                                if axis == "axial":
                                    output = np.swapaxes(output, 1, 3)
                                    image_data = np.swapaxes(image_data, 1, 3)
                                    slice_mask = np.swapaxes(slice_mask, 1, 3)

                                if mirror:
                                    output = output[:, ::-1].copy()
                                    image_data = image_data[:, ::-1]
                                    slice_mask = slice_mask[:, ::-1]

                                if angle != 0:
                                    output = rotate_stack(output, -angle, rot_axis)
                                    slice_mask = (
                                        rotate_stack(slice_mask, -angle, rot_axis) > 0
                                    ).astype(np.uint8)

                                output[: len(TARGET_LABEL_NAMES)][
                                    np.logical_not(slice_mask)
                                ] = np.nan

                                ensemble_logits.append(
                                    output[: len(TARGET_LABEL_NAMES)]
                                )

                                flip = (
                                    (
                                        -torch.from_numpy(
                                            output[len(TARGET_LABEL_NAMES) :]
                                        ).exp()
                                    ).sigmoid()
                                ).numpy()

                                flip[np.logical_not(slice_mask)] = np.nan
                                ensemble_flips.append(flip)

                                slice_masks.append(slice_mask)

    ensemble_counts = np.sum(slice_masks, axis=0)
    print(f"Ensemble count shape is {ensemble_counts.shape}")

    full_logit = np.sum(
        np.divide(
            np.nan_to_num(np.array(ensemble_logits), 0),
            ensemble_counts,
            out=np.zeros_like(np.array(ensemble_logits)),
            where=ensemble_counts != 0,
        ),
        axis=0,
    )

    ensemble_predictions = np.greater(np.nan_to_num(ensemble_logits, -10), 0)

    print(f"Full logit shape is {full_logit.shape}")

    print(
        f"Axes: {axes}, do mirroring {do_mirroring}, rot angles {rot_angles}, rot_axes {rot_axes}, models {len(models)}"
    )
    full_predictions = np.stack(
        [np.greater(full_logit, 0)]
        * (
            len(axes)
            * len(do_mirroring)
            * len(rot_angles)
            * len(rot_axes)
            * len(models)
        ),
        0,
    )

    print(
        f"Shape of full predictions is {full_predictions.shape} and ensemble predictions is {ensemble_predictions.shape}"
    )

    preds_agree = np.equal(full_predictions, ensemble_predictions).astype(np.uint8)

    with np.errstate(divide="ignore", invalid="ignore"):
        full_flips = (
            np.sum(
                np.nan_to_num(np.array(ensemble_flips), 0) * (preds_agree)
                + np.nan_to_num((1 - np.array(ensemble_flips)), 0) * (1 - preds_agree),
                axis=0,
            )
            / ensemble_counts
        )

    full_flips = np.nan_to_num(full_flips, 0)

    return np.concatenate([full_logit, full_flips])


def apply_to_case_uncert_and_get_target_mask_probs_logits(
    model,
    subject,
    do_mirroring=False,
    rot_angles=[0],
    rot_axes=[0],
    patch_size=PATCH_SIZE,
    axes=["axial"],
):
    """
    Applies a model to a subject for prediction, calculates logits, probabilities,
    and applies a gradient mask to refine the target predictions.

    Args:
        model (nn.Module): The trained model used for prediction.
        subject (str): Identifier for the subject data to be processed.
        do_mirroring (bool, optional): Whether to apply mirroring during prediction. Defaults to False.
        rot_angles (list of int, optional): Angles (in degrees) for data rotation. Defaults to [0].
        rot_axes (list of int, optional): Axes indices for rotation. Defaults to [0].
        patch_size (tuple, optional): Dimensions of the input patch for the model. Defaults to PATCH_SIZE.
        axes (list of str, optional): Axes for applying model predictions (e.g., ["axial"]). Defaults to ["axial"].

    Returns:
        tuple:
            - target_probs (np.ndarray): Probability map for each target label after applying gradient mask.
            - target_logits (np.ndarray): Logits for each target label after applying gradient mask.
    """

    subject_data, _ = load_subject(subject, axis="sagittal")

    logits = apply_to_case_uncert(
        model,
        subject=subject,
        do_mirroring=do_mirroring,
        rot_angles=rot_angles,
        rot_axes=rot_axes,
        patch_size=patch_size,
        axes=axes,
    )

    target = make_target(subject_data[-1][None, None], train=False)[0]

    target_logits = logits[: len(TARGET_LABEL_NAMES)]

    target_probs = torch.from_numpy(target_logits).sigmoid().numpy()

    gradient_mask = get_gradient_mask(subject_data[:-1]).astype(float)

    target_logits = target_logits * gradient_mask[None]

    target_probs = target_probs * gradient_mask[None]

    return logits, gradient_mask, target, target_logits, target_probs


def get_TP_FP_FN(target, prediction, from_logits=True):
    """
    Computes the True Positives (TP), False Positives (FP), False Negatives (FN),
    and True Negatives (TN) for binary classification predictions.

    Args:
        target (np.ndarray): Ground truth labels. Should have the same shape as 'prediction'.
        prediction (np.ndarray): Predicted values, either logits or probabilities.
        from_logits (bool, optional):
            If True, treats "prediction" as logits and applies a threshold of 0.
            If False, treats "prediction" as probabilities and applies a threshold of 0.5. Defaults to True.

    Returns:
        tuple:
            - TP (np.ndarray): True positive counts for each sample.
            - FP (np.ndarray): False positive counts for each sample.
            - FN (np.ndarray): False negative counts for each sample.
    """

    if from_logits:
        threshold = 0
    else:
        threshold = 0.5

    TP = np.sum(np.logical_and(prediction > threshold, target > 0), axis=(1, 2, 3))
    FP = np.sum(np.logical_and(prediction > threshold, target == 0), axis=(1, 2, 3))
    FN = np.sum(np.logical_and(prediction <= threshold, target > 0), axis=(1, 2, 3))

    return TP, FP, FN


def get_dices_from_TP_FP_FN(TP, FP, FN):
    """
    Computes the Dice coefficient for binary classification based on True Positives (TP),
    False Positives (FP), and False Negatives (FN).

    Args:
        TP (np.ndarray): Array of True Positive counts for each sample.
        FP (np.ndarray): Array of False Positive counts for each sample.
        FN (np.ndarray): Array of False Negative counts for each sample.

    Returns:
        np.ndarray: Array of Dice coefficients for each sample.
    """
    epsilon = 0.000001
    dices = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
    return dices


def get_dices_from_prediction(target, prediction, from_logits=True):
    """
    Computes the Dice coefficients for binary classification predictions.

    Args:
        target (np.ndarray): Ground truth labels. Should have the same shape as "prediction".
        prediction (np.ndarray): Predicted values, either logits or probabilities.
        from_logits (bool, optional):
            If True, treats "prediction" as logits and applies a threshold of 0.
            If False, treats "prediction" as probabilities and applies a threshold of 0.5. Defaults to True.

    Returns:
        np.ndarray: Array of Dice coefficients for each sample.
    """

    TP, FP, FN = get_TP_FP_FN(target, prediction, from_logits=from_logits)
    return get_dices_from_TP_FP_FN(TP, FP, FN)


def reduce_3d_depth(in_channel, out_channel, kernel_size, padding):
    layer = nn.Sequential(
        OrderedDict(
            [
                ("pad1", nn.ReplicationPad3d((1, 1, 1, 1, 0, 0))),
                (
                    "conv1",
                    nn.Conv3d(
                        in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ),
                ("bn1", nn.InstanceNorm3d(out_channel, affine=False)),
                ("relu1", nn.ReLU()),
                # ("dropout", nn.Dropout(p=0.2))
            ]
        )
    )
    return layer


def down_layer(in_channel, out_channel, kernel_size, padding):
    """
    Creates a downsampling layer consisting of two convolutional blocks with
    normalization, activation, and optional dropout.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        padding (int or tuple): Padding applied to the input for convolutional layers.

    Returns:
        nn.Sequential: A sequential container of the downsampling layer.
    """

    layer = nn.Sequential(
        OrderedDict(
            [
                ("pad1", nn.ReplicationPad2d(1)),
                (
                    "conv1",
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ),
                ("bn1", nn.InstanceNorm2d(out_channel, affine=False)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(p=0.0)),
                ("pad2", nn.ReplicationPad2d(1)),
                (
                    "conv2",
                    nn.Conv2d(
                        out_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ),
                ("bn2", nn.InstanceNorm2d(out_channel, affine=False)),
                ("relu2", nn.ReLU()),
                ("dropout2", nn.Dropout(p=0.0)),
            ]
        )
    )
    return layer


def up_layer(in_channel, out_channel, kernel_size, padding):
    """
    Creates an upsampling layer consisting of two convolutional blocks with
    normalization, activation, and optional dropout.

    Args:
        in_channel (int): Number of input channels.
        out_channel (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolutional kernel.
        padding (int or tuple): Padding applied to the input for convolutional layers.

    Returns:
        nn.Sequential: A sequential container of the upsampling layer.
    """

    layer = nn.Sequential(
        OrderedDict(
            [
                ("pad1", nn.ReplicationPad2d(1)),
                (
                    "conv1",
                    nn.Conv2d(
                        in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ),
                ("bn1", nn.InstanceNorm2d(out_channel, affine=False)),
                ("relu1", nn.ReLU()),
                ("dropout1", nn.Dropout(p=0.0)),
                ("pad2", nn.ReplicationPad2d(1)),
                (
                    "conv2",
                    nn.Conv2d(
                        out_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        padding=padding,
                    ),
                ),
                ("bn2", nn.InstanceNorm2d(out_channel, affine=False)),
                ("relu2", nn.ReLU()),
                ("dropout2", nn.Dropout(p=0.0)),
            ]
        )
    )
    return layer


class DilatedDenseUnit(nn.Module):
    """
    Implements a Dilated Dense Unit, a building block for dense neural networks
    that incorporates dilation to expand the receptive field.

    Args:
        in_channel (int): Number of input channels.
        growth_rate (int): Number of output channels produced by the unit.
        kernel_size (int or tuple): Size of the convolutional kernel.
        dilation (int): Dilation factor for the convolutional layer.

    Attributes:
        layer (nn.Sequential): A sequential container that performs the following operations:
            1. Instance normalization.
            2. ReLU activation.
            3. Replication padding (based on the dilation factor).
            4. Dilated convolution.
            5. Dropout (with a probability of 0.0 by default).

    Forward Pass:
        - Input is passed through the "layer".
        - The output is concatenated with the original input to enable dense connectivity.

    Methods:
        forward(x):
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).

            Returns:
                torch.Tensor: Concatenated tensor of shape
                (batch_size, in_channel + growth_rate, height, width).
    """

    def __init__(self, in_channel, growth_rate, kernel_size, dilation):
        super(DilatedDenseUnit, self).__init__()
        self.layer = nn.Sequential(
            OrderedDict(
                [
                    ("bn1", nn.InstanceNorm2d(in_channel, affine=False)),
                    ("relu1", nn.ReLU()),
                    ("pad1", nn.ReplicationPad2d(dilation)),
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channel,
                            growth_rate,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            padding=0,
                        ),
                    ),
                    ("dropout", nn.Dropout(p=0.0)),
                ]
            )
        )

    def forward(self, x):
        out = x
        out = self.layer(out)
        out = concatenate(x, out)
        return out


class AttentionModule(nn.Module):
    """
    Implements an Attention Module that applies spatial attention to enhance input feature maps.

    Args:
        in_channel (int): Number of input channels.
        intermediate_channel (int): Number of channels in the intermediate feature map.
        out_channel (int): Number of output channels for the attention mask.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.

    Attributes:
        layer (nn.Sequential): A sequential container consisting of:
            - Instance normalization and ReLU activation.
            - Replication padding to preserve spatial dimensions.
            - Convolutional layers to transform the input and compute the attention mask.
            - Sigmoid activation to produce an attention mask in the range [0, 1].

    Methods:
        forward(x):
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channel, height, width).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, in_channel, height, width)
                with enhanced features based on the attention mask.
    """

    def __init__(self, in_channel, intermediate_channel, out_channel, kernel_size=3):
        super(AttentionModule, self).__init__()
        self.layer = nn.Sequential(
            OrderedDict(
                [
                    ("bn1", nn.InstanceNorm2d(in_channel, affine=False)),
                    ("relu1", nn.ReLU()),
                    ("pad1", nn.ReplicationPad2d(1)),
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channel,
                            intermediate_channel,
                            kernel_size=kernel_size,
                            padding=0,
                        ),
                    ),
                    ("bn2", nn.InstanceNorm2d(intermediate_channel, affine=False)),
                    ("relu2", nn.ReLU()),
                    ("pad2", nn.ReplicationPad2d(1)),
                    (
                        "conv2",
                        nn.Conv2d(
                            intermediate_channel,
                            out_channel,
                            kernel_size=kernel_size,
                            padding=0,
                        ),
                    ),
                    ("sigmoid", nn.Sigmoid()),
                ]
            )
        )

    def forward(self, x):
        out = x
        out = self.layer(out)
        out = x * out
        return out


def center_crop(layer, target_size):
    """
    Performs a center crop on the given input tensor.

    Args:
        layer (torch.Tensor): Input tensor of shape
            (batch_size, channels, width, height).
        target_size (int): The size of the cropped region.
            Assumes a square crop (i.e., height = width = target_size).

    Returns:
        torch.Tensor: Cropped tensor of shape
        (batch_size, channels, target_size, target_size).
    """

    _, _, layer_width, _ = layer.size()
    start = (layer_width - target_size) // 2
    crop = layer[:, :, start : (start + target_size), start : (start + target_size)]
    return crop


def concatenate(link, layer):
    """
    Concatenates two tensors along the channel dimension.

    Args:
        link (torch.Tensor): First input tensor of shape
            (batch_size, channels_1, width, height).
        layer (torch.Tensor): Second input tensor of shape
            (batch_size, channels_2, width, height).

    Returns:
        torch.Tensor: Concatenated tensor of shape
        (batch_size, channels_1 + channels_2, width, height).
    """

    concat = torch.cat([link, layer], 1)
    return concat


def dense_bottleneck(in_channel, growth_rate=12, depth=[4, 4, 4, 4]):
    """
    Creates a dense bottleneck block consisting of multiple dilated dense units
    and optional attention modules.

    Args:
        in_channel (int): Number of input channels to the bottleneck block.
        growth_rate (int, optional): Number of channels added by each dilated dense unit.
            Default is 12.
        depth (list[int], optional): A list where each element specifies the number of
            dilated dense units at a particular dilation rate. The length of the list
            determines the number of distinct dilation rates. Default is [4, 4, 4, 4].

    Returns:
        nn.Sequential: A sequential container of dilated dense units and attention modules.
        int: The number of output channels after the bottleneck block.
    """

    layer_dict = OrderedDict()
    for idx, growth_steps in enumerate(depth):
        dilation_rate = 2**idx
        for y in range(growth_steps):
            layer_dict["dilated_{}_{}".format(dilation_rate, y)] = DilatedDenseUnit(
                in_channel, growth_rate, kernel_size=3, dilation=dilation_rate
            )
            in_channel = in_channel + growth_rate

        if USE_ATTENTION:
            layer_dict["attention_{}".format(dilation_rate)] = AttentionModule(
                in_channel, in_channel // 4, in_channel
            )
    return nn.Sequential(layer_dict), in_channel


class UNET_3D_to_2D(nn.Module):
    """
    A 3D-to-2D U-Net architecture designed for 3D volumetric data processed in 2D slices.
    Incorporates dense bottleneck layers and optional attention mechanisms.

    Attributes:
        depth (int): Number of downsampling and upsampling steps in the U-Net.
        channels_in (int): Number of input channels for the 3D data. Default is 1.
        channels_2d_to_3d (int): Number of channels after reducing 3D depth. Default is 32.
        channels (int): Number of feature channels in the first U-Net layer. Default is 32.
        output_channels (int): Number of output channels for the final prediction. Default is 1.
        slices (int): Number of slices in the 3D volume. Default is 5.
        dilated_layers (list[int]): Number of dilated dense units for each dilation rate. Default is [4, 4, 4, 4].
        growth_rate (int): Number of feature maps added by each dilated dense unit. Default is 12.

    Methods:
        forward(x):
            Performs the forward pass of the network.
        predict_3D(x, patch_size=(5, 194, 194), batch_size=BATCH_SIZE):
            Predicts the output logits for the full 3D volume using overlapping 2D patches.

    """

    def __init__(
        self,
        depth,
        channels_in=1,
        channels_2d_to_3d=32,
        channels=32,
        output_channels=1,
        slices=5,
        dilated_layers=[4, 4, 4, 4],
        growth_rate=12,
    ):
        """
        Initializes the UNET_3D_to_2D model.

        Args:
            depth (int): Number of downsampling and upsampling layers.
            channels_in (int, optional): Number of input channels for the 3D data. Default is 1.
            channels_2d_to_3d (int, optional): Number of channels after reducing 3D depth. Default is 32.
            channels (int, optional): Number of feature channels in the first U-Net layer. Default is 32.
            output_channels (int, optional): Number of output channels for the final prediction. Default is 1.
            slices (int, optional): Number of slices in the 3D input volume. Default is 5.
            dilated_layers (list[int], optional): Number of dense units per dilation rate. Default is [4, 4, 4, 4].
            growth_rate (int, optional): Number of channels added by each dense unit. Default is 12.
        """
        super(UNET_3D_to_2D, self).__init__()

        self.output_channels = output_channels
        self.main_modules = []

        self.depth = depth
        self.slices = slices

        self.depth_reducing_layers = ModuleList(
            [
                reduce_3d_depth(in_channel, channels_2d_to_3d, kernel_size=3, padding=0)
                for in_channel in [channels_in]
                + [channels_2d_to_3d] * (slices // 2 - 1)
            ]
        )

        self.down1 = down_layer(
            in_channel=channels_2d_to_3d, out_channel=channels, kernel_size=3, padding=0
        )
        self.main_modules.append(self.down1)
        self.max1 = nn.MaxPool2d(2)
        self.down_layers = ModuleList(
            [
                down_layer(
                    in_channel=channels * (2**i),
                    out_channel=channels * (2 ** (i + 1)),
                    kernel_size=3,
                    padding=0,
                )
                for i in range(self.depth)
            ]
        )
        self.main_modules.append(self.down_layers)
        self.max_layers = ModuleList([nn.MaxPool2d(2) for i in range(self.depth)])

        self.bottleneck, bottleneck_features = dense_bottleneck(
            channels * 2**self.depth, growth_rate=growth_rate, depth=dilated_layers
        )
        self.main_modules.append(self.bottleneck)

        self.upsampling_layers = ModuleList(
            [
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "upsampling",
                                nn.Upsample(
                                    scale_factor=2, mode="bilinear", align_corners=True
                                ),
                            ),  # since v0.4.0 align_corners= False is default, before was True
                            ("pad", nn.ReplicationPad2d(1)), 
                            (
                                "conv",
                                nn.Conv2d(
                                    in_channels=bottleneck_features,
                                    out_channels=bottleneck_features,
                                    kernel_size=3,
                                    padding=0,
                                ),
                            ),
                        ]
                    )
                )
                for i in range(self.depth, -1, -1)
            ]
        )
        self.main_modules.append(self.upsampling_layers)
        self.up_layers = ModuleList(
            [
                up_layer(
                    in_channel=bottleneck_features + channels * (2 ** (i)),
                    out_channel=bottleneck_features,
                    kernel_size=3,
                    padding=0,
                )
                for i in range(self.depth, -1, -1)
            ]
        )

        self.main_modules.append(self.up_layers)
        self.last = nn.Conv2d(
            in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1
        )
        self.main_modules.append(self.last)

        self.logvar = nn.Conv2d(
            in_channels=bottleneck_features, out_channels=output_channels, kernel_size=1
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, slices, depth, height, width).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, 2 * output_channels, height, width).
                        Contains both prediction and log-variance.
        """

        # Down

        out = x

        for i in range(self.slices // 2):
            out = self.depth_reducing_layers[i](out)

        out.transpose_(1, 2).contiguous()
        size = out.size()
        out = out.view((-1, size[2], size[3], size[4]))

        links = []
        out = self.down1(out)
        links.append(out)
        out = self.max1(out)

        for i in range(self.depth):
            out = self.down_layers[i](out)
            links.append(out)
            out = self.max_layers[i](out)

        out = self.bottleneck(out)

        links.reverse()

        # Up

        for i in range(self.depth + 1):

            out = self.upsampling_layers[i](out)

            out = concatenate(links[i], out)
            out = self.up_layers[i](out)

        pred = self.last(out)
        logvar = self.logvar(out)

        return torch.cat([pred, logvar], axis=1)

    def predict_3D(
        self,
        x,
        patch_size=(5, 194, 194),
        batch_size=BATCH_SIZE,
    ):
        """
        Predicts the output for a 3D volume by processing overlapping 2D patches.

        Args:
            x (torch.Tensor): The input 3D tensor of shape (batch_size, depth, height, width).
            patch_size (tuple[int], optional): The size of patches to process. Default is (5, 194, 194).
            batch_size (int, optional): The number of patches processed in a batch. Default is BATCH_SIZE.

        Returns:
            numpy.ndarray: The predicted logits for the entire 3D volume.
        """

        self.eval()
        with torch.no_grad():

            logit_total = []

            stack_depth = patch_size[0]

            padding = stack_depth // 2

            input = torch.nn.ConstantPad3d((0, 0, 0, 0, padding, padding), 0)(x)

            slice = 0

            for idx in range(x.shape[1] // batch_size + 1):

                batch_list = []

                for y in range(batch_size):
                    if slice == x.shape[1]:
                        break
                    batch_list.append(input[:, slice : slice + stack_depth])
                    slice += 1
                if len(batch_list) == 0:
                    break

                batch_tensor = torch.stack(batch_list, 0)
                logit = self.forward(batch_tensor).transpose(0, 1)
                logit_total.append(logit)

            full_logit = torch.cat(logit_total, 1)

        return full_logit.detach().cpu().numpy()


def load_checkpoint(net, checkpoint_file):
    """Loads the checkpoint from the given file and updates the model state.

    Args:
        net (torch.nn.Module): The neural network model to which the checkpoint's state dict will be loaded.
        checkpoint_file (str): The path to the checkpoint file to be loaded.
    """

    if os.path.isfile(checkpoint_file):
        print("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        net.load_state_dict(checkpoint["state_dict"])


def get_bbox_from_mask(mask, outside_value=0):
    """Calculates the bounding box from a binary mask.

    Args:
        mask (np.ndarray): A 3D binary mask (numpy array) where non-zero values represent the region of interest.
        outside_value (int, optional): The value that represents the background or outside region in the mask. Default is 0.

    Returns:
        list: A list containing three elements, each representing the bounds along one axis (z, x, y).
            Each element is a list of two values: the minimum and maximum indices along that axis.
    """
    mask_voxel_coords = np.where(mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    return [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]


def crop_to_bbox(image, bbox):
    """Crops a 3D image to the specified bounding box.

    Args:
        image (np.ndarray): A 3D image (numpy array) to be cropped.
        bbox (list): A list of three elements, each containing two values representing the
                    minimum and maximum indices along the z, x, and y axes for cropping.

    Returns:
        np.ndarray: The cropped 3D image, which is a sub-array of the input image defined by the bounding box.
    """

    assert len(image.shape) == 3, "only supports 3d images"
    resizer = (
        slice(bbox[0][0], bbox[0][1]),
        slice(bbox[1][0], bbox[1][1]),
        slice(bbox[2][0], bbox[2][1]),
    )
    return image[resizer]


def crop_to_nonzero(data, seg=None, nonzero_label=0):
    """
    Crops the input data to the region containing non-zero values, based on a binary mask.

    This function identifies the bounding box of the non-zero regions in the input data (based on the given condition),
    then crops each channel or slice of the data to this bounding box.

    Args:
        data (np.ndarray): A 4D numpy array where the first dimension represents the channels or slices
                            and the remaining three dimensions represent the spatial data.
        seg (np.ndarray, optional): A segmentation mask (not used in the current implementation, included for future extension).
        nonzero_label (int, optional): The label used to identify non-zero values in the mask. Default is 0 (non-zero values are values greater than 0).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The cropped data, which is a sub-array of the input data, defined by the bounding box
                          around non-zero regions.
            - list: The bounding box used for cropping, in the format [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]].

    """
    bbox = get_bbox_from_mask(np.all(data[:-1] > 0, axis=0), 0)

    cropped_data = []
    for c in range(data.shape[0]):
        cropped = crop_to_bbox(data[c], bbox)
        cropped_data.append(cropped[None])
    data = np.vstack(cropped_data)

    return data, bbox


# Perform inference using the models from different folds
unets = []
for fold in [0, 1, 2, 3, 4]:
    model = UNET_3D_to_2D(
        0,
        channels_in=4,
        channels=64,
        growth_rate=8,
        dilated_layers=[4, 4, 4, 4],
        output_channels=len(TARGET_LABEL_NAMES),
    )

    model.load_state_dict(
        torch.load(f"fold{fold}_retrained_nn.pth.tar", map_location="cuda:0")
    )

    model.to(device)

    unets.append(model)


def mask_flip_probabilities(output, gradient_mask):
    """Calculates the flipped probability map for a given output and gradient mask.

    This function extracts the probability values from the output array (starting from a certain index based on
    the number of target label names) and then multiplies the result by a provided gradient mask. The result is
    a mask of probabilities that can be used for further processing, such as flipping or adjusting regions based
    on the gradient.

    Args:
        output (np.ndarray): The output array, typically representing the model's predictions or probabilities.
                              It is assumed to be of shape (N, C, H, W) where N is the batch size, C is the number
                              of channels, and H, W are the spatial dimensions. The relevant probability values
                              start from index len(TARGET_LABEL_NAMES).
        gradient_mask (np.ndarray): A binary mask (numpy array) of the same spatial dimensions as "output",
                                    used to mask the relevant probabilities.

    Returns:
        np.ndarray: A probability map of the same spatial dimensions as the input, containing the flipped
                    probabilities based on the gradient mask.
    """

    flip_probs = output[len(TARGET_LABEL_NAMES) :] * gradient_mask[None]

    return flip_probs


def resolve_file(casepath, modality):
    """
    Resolves and returns the file corresponding to a specific modality in a given case directory.

    This function searches for files in the specified directory that match the modality (in `.nii` format),
    ensuring that exactly one matching file is found. It returns the filename of the identified file.

    Args:
        casepath (str): The path to the case directory where the modality files are stored.
        modality (str): The modality to search for, which should correspond to part of the filename (excluding the `.nii` extension).

    Returns:
        str: The filename of the modality file that matches the provided modality.
    """
    filenames = [x for x in os.listdir(casepath) if modality + ".nii" in x]
    print(filenames)
    assert len(filenames) == 1
    return filenames[0]


def predict_with_uncertainty(
    id=INPUT_DIRS,
    models=unets,
    val_segs=OUTPUT_DIR,
    post_segs=OUTPUT_DIR,
    export_folder=OUTPUT_DIR,
    PATCH_SIZE=PATCH_SIZE,
):
    """Runs model predictions with uncertainty estimation on the provided medical imaging data.

    This function performs preprocessing, inference, and postprocessing steps for segmentation tasks,
    including handling uncertainty using model predictions, gradient masks, and postprocessing steps like
    flipping probabilities and fusion of target probabilities. It saves the results, including the segmentation
    and uncertainty maps, into specified output folders.

    Args:
        id (str): The directory path containing the subject data. Default is `INPUT_DIRS`.
        models (list): List of trained models (UNet models) used for prediction. Default is `unets`.
        val_segs (str, optional): Directory path to save the validation segmentation results. Default is `OUTPUT_DIR`.
        post_segs (str, optional): Directory path to save postprocessed segmentation results. Default is `OUTPUT_DIR`.
        export_folder (str, optional): Directory path to save uncertainty maps. Default is `OUTPUT_DIR`.
        PATCH_SIZE (tuple, optional): The patch size used for prediction. Default is `PATCH_SIZE`.

    Returns:
        np.ndarray: The postprocessed segmentation result.

    """

    print(id)
    if not os.path.isdir(val_segs):
        os.mkdir(val_segs)

    if not os.path.isdir(post_segs):
        os.mkdir(post_segs)
    if not os.path.isdir(export_folder):
        os.mkdir(export_folder)

    flair = np.copy(nib.load(id + resolve_file(id, "FLAIR")).get_fdata())
    t1 = np.copy(nib.load(id + resolve_file(id, "T1")).get_fdata())
    t2 = np.copy(nib.load(id + resolve_file(id, "T2")).get_fdata())
    t1ce = np.copy(nib.load(id + resolve_file(id, "T1GD")).get_fdata())

    gt = np.zeros_like(flair)

    val_subject_volume = np.stack([flair, t1, t2, t1ce, gt]).astype(np.int16)

    cropped, bbox = crop_to_nonzero(val_subject_volume)

    im_size = np.max(cropped.shape)

    print(f"cropped input image max dimension = {im_size}")

    if im_size > PATCH_SIZE[1]:
        PATCH_SIZE = (5, 2 * ((im_size + 1) // 2), 2 * ((im_size + 1) // 2))
        print(f"cropped image exceeds patch size: new patch size = {PATCH_SIZE}")

    if my_args.lite_mode or my_args.lite_mode_na or my_args.full_mode_na:

        logits, gradient_mask, _, _, target_probs = (
            apply_to_case_uncert_and_get_target_mask_probs_logits(models, cropped)
        )
    else:
        angles = [-45, 0, 45]
        logits, gradient_mask, _, _, target_probs = (
            apply_to_case_uncert_and_get_target_mask_probs_logits(
                models,
                cropped,
                axes=ENSEMBLE_AXES,
                rot_angles=angles,
                rot_axes=[0, 1, 2],
                patch_size=PATCH_SIZE,
                do_mirroring=True,
            )
        )

    target_probs_uncropped = np.zeros_like(
        np.stack([val_subject_volume[0]] * len(TARGET_LABEL_SETS))
    ).astype(np.float32)

    target_probs_uncropped[
        :, bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
    ] = target_probs

    nifti_orig = nib.load(id + resolve_file(id, "FLAIR"))
    nifti_affine = nifti_orig.affine

    flip = mask_flip_probabilities(logits, gradient_mask)

    flip_uncropped = np.zeros_like(
        np.stack([val_subject_volume[0]] * len(TARGET_LABEL_SETS))
    ).astype(np.float32)

    flip_uncropped[
        :, bbox[0][0] : bbox[0][1], bbox[1][0] : bbox[1][1], bbox[2][0] : bbox[2][1]
    ] = flip

    flip_prob_fusion = (target_probs_uncropped > 0.5) * (1 - flip_uncropped) + (
        target_probs_uncropped < 0.5
    ) * (flip_uncropped)

    seg = make_brats_segmentation(flip_prob_fusion * 100)

    seg_postprocessed, uncertainty = postprocess_brats_segmentation(
        seg, (flip_prob_fusion * 100).transpose((1, 2, 3, 0)), flair, t1
    )

    postprocessed_nifti = nib.Nifti1Image((seg).astype(np.uint8), nifti_affine)

    nib.save(postprocessed_nifti, f"{post_segs}/tumor_SCAN2020_class.nii.gz")

    for idx, name in zip([0, 1, 2], ["enhance", "core", "whole"]):
        unc_map = nib.Nifti1Image(
            (uncertainty[:, :, :, idx]).astype(np.uint8), nifti_affine
        )
        nib.save(unc_map, f"{export_folder}/tumor_SCAN2020_unc_{name}.nii.gz")

    return seg_postprocessed


def resolve_seg(casepath):
    """
    Resolves and returns the segmentation file in the given case directory.

    This function searches the provided directory for a file that contains the substring "_seg.nii.gz" in its name.
    If exactly one such file is found, it returns the filename; otherwise, it returns `None`.

    Args:
        casepath (str): The path to the directory containing the case files, including the segmentation file.

    Returns:
        str or None: The filename of the segmentation file (with "_seg.nii.gz" in the name) if found,
                     or `None` if no or more than one matching file is found.
    """
    filenames = [x for x in os.listdir(casepath) if "_seg.nii.gz" in x]
    print(filenames)
    if len(filenames) == 1:
        return filenames[0]
    else:
        return None


existing_seg = resolve_seg(INPUT_DIRS)

# Perform predictions
if existing_seg is None:
    if my_args.lite_mode or my_args.lite_mode_na:
        seg_postprocessed = predict_with_uncertainty(models=[unets[0]])
    else:
        seg_postprocessed = predict_with_uncertainty()
else:

    seg_postprocessed = nib.load(INPUT_DIRS + existing_seg).get_fdata()
