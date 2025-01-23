import torch
import torch.nn as nn


class BCE_from_logits_focal(nn.modules.Module):
    """
    Binary Cross-Entropy (BCE) loss with focal loss modification.

    This class implements a variant of the binary cross-entropy loss, modified with focal loss to
    handle imbalanced classification problems by down-weighting well-classified examples.
    It works with logits as inputs and includes optional support for masking specific elements in the loss calculation.

    Attributes:
        gamma (float): The focusing parameter for the focal loss. A higher value emphasizes
                       harder-to-classify examples more strongly.
        name (str): A name for the loss function ("focal loss").
        indices (list or None): A list of indices to select specific input channels for loss
                                computation. If `None`, all channels are used.

    Methods:
        forward(input, target, mask=None):
            Computes the focal loss given the input logits, target labels, and an optional mask.
            Returns the mean focal loss if no mask is provided or a masked average otherwise.

    Args:
        gamma (float): Focusing parameter for the focal loss term. Higher values place greater
                       focus on misclassified examples.
        indices (list, optional): List of indices for selecting specific channels from the input.
                                  If `None`, all channels are used.
    """

    def __init__(self, gamma, indices=None):
        super(BCE_from_logits_focal, self).__init__()
        self.gamma = gamma
        self.name = "focal loss"
        self.indices = indices

    def forward(self, input, target, mask=None):
        if self.indices is not None:
            input_target = input[:, self.indices]
        else:
            input_target = input
        assert input_target.shape == target.shape
        if mask is not None:
            if self.indices is not None:
                mask_target = mask[:, self.indices]
            else:
                mask_target = mask
            assert input_target.shape == mask_target.shape
        max_val = (-input_target).clamp(min=0)
        loss = (
            input_target
            - input_target * target
            + max_val
            + ((-max_val).exp() + (-input_target - max_val).exp()).log()
        )
        p = input_target.sigmoid()
        pt = (1 - p) * (1 - target) + p * target
        focal_loss = ((1 - pt).pow(self.gamma)) * loss
        if mask is None:
            return focal_loss.mean()
        else:
            eps = 1e-10
            return (focal_loss * mask_target).sum() / (torch.sum(mask_target) + eps)


class Heteroscedastic_loss(nn.modules.Module):
    """Heteroscedastic loss for handling uncertainty in classification tasks.

    This loss function combines focal loss with heteroscedastic modeling, incorporating predictions of the primary target
    and a secondary "flip" probability. The flip probability represents uncertainty or disagreement between predictions
    and ground truth, which can be used to attenuate the target labels.

    Attributes:
        target_gamma (float): Focusing parameter for the focal loss applied to the target prediction.
        flip_gamma (float): Focusing parameter for the focal loss applied to the flip probability.
                            Typically set to 0 for better convergence.
        name (str): The name of the loss function ("heteroscedastic loss").
        target_indices (list): Indices of the input tensor corresponding to target predictions.
        flip_indices (list): Indices of the input tensor corresponding to flip predictions.
        use_entropy (bool): Whether to include entropy-based regularization in the loss computation.

    Methods:
        forward(input, target, mask=None):
            Computes the heteroscedastic loss given input logits, target labels, and an optional mask.
    """

    def __init__(
        self, target_gamma, flip_gamma, target_indices, flip_indices, use_entropy=False
    ):
        super(Heteroscedastic_loss, self).__init__()
        # both the prediction of the target and the prediction of the disgreement
        # can be focal.  Recommendation is to set flip_gamma to be zero, since the
        # error rate may (hopefully) decrease over time.
        self.target_gamma = target_gamma
        self.flip_gamma = flip_gamma
        self.name = "heteroscedastic loss"
        self.target_indices = target_indices
        self.flip_indices = flip_indices
        self.use_entropy = use_entropy

    def forward(self, input, target, mask=None):
        input_target = input[:, self.target_indices]
        input_flip = input[:, self.flip_indices]
        assert input_target.shape == target.shape
        assert input_flip.shape == target.shape
        if mask is not None:
            mask_target = mask[:, self.target_indices]
            assert input_target.shape == mask_target.shape
        else:
            mask_target = None

        # flip output (range[-inf , inf]) is log (-logit(p)). where p is the flip probability
        # this ensures that the flip prob is below 0.5
        flip_prob = torch.sigmoid(-torch.exp(input_flip))

        # attenuate the target, by reducing its label in places where the classifier predicts disagreement

        flip_attenuated_target = (1 - target) * flip_prob + target * (1 - flip_prob)

        # generate a 'ground truth' for where the classifier and label set disagree

        false_neg = ((-input_target).sign().clamp(min=0)) * target
        false_pos = input_target.sign().clamp(min=0) * (1 - target)
        label_disagreement = false_neg + false_pos

        flip_loss = BCE_from_logits_focal(self.flip_gamma)(
            -torch.exp(input_flip), label_disagreement, mask_target
        )

        if self.use_entropy:
            flip_logit = -torch.exp(input_flip)
            max_val = (-input_target).clamp(min=0)
            max_val_flip = (-flip_logit).clamp(min=0)
            loss = (
                input_target
                - input_target * flip_attenuated_target
                + max_val
                + ((-max_val).exp() + (-input_target - max_val).exp()).log()
            )
            entropy = (
                flip_logit
                - flip_logit * flip_prob
                + max_val_flip
                + ((-max_val_flip).exp() + (-flip_logit - max_val_flip).exp()).log()
            )
            p = input_target.sigmoid()
            pt = torch.abs(
                1 - (p - flip_attenuated_target)
            )  # (1-p)*(1-flip_attenuated_target) + p*flip_attenuated_target
            focal_loss = ((1 - pt).pow(self.target_gamma)) * (loss - entropy)
            if mask is None:
                target_loss = focal_loss.mean()
            else:
                eps = 1e-10
                target_loss = (focal_loss * mask_target).sum() / (
                    torch.sum(mask_target) + eps
                )
        else:
            target_loss = BCE_from_logits_focal(self.target_gamma)(
                input_target, flip_attenuated_target, mask_target
            )

        return target_loss + flip_loss


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss for segmentation tasks.

    This loss function computes the Dice coefficient in a soft, differentiable manner,
    making it suitable for training segmentation models. It measures the overlap
    between the predicted and ground truth labels and can be used for binary
    or multi-class segmentation.

    Attributes:
        batch_dice (bool): If True, computes Dice loss over the entire batch;
            otherwise, computes Dice loss for each sample individually. Defaults to False.
        smooth (float): Smoothing factor to prevent division by zero. Defaults to 1.0.
        square (bool): If True, squares the input tensors before computing the loss,
            which can emphasize large predictions. Defaults to False.

    Methods:
        forward(x, y_onehot, loss_mask=None):
            Computes the Soft Dice Loss for the given inputs.
    """

    def __init__(self, batch_dice=False, dice_per=False, square=False, smooth=1.0):
        super(SoftDiceLoss, self).__init__()

        self.batch_dice = batch_dice
        self.smooth = smooth
        self.square = square

    def forward(self, x, y_onehot, loss_mask=None):

        if self.batch_dice:
            axes = [0] + list(range(2, len(x.shape)))
        else:
            axes = list(range(2, len(x.shape)))

        intersect = x * y_onehot

        if self.square:

            denominator = x**2 + y_onehot**2

        else:
            denominator = x + y_onehot

        intersect = intersect.sum(axes)

        denominator = denominator.sum(axes)

        dc = ((2 * intersect) + self.smooth) / (denominator + self.smooth)

        dc = dc.mean()

        return -dc


class DiceLoss(nn.modules.Module):
    """
    Dice Loss for segmentation tasks.

    This loss function is based on the Dice coefficient and is designed for
    segmentation tasks, with optional handling of logits and batch-wise computation.

    Attributes:
        batch_dice (bool): If True, computes Dice loss over the entire batch;
            otherwise, computes Dice loss for each sample individually. Defaults to True.
        from_logits (bool): If True, applies a sigmoid activation to the input logits
            before computing the loss. Defaults to True.
        name (str): The name of the loss function, set to "Dice loss".
        indices (list, optional): Specifies which indices of the input tensor to include
            in the loss calculation. If None, all indices are included. Defaults to None.
    """

    def __init__(self, batch_dice=True, from_logits=True, indices=None):
        super(DiceLoss, self).__init__()
        self.batch_dice = batch_dice
        self.from_logits = from_logits
        self.name = "Dice loss"
        self.indices = indices

    def forward(self, input, target, mask=None):
        if self.indices is not None:
            input_target = input[:, self.indices]
        else:
            input_target = input
        assert input_target.shape == target.shape
        if mask is not None:
            if self.indices is not None:
                mask_target = mask[:, self.indices]
            else:
                mask_target = mask
            assert input_target.shape == mask_target.shape
        if self.from_logits:
            sigmoid_input = input_target.sigmoid()
        else:
            sigmoid_input = input_target
        if mask is not None:
            sigmoid_input = sigmoid_input * mask_target

        return 1 + SoftDiceLoss(batch_dice=self.batch_dice, square=False)(
            target, sigmoid_input
        )
