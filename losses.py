import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    r"""Creates a criterion to measure Dice loss:

    .. math:: L(precision, recall) = 1 - (1 + \beta^2) \frac{precision \cdot recall}
        {\beta^2 \cdot precision + recall}

    The formula in terms of *Type I* and *Type II* errors:

    .. math:: L(tp, fp, fn) = \frac{(1 + \beta^2) \cdot tp} {(1 + \beta^2) \cdot fp + \beta^2 \cdot fn + fp}

    where:
         - tp - true positives;
         - fp - false positives;
         - fn - false negatives;

    Args:
        beta: Float or integer coefficient for precision and recall balance.
        class_weights: Array (``np.array``) of class weights (``len(weights) = num_classes``).
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.
        per_image: If ``True`` loss is calculated for each image in batch and then averaged,
        else loss is calculated for the whole batch.
        smooth: Value to avoid division by zero.

    Returns:
        A callable ``dice_loss`` instance. Can be used in ``model.compile(...)`` function`
        or combined with other losses.
    """

    def __init__(self, beta = 1, smooth = 1e-5, class_weights = None):
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.smooth = smooth
        self.axes = [0, 1, 2]
        self.class_weights = class_weights
    
    def average(self, x, per_image=False, class_weights=None):
        if per_image:
            x = torch.mean(x, axis=0)
        if class_weights is not None:
            x = x * class_weights
        return torch.mean(x)

    def forward(self, pr, gt):

        gt = gt.permute((0, 2, 3, 1)) # B, C, H, W -> B, H, W, C
        pr = pr.permute((0, 2, 3, 1))

        tp = torch.sum(gt * pr, axis = self.axes)
        fp = torch.sum(pr, axis = self.axes) - tp
        fn = torch.sum(gt, axis = self.axes) - tp

        score = ((1 + self.beta ** 2) * tp + self.smooth) \
                / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + self.smooth)
        
        score = self.average(score, False, self.class_weights)

        return 1 - score

class CategoricalFocalLoss(nn.Module):
    r"""Creates a criterion that measures the Categorical Focal Loss between the
    ground truth (gt) and the prediction (pr).

    .. math:: L(gt, pr) = - gt \cdot \alpha \cdot (1 - pr)^\gamma \cdot \log(pr)

    Args:
        alpha: Float or integer, the same as weighting factor in balanced cross entropy, default 0.25.
        gamma: Float or integer, focusing parameter for modulating factor (1 - p), default 2.0.
        class_indexes: Optional integer or list of integers, classes to consider, if ``None`` all classes are used.

    Returns:
        A callable ``categorical_focal_loss`` instance. Can be used in ``model.compile(...)`` function
        or combined with other losses.
    """

    def __init__(self, alpha=0.25, gamma=2., class_indexes=None):
        super(CategoricalFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_indexes = class_indexes

    def forward(self, pr, gt):

        epsilon = 1e-07
        
        gt = gt.permute((0, 2, 3, 1)) # B, C, H, W -> B, H, W, C
        pr = pr.permute((0, 2, 3, 1))

        # clip to prevent NaN's and Inf's
        pr = torch.clip(pr, epsilon, 1.0 - epsilon)

        # Calculate focal loss
        loss = - gt * (self.alpha * torch.pow((1 - pr), self.gamma) * torch.log(pr))

        return torch.mean(loss)


class CombinedLoss(nn.Module):

    def __init__(self, class_weights = None):
        super(CombinedLoss, self).__init__()

        self.dice_loss = DiceLoss(class_weights=class_weights)
        self.focal_loss = CategoricalFocalLoss()

        """
        self.dice_loss = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=False)
        #self.focal_loss = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        self.focal_loss = FocalLoss(gamma=0)
        self.cross_entropy = nn.CrossEntropyLoss()
        """

    def forward(self, pr, gt):

        dice = self.dice_loss(pr, gt)
        #focal = self.focal_loss(pr, gt)
        return dice