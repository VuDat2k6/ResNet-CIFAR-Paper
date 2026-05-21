# Models package for Deep Residual Learning project.

from src.models.resnet import (
    resnet20,
    resnet32,
    LabelSmoothingCrossEntropy as BaselineLabelSmoothingCrossEntropy,
)
from src.models.se_resnet import (
    seresnet20,
    seresnet32,
    seresnet20_wide,
    seresnet20_v2,
    LabelSmoothingCrossEntropy,
    CutMixCriterion,
    MixUpCutMixCriterion,
)
