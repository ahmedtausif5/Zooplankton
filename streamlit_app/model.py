import torch.nn as nn
import torchvision.models as models

# ==========================================
# BIOLOGICAL MAPPINGS & CONSTANTS
# ==========================================
LEVEL_0_MAP = {"Not-Zooplankton": 0, "Zooplankton": 1}
LEVEL_1_MAP = {"Cladocera": 0, "Rotifer": 1, "Copepoda": 2, "Bubble": 3, "Exoskeleton": 4, "Fiber": 5, "Plant_Matter": 6, "Unknown": -1}
LEVEL_2_MAP = {"Bosmina": 0, "Daphnia": 1, "Nauplius": 2, "Cyclopoid": 3, "Harpacticoid": 4, "Calanoid": 5, "Unknown": -1}

REVERSE_L0_MAP = {value:(key if value != -1 else "Terminal Parent or Unknown") for key,value in LEVEL_0_MAP.items()}
REVERSE_L1_MAP = {value:(key if value != -1 else "Terminal Parent or Unknown") for key,value in LEVEL_1_MAP.items()}
REVERSE_L2_MAP = {value:(key if value != -1 else "Terminal Parent or Unknown") for key,value in LEVEL_2_MAP.items()}

VALID_L0_MAPPING = {"ROOT": [0, 1]}
VALID_L1_MAPPING = {1: [0, 1, 2], 0: [3, 4, 5, 6]}
VALID_L2_MAPPING = {0: [0, 1], 2: [2, 3, 4, 5]}

# ==========================================
# NEURAL NETWORK ARCHITECTURE
# ==========================================
class HierarchicalZooplanktonViT(nn.Module):
    """
    A vision transformer with multi-head outputs for hierarchical classification.
    """
    def __init__(self, pretrained=False, freeze_backbone=True):
        super().__init__()
        weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
        self.backbone = models.vit_b_16(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity()

        # Defining the three separate hierarchical heads
        self.head_l0 = nn.Linear(in_features, 2)
        self.head_l1 = nn.Linear(in_features, 7)
        self.head_l2 = nn.Linear(in_features, 6)

    def forward(self, x):
        features = self.backbone(x)
        return {
            'L0': self.head_l0(features),
            'L1': self.head_l1(features),
            'L2': self.head_l2(features)
        }