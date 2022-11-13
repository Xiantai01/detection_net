from torch import nn
from .resnet_model import ResNet18, ResNet34, ResNet101, ResNet50, ResNet18_backbone, ResNet50_backbone
from .model_hrnet import LiteHRNet


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False)  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


class SimSiam(nn.Module):
    def __init__(self, arch='resnet50', feat_dim=1024, num_proj_layers=2):
        super(SimSiam, self).__init__()
        self.backbone = SimSiam.get_backbone(arch)
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.projector = projection_MLP(out_dim, feat_dim,
                                        num_proj_layers)

        self.encoder = nn.Sequential(
            self.backbone,
            self.projector
        )

        self.predictor = prediction_MLP(feat_dim)

    @staticmethod
    def get_backbone(backbone_name):
        base_channel = 16
        extra = dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('LITE', 'LITE', 'LITE'),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=(
                    (base_channel, base_channel * 2),
                    (base_channel, base_channel * 2, base_channel * 4),
                    (base_channel, base_channel * 2, base_channel * 4, base_channel * 8),
                )),
        )
        return {'resnet18': ResNet18(),
                'resnet34': ResNet34(),
                'resnet50': ResNet50(),
                'resnet101': ResNet101(),
                'lite-HRnet': LiteHRNet(extra=extra, include_top=True, in_channels=3)
                }[backbone_name]

    def forward(self, im_aug1, im_aug2):

        z1 = self.encoder(im_aug1)
        z2 = self.encoder(im_aug2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {'z1': z1, 'z2': z2, 'p1': p1, 'p2': p2}







