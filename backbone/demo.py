from backbone.model_hrnet import LiteHRNet
import torch
from backbone.Simsiam import SimSiam

base_channel = 16
extra=dict(
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
                    (base_channel, base_channel*2),
                    (base_channel, base_channel*2, base_channel*4),
                    (base_channel, base_channel*2, base_channel*4, base_channel*8),
                )),
        )
    # print(extra['stages_spec']['num_channels'][0][0])
net = LiteHRNet(extra, include_top=False, in_channels=3)
checkpoint = torch.load('ckpt_epoch_30_2_litehrnet.pth')
model2 = SimSiam()
model2.load_state_dict(checkpoint['state_dict'])
other = checkpoint['state_dict']
pretext_model = model2.backbone
model2_dict = net.state_dict()
pre_dict = pretext_model.state_dict()
state_dict = {k:v for k,v in pretext_model.items() if k in model2_dict.keys()}
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)
# a2 = model.state_dict()
# print(a1)
# img = torch.rand(2,3,256,256)
# outs = model(img)
# print(outs)
# print(model.stage2)
# return_layers = {'stage2'}
#
#
# new_backbone = create_feature_extractor(model, return_layers)