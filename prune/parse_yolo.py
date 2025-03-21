# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python path/to/models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import sys
from copy import deepcopy
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
# ROOT = ROOT.relative_to(Path.cwd())  # relative

from models.yolo import *
from models.common import *
from models.pruned_common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import fuse_conv_and_bn, initialize_weights, model_info, scale_img, select_device, time_sync


try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None
    
def parse_pruned_model(maskbndict, d, ch):  # model_dict, input_channels(3)
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    ch = [3]
    fromlayer = []  # last module bn layer name
    from_to_map = {}
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except NameError:
                pass

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        named_m_base = "model.{}".format(i)
        if m in [Conv]:
            named_m_bn = named_m_base + ".bn"

            bnc = int(maskbndict[named_m_bn].sum())
            c1, c2 = ch[f], bnc
            args = [c1, c2, *args[1:]]
            layertmp = named_m_bn
            if i>0:
                from_to_map[layertmp] = fromlayer[f]
            fromlayer.append(named_m_bn)

        elif m in [C3Pruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            named_m_cv3_bn = named_m_base + ".cv3.bn"
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = fromlayer[f]
            fromlayer.append(named_m_cv3_bn)

            cv1in = ch[f]
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            cv3out = int(maskbndict[named_m_cv3_bn].sum())
            args = [cv1in, cv1out, cv2out, cv3out, n, args[-1]]
            bottle_args = []
            chin = [cv1out]

            c3fromlayer = [named_m_cv1_bn]
            for p in range(n):
                named_m_bottle_cv1_bn = named_m_base + ".m.{}.cv1.bn".format(p)
                named_m_bottle_cv2_bn = named_m_base + ".m.{}.cv2.bn".format(p)
                bottle_cv1in = chin[-1]
                bottle_cv1out = int(maskbndict[named_m_bottle_cv1_bn].sum())
                bottle_cv2out = int(maskbndict[named_m_bottle_cv2_bn].sum())
                chin.append(bottle_cv2out)
                bottle_args.append([bottle_cv1in, bottle_cv1out, bottle_cv2out])
                from_to_map[named_m_bottle_cv1_bn] = c3fromlayer[p]
                from_to_map[named_m_bottle_cv2_bn] = named_m_bottle_cv1_bn
                c3fromlayer.append(named_m_bottle_cv2_bn)
            args.insert(4, bottle_args)
            c2 = cv3out
            n = 1
            from_to_map[named_m_cv3_bn] = [c3fromlayer[-1], named_m_cv2_bn]
        elif m in [SPPFPruned]:
            named_m_cv1_bn = named_m_base + ".cv1.bn"
            named_m_cv2_bn = named_m_base + ".cv2.bn"
            cv1in = ch[f]
            from_to_map[named_m_cv1_bn] = fromlayer[f]
            from_to_map[named_m_cv2_bn] = [named_m_cv1_bn]*4
            fromlayer.append(named_m_cv2_bn)
            cv1out = int(maskbndict[named_m_cv1_bn].sum())
            cv2out = int(maskbndict[named_m_cv2_bn].sum())
            args = [cv1in, cv1out, cv2out, *args[1:]]
            c2 = cv2out

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
            inputtmp = [fromlayer[x] for x in f]
            fromlayer.append(inputtmp)
        elif m in {Detect, Segment}:
            if m is Segment:
                from_to_map[named_m_base + ".m.0"] = fromlayer[f[0]]
                from_to_map[named_m_base + ".m.1"] = fromlayer[f[1]]
                from_to_map[named_m_base + ".m.2"] = fromlayer[f[2]]
                from_to_map[named_m_base + ".proto.cv1.bn"] = named_m_base + ".proto.cv1.bn"
                from_to_map[named_m_base + ".proto.cv2.bn"] = named_m_base + ".proto.cv2.bn"
                from_to_map[named_m_base + ".proto.cv3.bn"] = named_m_base + ".proto.cv3.bn"
            elif m is Detect:
                from_to_map[named_m_base + ".m.0"] = fromlayer[f[0]]
                from_to_map[named_m_base + ".m.1"] = fromlayer[f[1]]
                from_to_map[named_m_base + ".m.2"] = fromlayer[f[2]]
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]
            fromtmp = fromlayer[-1]
            fromlayer.append(fromtmp)

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save), from_to_map