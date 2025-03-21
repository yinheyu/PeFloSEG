This is an optimized model for peach tree flower bud estimation, which includes P2 detection layer, focal-EIoU, and model pruning techniques.
1.Training segment/train.py
2.Is focal EIoU enabled? Please check utils/loss.py
3.Determine whether to enable the P2 detection layer and select the appropriate YAML file
4.Model pruning: In train.py, sr stands for sparse training, and pruned is the coefficient for setting the pruning rate.
