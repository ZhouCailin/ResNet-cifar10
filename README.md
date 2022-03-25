# ResNet-cifar10
Mini-project for NYU Tandon ECE Deep Learning course.

## Requirement
- Python 3.7
- Pytorch 1.11.0

## Training
```
python project1_model.py
```

## Architecture
|    layer name    | output size | convolution | stride | blocks |   params  |
|:----------------:|:-----------:|:-----------:|:------:|:------:|:---------:|
|     init_conv    |    32,32    |    5,5,64   |    1   |        |   4,928   |
|    res_layer_1   |    32,32    |    5,5,64   |    1   |    3   |  615,168  |
|    res_layer_2   |    16,16    |   5,5,128   |    2   |    5   | 1,477,376 |
|    res_layer_3   |     8,8     |   5,5,256   |    2   |    2   | 2,361,856 |
|    avg_pool_4    |     2,2     |   4,4,256   |    4   |        |           |
| fc-1024, softmax |        |             |        |        |   10,250  |
|   Total params   |             |             |        |        | 4,469,578 |

## Accuracy
| Test Error            | Acc.        |
| ----------------- | ----------- |
| 0.0893             | 92.64%      |