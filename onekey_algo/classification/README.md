## run_classification
```shell
$ python run_classification.py -h
usage: run_classification.py [-h] [--train [TRAIN [TRAIN ...]]]
                             [--valid [VALID [VALID ...]]]
                             [--labels_file LABELS_FILE] [-j J]
                             [--max2use MAX2USE]
                             [--data_pattern [DATA_PATTERN [DATA_PATTERN ...]]]
                             [--normalize_method {-1+1,imagenet}]
                             [--model_name MODEL_NAME]
                             [--gpus [GPUS [GPUS ...]]]
                             [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                             [--init_lr INIT_LR] [--optimizer OPTIMIZER]
                             [--retrain RETRAIN] [--model_root MODEL_ROOT]
                             [--iters_verbose ITERS_VERBOSE]
                             [--iters_start ITERS_START] [--pretrained]

PyTorch Classification Training

optional arguments:
  -h, --help            show this help message and exit
  --train [TRAIN [TRAIN ...]]
                        Training dataset
  --valid [VALID [VALID ...]]
                        Validation dataset
  --labels_file LABELS_FILE
                        Labels file
  -j J, --worker J      Number of workers.(default=1)
  --max2use MAX2USE     Maximum number of sample per class to be used!
  --data_pattern [DATA_PATTERN [DATA_PATTERN ...]]
                        Where to save origin image data.
  --normalize_method {-1+1,imagenet}
                        Normalize method.
  --model_name MODEL_NAME
                        Model name
  --gpus [GPUS [GPUS ...]]
                        GPU index to be used!
  --batch_size BATCH_SIZE
  --epochs EPOCHS       number of total epochs to run
  --init_lr INIT_LR     initial learning rate
  --optimizer OPTIMIZER
                        Optimizer
  --retrain RETRAIN     Retrain from path
  --model_root MODEL_ROOT
                        path where to save
  --iters_verbose ITERS_VERBOSE
                        print frequency
  --iters_start ITERS_START
                        Iters start
  --pretrained          Use pretrained core or not
```

### 常用参数

* `train`，`valid`：分别指定训练集和测试集，均不能为空。数据集格式支持两种。
  1. FolderDataset，传入对应数据的上级目录，每一个类别的数据放在一个文件夹中。
  2. ListDataset，传入的为文本文件，行一个样本，支持2列和6列数据，其中两列的输入最为常用，第一列为文件名称，第二列标签。6列数据中后四列为(`x`, `y`, `width`, `height`)即对bbox内部的物体进行分类。

* `labels_file`：当数据集为FolderDataset时，只会使用`labels_file`中指定的标签进行训练，文件格式为每行一个标签。
* `data_pattern`：当数据集为ListDataset时，文件路径最终为`data_pattern`\\`file_name`。
* `-j`：读取数据的并发度。最大建议为CPU的线程数。
* `max2use`：数据集中的每个标签最大使用的样本数。默认全部使用。
* `normalize_method`：对数据正则化的方法，具体操作为减均值除方差。
* `model_name`：选用的模型名称，例如`resnet18`、`inception_v3`等。
* `gpus`：使用到的GPU序号。
* `retrain`：接着上次训练保存的参数继续训练。
* `model_root`：模型参数存储的位置。
* `iters_verbose`：多少次迭代打印一次log。
* `iters_start`：指定当前训练的步数。
* `pretrained`：开关参数，是否使用预训练的参数。

### COVID-19
``shell
python run_classification.py --train $DS_ROOT/data.txt --valid $DS_ROOT/data.txt --labels_file $DS_ROOT/labels.txt
 --data_pattern $DS_ROOT/images/ --model_root $DS_ROOT/model -j 2 --batch_size 16 --init_lr 0.1 --epoch 300
``