
# TagOP, initialized with GenBERT's weights

In this fork we implemented BERT, initialized with GenBERT's weights, as TagOP's encoder

GenBERT: [[CODE]](https://github.com/ag1988/injecting_numeracy), [[PDF]](https://arxiv.org/pdf/2004.04487.pdf)

### Requirements

`pip install -r requirement.txt`

### Usage
#### Download pretrained GenBERT

`cd tag_op`

`pip install --upgrade --no-cache-dir gdown`

`gdown --folder https://drive.google.com/drive/folders/1-KmhWF4Jex4gyuz1J2VANd1ycmtsggQ3`

#### Prepare dataset

`python tag_op/prepare_dataset.py --mode train`

`python tag_op/prepare_dataset.py --mode dev`

Note: The result will be written into the folder `./tag_op/cache` default.

### Train

```bash
python tag_op/trainer.py --data_dir tag_op/cache/ --save_dir ./checkpoint --batch_size 48 \
--eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 \
--weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 \
--bert_weight_decay 0.01 --log_per_updates 50 --eps 1e-6 --init_weights_dir tag_op/genbert
```

### Eval

```bash
python tag_op/predictor.py --data_dir tag_op/cache/ --test_data_dir tag_op/cache/ \
--save_dir tag_op/ --eval_batch_size 32 --model_path ./checkpoint
```
