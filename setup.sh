# Create environment
conda create -n tat-qa python==3.7
conda activate tat-qa
pip install -r requirement.txt
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+cu+${CUDA}.html

# Download pretrained GenBERT (out_syntext_and_numeric_finetune_numeric)
cd tag_op
pip install --upgrade --no-cache-dir gdown
gdown --folder https://drive.google.com/drive/folders/1-KmhWF4Jex4gyuz1J2VANd1ycmtsggQ3
cd ..

# Prepare dataset
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --mode train
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --mode dev
PYTHONPATH=$PYTHONPATH:$(pwd):$(pwd)/tag_op python tag_op/prepare_dataset.py --mode test

# Training
PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/trainer.py --data_dir tag_op/cache/ \
--save_dir ./checkpoint --batch_size 48 --eval_batch_size 8 --max_epoch 50 --warmup 0.06 --optimizer adam --learning_rate 5e-4 \
--weight_decay 5e-5 --seed 123 --gradient_accumulation_steps 4 --bert_learning_rate 1.5e-5 --bert_weight_decay 0.01 \
--log_per_updates 50 --eps 1e-6 --init_weights_dir tag_op/genbert

# Testing
PYTHONPATH=$PYTHONPATH:$(pwd) python tag_op/predictor.py --data_dir tag_op/cache/ --test_data_dir tag_op/cache/ \\
--save_dir tag_op/ --eval_batch_size 32 --model_path ./checkpoint