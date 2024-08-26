seed=0
gpu=0

echo "Running experiments for UCI-HAR dataset, seed: $seed, gpu: $gpu ..."

python image_experiment.py \
            --training 1 \
            --seed $seed \
            --data_name MNIST \
            --hidden_dims 64,64,32 \
            --order 2 \
            --rank 8 \
            --initial Taylor \
            --output_penalty 0.0 \
            --learning_rate 0.01 \
            --gpu $gpu \
            --num_epochs 100 \
            --log_interval 10 \
            --patience 10

echo "CAT order 2 results:"
python image_experiment.py \
            --training 0 \
            --seed $seed \
            --data_name MNIST \
            --hidden_dims 64,64,32 \
            --order 2 \
            --rank 8 \
            --gpu $gpu

python image_experiment.py \
            --training 1 \
            --seed $seed \
            --data_name MNIST \
            --hidden_dims 64,64,32 \
            --order 3 \
            --rank 16 \
            --initial Taylor \
            --output_penalty 0.0 \
            --learning_rate 0.01 \
            --gpu $gpu \
            --num_epochs 100 \
            --log_interval 10 \
            --patience 10

echo "CAT order 3 results:"
python image_experiment.py \
            --training 0 \
            --seed $seed \
            --data_name MNIST \
            --hidden_dims 64,64,32 \
            --order 3 \
            --rank 16 \
            --gpu $gpu