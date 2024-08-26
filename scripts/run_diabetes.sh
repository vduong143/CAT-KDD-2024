seed=0
gpu=0

echo "Running experiments for Diabetes dataset, seed: $seed, gpu: $gpu ..."

python tabular_experiment.py \
            --training 1 \
            --seed $seed \
            --data_name diabetes \
            --hidden_dims 64,64,32 \
            --order 2 \
            --rank 8 \
            --initial Taylor \
            --concept_dropout 0.1 \
            --output_penalty 0.05 \
            --learning_rate 0.005 \
            --gpu $gpu \
            --num_epochs 100 \
            --log_interval 25 \
            --patience 10

echo "CAT order 2 results:"
python tabular_experiment.py \
            --training 0 \
            --seed $seed \
            --data_name diabetes \
            --hidden_dims 64,64,32 \
            --order 2 \
            --rank 8 \
            --gpu $gpu

python tabular_experiment.py \
            --training 1 \
            --seed $seed \
            --data_name diabetes \
            --hidden_dims 64,64,32 \
            --order 3 \
            --rank 16 \
            --initial Taylor \
            --concept_dropout 0.1 \
            --output_penalty 0.05 \
            --learning_rate 0.005 \
            --gpu $gpu \
            --num_epochs 100 \
            --log_interval 25 \
            --patience 10

echo "CAT order 3 results:"
python tabular_experiment.py \
            --training 0 \
            --seed $seed \
            --data_name diabetes \
            --hidden_dims 64,64,32 \
            --order 3 \
            --rank 16 \
            --gpu $gpu