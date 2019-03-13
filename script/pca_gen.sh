#/bin/bash
MODEL_PATH="../pytorch-examples/cifar10/results/$1"
NET="5-5-5"

python plot_trajectory.py --model CifarResNetBasic --num-blocks 5-5-5 --model_folder ${MODEL_PATH} --dir_type weights  --ignore biasbn  --start_epoch 0 --max_epoch 396 --save_epoch 4 --data-parallel | tee /tmp/coordinates.dat 

COOR_ARGS=$(cat /tmp/coordinates.dat | grep coordinates | awk '{print "--x=" $2 ":" $3 ":51" " --y=" $4 ":" $5 ":51"}')
echo $COOR_ARGS
ls -al $MODEL_PATH/PCA*/*pdf 
