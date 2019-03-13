#/bin/bash
MODEL_PATH="../pytorch-examples/cifar10/results/$1"
NET="2-4-3"
MAX_EPOCH="1888"
python plot_trajectory.py --model CifarResNetBasic --num-blocks ${NET} --model_folder ${MODEL_PATH} --dir_type weights  --ignore biasbn  --start_epoch 0 --max_epoch ${MAX_EPOCH} --save_epoch 4 --data-parallel | tee /tmp/coordinates.dat

COOR_ARGS=$(cat /tmp/coordinates.dat | grep coordinates | awk '{print "--x=" $2 ":" $3 ":51" " --y=" $4 ":" $5 ":51"}')
echo $COOR_ARGS

mpirun -n 4 python plot_surface.py --mpi --cuda --model CifarResNetBasic --num-blocks ${NET} ${COOR_ARGS} --model_file ${MODEL_PATH}/best_ckpt.t7 --dir_type weights --xignore biasbn --yignore biasbn --proj_file ${MODEL_PATH}/PCA_weights_ignore\=biasbn_save_epoch\=4/directions.h5_proj_cos.h5 --dir_file ${MODEL_PATH}/PCA_weights_ignore\=biasbn_save_epoch\=4/directions.h5  --plot --data-parallel

ls -al $MODEL_PATH/*pdf
ls -al $MODEL_PATH/*h5
ls -al $MODEL_PATH/PCA*/*
