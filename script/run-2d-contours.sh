#/bin/bash
MODEL_PATH="../pytorch-examples/cifar10/results/2019-03-14_07-32-55"
NET="6-6-6-6"


mpirun -n 4 python plot_surface.py --mpi --cuda --model PlainNet --num-blocks ${NET} --x=-1:1:51 \
--model_file ${MODEL_PATH}/best_ckpt.t7 \
--dir_type weights --xnorm filter --xignore biasbn --plot --data-parallel --log

mpirun -n 4 python plot_surface.py --mpi --cuda --model PlainNet --num-blocks ${NET} --x=-1:1:51 --y=-1:1:51 \
--model_file ${MODEL_PATH}/best_ckpt.t7 \
--dir_type weights --xnorm filter --xignore biasbn --ynorm filter --yignore biasbn  --plot --data-parallel

ls -al $MODEL_PATH/*pdf
