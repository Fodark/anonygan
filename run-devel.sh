docker run --gpus '"device=0,1"' --rm -ti --shm-size=64gb -v $PWD:/src -v /raid/home/dvl/datasets:/datasets dvl-anonygan-ndallasen:devel ./train.sh
