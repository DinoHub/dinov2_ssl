WORKSPACE=/media/data/dinov2_ssl
DATA=/media/data/datasets

docker run -it --rm \
	--gpus all \
	-w $WORKSPACE \
	-v $WORKSPACE:$WORKSPACE \
	-v $DATA:$DATA \
	--ipc host \
	--ulimit memlock=-1 \
	--ulimit stack=67108864 \
	vit_dinov2:4.0
