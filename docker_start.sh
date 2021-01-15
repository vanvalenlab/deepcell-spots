docker run --gpus '"device=0"'  -it \
    -p 92:8888 \
    -v /home/mly/deepcell-spots-new/deepcell_spots:/usr/local/lib/python3.6/dist-packages/deepcell_spots-new \
    -v /home/mly/deepcell-spots-new/notebooks:/notebooks \
    -v /home/mly/spot_training_data:/data \
    mly/deepcell-spots-new:latest
