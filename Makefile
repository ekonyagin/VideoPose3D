# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

PROJECT?=$(shell basename $(shell pwd))
WD=$(shell pwd)

DEPS_DIR=$(WD)/deps
DATA_DIR=$(WD)/data
RESULTS_DIR=$(WD)/results
CHECKPOINT_DIR=$(WD)/checkpoint

RAW_VIDEOS_DIR=$(DATA_DIR)/videos/raw
PREPROCESSED_VIDEOS_DIR=$(DATA_DIR)/videos/preprocessed
OUTPUT_VIDEOS_DIR=$(RESULTS_DIR)/videos

COCOAPI=$(DEPS_DIR)/cocoapi
DETECTRON=$(DEPS_DIR)/detectron

.PHONY: setup
setup:
	echo "wargning: the script will additionally create conda environment with name: $PROJECT"
	mkdir -p $(DEPS_DIR) $(DATA_DIR) $(RESULTS_DIR) $(CHECKPOINT_DIR) $(INPUT_VIDEOS_DIR) $(OUTPUT_VIDEOS_DIR)

	conda create -y -n $(PROJECT) python==3.7
	$(CONDA_ACTIVATE) $(PROJECT)

	conda install -y pytorch-nightly -c pytorch
	conda install -y ffmpeg

	pip install --upgrade google-api-python-client future h5py

	# To check if Caffe2 build was successful
	python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

	# To check if Caffe2 GPU build was successful
	# This must print a number > 0 in order to use Detectron
	[ $(shell python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())') -ge 1 ]

	git clone https://github.com/cocodataset/cocoapi.git $(COCOAPI) || :
	cd $(COCOAPI)/PythonAPI && make install

	git clone https://github.com/facebookresearch/detectron $(DETECTRON) || :

	pip install -r $(DETECTRON)/requirements.txt
	cd $(DETECTRON) && make

	python $(DETECTRON)/detectron/tests/test_spatial_narrow_as_op.py

	cp inference/infer_video.py $(DETECTRON)/tools

	wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin \
		-O $(CHECKPOINT_DIR)/pretrained_h36m_detectron_coco.bin


FORMAT?=MOV
VIDEOS?=$(RAW_VIDEOS_DIR)/*
CUSTOM_DATASET_NAME?=videos
ACTION?=output # Can be either output or export

.PHONY: preprocess-videos
preprocess-videos:
	cd $(DETECTRON) && python tools/infer_video.py \
		--cfg configs/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml \
		--output-dir $(PREPROCESSED_VIDEOS_DIR) \
		--image-ext $(FORMAT) \
		--wts https://dl.fbaipublicfiles.com/detectron/37698009/12_2017_baselines/e2e_keypoint_rcnn_R-101-FPN_s1x.yaml.08_45_57.YkrJgP6O/output/train/keypoints_coco_2014_train:keypoints_coco_2014_valminusminival/generalized_rcnn/model_final.pkl \
		$(RAW_VIDEOS_DIR)

.PHONY: infer-videos
infer-videos:
	cd $(DATA_DIR) && python prepare_data_2d_custom.py -i $(PREPROCESSED_VIDEOS_DIR) -o $(CUSTOM_DATASET_NAME)

	for file in $(VIDEOS); do \
		output=$(OUTPUT_VIDEOS_DIR)/$$(basename $$file | sed 's/\..[^.]*$$/.mp4/'); \
		echo Processing $$output...; \
		python run.py \
			-d custom \
			-k $(CUSTOM_DATASET_NAME) \
			-arc 3,3,3,3,3 \
			-c checkpoint \
			--evaluate pretrained_h36m_detectron_coco.bin \
			--render \
			--viz-action custom \
			--viz-camera 0 \
			--viz-size 6 \
			--viz-video $$file \
			--viz-$(ACTION) $$output \
			--viz-subject $$(basename $$file); \
		done
