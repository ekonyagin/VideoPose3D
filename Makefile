# Need to specify bash in order for conda activate to work.
SHELL=/bin/bash
# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

PROJECT?=$(shell basename $(shell pwd))
WD=$(shell pwd)

DEPS_DIR=$(WD)/deps
RESULTS_DIR=$(WD)/results
CHECKPOINT_DIR=$(WD)/checkpoint

COCOAPI=$(DEPS_DIR)/cocoapi
DETECTRON=$(DEPS_DIR)/detectron

.PHONY: setup
setup:
	echo "wargning: the script will additionally create conda environment with name: $PROJECT"
	mkdir -p $(RESULTS_DIR) $(DEPS_DIR) $(CHECKPOINT_DIR) || :

	conda create -y -n $(PROJECT) python==3.7
	$(CONDA_ACTIVATE) $(PROJECT)

	conda install -y pytorch-nightly -c pytorch
	conda install -y ffmpeg

	pip install --upgrade google-api-python-client future

	# To check if Caffe2 build was successful
	python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"

	# To check if Caffe2 GPU build was successful
	# This must print a number > 0 in order to use Detectron
	[ $(shell python -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())') -ge 1 ]

	#git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
	#cd $COCOAPI/PythonAPI && make install && cd -

	git clone https://github.com/facebookresearch/detectron $(DETECTRON)

	pip install -r $(DETECTRON)/requirements.txt
	cd $(DETECTRON) && make && cd -

	python $(DETECTRON)/detectron/tests/test_spatial_narrow_as_op.py

	cp inference/infer_video.py $(DETECTRON)/tools

	wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin \
		-O $(CHECKPOINT_DIR)/pretrained_h36m_detectron_coco.bin
