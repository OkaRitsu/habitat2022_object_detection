FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04
RUN apt-get update && apt-get install -y \
	sudo \
	wget \
	vim \
	git
WORKDIR /opt
RUN wget https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh && \
	sh /opt/Anaconda3-2021.05-Linux-x86_64.sh -b -p /opt/anaconda3 && \
	rm -f /opt/Anaconda3-2021.05-Linux-x86_64.sh
ENV PATH /opt/anaconda3/bin:$PATH

RUN pip install --upgrade pip

WORKDIR /usr/local
RUN conda install \
	pytorch \
	torchvision \
	torchaudio \
	cudatoolkit=11.1 -c pytorch-lts -c nvidia && \
	apt-get install -y libgl1-mesa-dev

RUN git clone https://github.com/facebookresearch/detectron2.git && \
	cd detectron2 && \
	pip install -e .
RUN git clone https://github.com/facebookresearch/Detic.git --recurse-submodules && \
	cd Detic && \
	pip install -r requirements.txt
	
WORKDIR /
CMD ["bash"]
