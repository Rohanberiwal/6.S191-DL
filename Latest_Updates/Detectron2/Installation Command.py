!apt-get install -y python3-opencv
!pip install pydot pydot-ng
!pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# Step 2: Install Detectron2
!pip install 'git+https://github.com/facebookresearch/detectron2.git'
