conda create --prefix ./myenv python=3.10
conda activate ./myenv
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python train.py