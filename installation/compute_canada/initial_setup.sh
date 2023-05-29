# Install MuJoCo binary
mkdir ~/mujocodwn && cd ~/mujocodwn
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz 
mkdir ~/.mujoco
tar -xvzf ~/mujocodwn/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
rm -rf ~/mujocodwn

# Install Python packages
module load python/3.9
module load mujoco
python -m venv ~/jaxl_env
source ~/jaxl_env/bin/activate

cd ~/scratch/jaxl
pip install -e .
pip install mujoco_py
