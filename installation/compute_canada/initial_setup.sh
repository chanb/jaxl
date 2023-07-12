# Install MuJoCo binary
mkdir ~/mujocodwn
mkdir ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/mujocodwn
tar -xvzf ~/mujocodwn/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
rm -rf ~/mujocodwn

# Install Python packages
module load python/3.9
module load mujoco
python -m venv ~/jaxl_env
source ~/jaxl_env/bin/activate

cd ~/src/jaxl
python -m pip install absl-py
python -m pip install imageio
python -m pip install pygame
python -m pip install -r requirements/minimal.txt
python -m pip install -e .
