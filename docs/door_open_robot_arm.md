# DoorOpenRobotArm


## Environment


## Install
1. [NVIDIA](https://developer.nvidia.com/isaac-gym)からIsaacGymをダウンロードする  
2. Anacondaによる環境構築
```
~$ cd isaacgym
~/isaacgym$ ./create_conda_env_rlgpu.sh 
~/isaacgym$ conda activate rlgpu 
~/isaacgym$ cd python
~/isaacgym/python$ pip install -e . 
```
3. 本リポジトリのクローン
```
~$ git clone https://github.com/open-rdc/IsaacGymEnvs.git
~$ cd ~/IsaacGymEnvs/
~/IsaacGymEnvs$ pip install -e .
```
エラーに対する対応
```
ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory # エラー内容
sudo apt install libpython3.7 # 対応方法
```  

## Training
Anacondaを起動
```
$ conda activate rlgpu
```
Fixed gain
```
python train.py task=DoorOpenRobotArm task.env.control=0
```
Variable gain
```
python train.py task=DoorOpenRobotArm task.env.control=1
```


## Test
[DoorOpenRobotArm Directory]は，学習済みモデルがあるディレクトリを選択する．  
Fixed gain
```
python train.py task=DoorOpenRobotArm checkpoint=runs/[DoorOpenRobotArm Directory]/nn/DoorOpenRobotArm.pth test=True num_envs=10 task.env.randomize_env=False task.env.test_flag=True task.env.control=0
```
Variable gain
```
python train.py task=DoorOpenRobotArm checkpoint=runs/[DoorOpenRobotArm Directory]/nn/DoorOpenRobotArm.pth test=True num_envs=10 task.env.randomize_env=False task.env.test_flag=True task.env.control=1
```
