# DSCNet Env Usage Instructions

##### Set-Up:

Navigate to the root `DSCNet` folder.

In `DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py`:

- Update the root directory to point to the local `DSCNet_3D_opensource` folder.

- Update the data directory to point to the desired data folder.

- Update the run label to the desired identifier.

- Update the default `--GPU_id` as required. (ID=0 is usually the GPU, ID=1 is the CPU).

##### Remote RL Instructions:

Connect machine to the appropriate VPN.

SSH via VSCode:

- Open VSCode

- Press `Ctrl + Shift + P` and choose `Remote SSH`

- Choose the appropriate SSH host or enter `user@host`

- Enter password and press `Enter`

- Open terminal to access server

For file transfer:

- In a local terminal, navigate to location of zipped data file

- Run: `scp file_name.zip user@host:DSCNet/Data`

Run via `tmux` to keep window open:

- Open session: `tmux new -s <session_name>`

- Create an environment if first use (see below), then activate the environment with `source DSCNetEnv/bin/activate`

- Check available resources: `htop` and `nvtop`

- Run with allocation of resources and real time log display: `OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=<num> python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label <run_label>  2>&1 | tee logs/session_$(date +%F_%H-%M-%S).log` OR configure multiple runs in `run_model.sh` and run `bash run_models.sh`

- Split panes to view `nvtop` in parallel: `Ctrl +B` then `%`, navigate panes with `Ctrl + B` then left and right arrow keys

- Scroll in `tmux`: `Ctrl + B` then `[`. `q` to exit scrolling mode

- Exit: `Ctrl + B` to go into command input mode, then `D` to detach from session

- Reconnect: `tmux ls` to check running sessions, `tmux attach -t <session _name` to reconnect

- Close session: `tmux kill-session -t <session-name>`

- Confirm process ended via `htop`: 
  
  - Sort by CPU/GPU usage with `F6`
  
  - Use arrow keys to move to desired process
  
  - Press `F9` to open the kill menu
  
  - `SIGINT` for `Ctrl + C`, `SIGTERM` for polite stop request, `SIGKILL` for force kill immediately.

- Clean tmux log scripts in root folder: `rm ~/tmux-client-*.log ~/tmux-server-*.log`

##### Environment Instructions:

Navigate to the root `DSCNet` folder.

For creation of environment (only required for the first time):

- `python -m venv DSCNetEnv`

For creation of environment on server (only required for the first time):

- `virtualenv DSCNetEnv`

For activating environment:

- `DSCNetEnv\Scripts\activate`

For activating environment on server:

- `source DSCNetEnv/bin/activate`

For installing requirements (`requirements0.txt` for Jon's laptop, `requirements_1.txt` for Jon's desktop, `requirements2.txt` for Jon's server configuration, try the above or create your own `requirements.txt` file):

- `pip install -r requirements<X>.txt`

For installing new packages:

- `pip install <pakage_name>`

For updating `requirements<X>.txt`:

- `pip freeze > requirements<X>.txt`

For deactivating environment:

- `deactivate`

##### Other local commands:

For training 2D version:

- `python DSCNet_2D_opensource/Code/DRIVE/DSCNet/S0_Main.py`

For training 3D version:

- `set OMP_NUM_THREADS=8 && set CUDA_VISIBLE_DEVICES=0 && python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label <run_label>`

For predicting 3D version with trained model:

- `python DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --run_label --Dir_Weights <path/to/weights/dir> --model_name_max --if_onlytest True`
