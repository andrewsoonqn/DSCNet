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

- Prepare one run: `python DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --action prepare --run_label <run_label>`

- Train with validation checkpoint selection: `OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=<num> python -u DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --action train --run_label <run_label> 2>&1 | tee logs/session_$(date +%F_%H-%M-%S).log`

- Evaluate the frozen test set only after the protocol and checkpoint are fixed: `python DSCNet_3D_opensource/Code/Kipa/DSCNet/S0_Main.py --action evaluate --run_label <run_label>`

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

For activating the environment on macOS, Linux, or the server:

- `source DSCNetEnv/bin/activate`

For activating it on Windows:

- `DSCNetEnv\Scripts\activate`

For installing requirements:

- `python -m pip install -r requirements.txt`

For installing a new package:

- `python -m pip install <package_name>`

For deactivating environment:

- `deactivate`

##### Other local commands:

For preparing and training the 3D model:

- `bash run_models.sh prepare unet4`
- `bash run_models.sh train unet4`

For evaluating a trained 3D model:

- `bash run_models.sh evaluate unet4 --Dir_Weights <path/to/weights/dir> --model_name_max <checkpoint_name>`
