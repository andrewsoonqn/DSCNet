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

##### Temporary MLflow UI:

Training writes directly to the shared MLflow database and artifact directory. The UI does not need to be running during training.

From the repository root on the laptop:

1. Start the temporary login-node reader and localhost-only SSH tunnel:

   `.venv/bin/python tools/expctl.py ui start`

2. Open the returned `http://127.0.0.1:5000` URL in a browser. Enter the returned reader username and password when prompted.

3. Check both the remote reader and local tunnel:

   `.venv/bin/python tools/expctl.py ui status`

4. Stop both processes when finished:

   `.venv/bin/python tools/expctl.py ui stop`

Use `--local-port <port>` after `ui start` if port 5000 is already needed by another local process. The remote reader always binds to `127.0.0.1`, and the SSH tunnel always binds to local `127.0.0.1`; neither process listens on a public interface. Because the login node is shared, loopback binding alone does not isolate users: the reader also requires a project-specific random basic-auth credential stored in a user-private cluster directory. The configured 60-minute timeout stops the reader and tunnel without deleting runs or interrupting training.

MLflow metadata is stored at `/home/a/andrewsq/data/urop/experiments/mlflow.db`. Artifacts are stored at `/home/a/andrewsq/data/urop/experiments/mlflow-artifacts`. `expctl` does not automatically delete either location. No separate project backup is configured or assumed, so retrieve the declared run artifacts needed for recovery and monitor the experiment filesystem quota. Automatic retrieval fails closed above 8 GiB per file or 12 GiB per run. The current standard and optimized model checkpoint estimate, including populated Adam state, is about 0.052 GiB and fits those limits.
