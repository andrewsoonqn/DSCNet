# DSCNet Env Usage Instructions

##### Set-Up:

- Navigate to the root `DSCNet` folder.

- In `DSCNet_2D_opensource/Code/DRIVE/DSCNET/S0_Main.py`:
  
  - Update the root directory to point to the local `DSCNet_2D_opensource` folder.
  
  - Update the default `--GPU_id` as required. (ID=0 is usually the GPU, ID=1 is the CPU).

##### Running within environment:

Navigate to the root `DSCNet` folder.

For creation of environment (only required for the first time): 

```python
python -m venv DSCNetEnv
```

For activating environment: 

```python
DSCNetEnv\Scripts\activate
```

For installing requirements (`requirements0.txt` for Jon's laptop, `requirements_1.txt` for Jon's desktop, try the above or create your own `requirements.txt` file):

```python
pip install -r requirements<X>.txt
```

For installing new packages:

```python
pip install <package_name>
```

For updating `requirements<X>.txt`:

```python
pip freeze > requirements<X>.txt
```

For training 2D version:

```python
python DSCNet_2D_opensource\Code\DRIVE\DSCNet\S0_Main.py
```

For training 3D version on new dataset:

```python
python DSCNet_3D_opensource\Code\Kipa\DSCNet\S0_Main.py
```

Or:

```
set PYTORCH_ALLOC_CONF=expandable_segments:True && python DSCNet_3D_opensource\Code\Kipa\DSCNet\S0_Main.py
```

For predicting 3D version with trained model"

```python
  python DSCNet_3D_opensource\Code\Kipa\DSCNet\S0_Main.py --Dir_Weights <path\to\weights\dir> --model_name_max <name_of_model> --if_onlytest True
```

For deactivating environment:

```python
deactivate
```
