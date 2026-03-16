# UCF-101 Data Generation
First, prepare for optical flow and pose extraction:
- Download the data [link](https://www.crcv.ucf.edu/data/UCF101.php)
  - You must download the *UCF101 data set* itself AND the *Train/Test Splits for Action Recognition* (text files)
  - **NOTE:** The extraction pipeline assumes that the dataset and train/test split .txt files are within the same directory initially as follows:
    ```
    ../Datasets/UCF-101
    |-- ApplyEyeMakeup
    |   |-- v_ApplyEyeMakeup_g01_c01.avi
    |   |-- ...
    |   \-- v_ApplyEyeMakeup_g25_c07.avi
    |-- ...
    |-- testlist01.txt
    |-- ...
    \-- trainlist03.txt
    ```
- In the config files `./data/config/ucf101/*.yaml` update the following properties:
  - extractor: data_paths: rgb_path: 
    - Default is `../Datasets/UCF-101/`

Run the bash script to generate annotations and extract the optical flow, pose and then flowpose for the ucf101 dataset.

``` bash
bash ./data_gen/ucf101/ucf101_extract.sh
```


Uncomment the appropriate line in [the extraction file](data_gen/UCF-101_extract.sh). Optical flow and pose extraction must precede flowpose extraction.

From the root directory, sbatch the extraction file.
```
bash data_gen/ucf101/UCF-101_extract.sh
```

Depending on whether you're extracting optical flow or poses, this may take a few hours. 

TODO: Make sure this section makes sense. Is there a better way to go about doing it?
NOTE: You must add this folder to your PYTHONPATH
```
export PYTHONPATH="${PYTHONPATH}:/path/to/this/folder/MS-G3D"
```

## NTU RGB+D / NTU RGB+D120 Extract
For each dataset, the following files will need to be run in this order:
| File | Time | Memory | CUDA |
| --- | --- | --- | --- |
| `get_raw_skes_data.py` | 1.5 hours | 10g? | ❌ |
| `get_raw_denoised_data.py` | 20 minutes | 6g | ❌ |
| `get_flowpose_samples.py` (batches of 2000) | ~8.5 hours | 25g | ✔ |
| the code block below (to combine each extraction) | 10 minutes | 75g | ❌ |
| `seq_transformation.py` | 2 hours | 450g? | ❌ |

Edit the file `data_gen/NTU/ntu_gendata.sh` and comment out each section as you go.
**NOTE:**: some work may need to be done to ensure that you're using the correct form of optical flow extraction.
Refer to the table above for resources needed for these extractions.

<u>get_raw_skes_data.py</u>
- Extracts raw skeleton data, saving the skeleton data, frame counts, and frame drops to individual pickle files.
<u>get_raw_denoised_data.py</u>
- Cleans up the data, denoising for length, missing frames, spread, motion, etc. 
<u>get_flowpose_samples.py</u>
- Processes in batches of 2000 by default, gets *flow* data surrounding each keypoint, saving temporary files for each batch as pickle files which are concatenated with `concat_flow`.
<u>seq_transformation.py</u>
- Aligns the sequences, this runs twice, first to split the dataset into `x_train`, `y_train`, `x_test`, and `y_test` splits for each evaluation (e.g. `CV` and `CS` for ntu), saving the full dataset as a `.npz` file (the pose and PoseOFF datasets are saved separately), then runs again using the `data_gen.utils.postprocess.create_aligned_dataset()` which aligns *just* the skeletons, then appends the flow data to this. Each split (e.g. `x_test`) is of shape `(N, T, (M V C))`, whic corresponds to (number of samples in split, frames (300), (bodies, vertices, channels)).

```python
dataset = 'ntu120'
data_path = f'./data/{dataset}/flow_data'
files = os.listdir(data_path)
files.sort()

flow_data = []

# Iterate over the files
for file in files:
    print(f'Appending {file}')
    with open(osp.join(in_path,file), 'rb') as fr:  # load raw skeletons data
        data = pickle.load(fr)
    
    for sample in data:
        flow_data.append(sample)
    print(f'Flow data samples: {len(flow_data)}')


with open(osp.join(data_path, 'flow_data.pkl'), 'wb') as f:
    pickle.dump(flow_data, f, pickle.HIGHEST_PROTOCOL)
```

## NW-UCLA extract
- Download the data (put it in any folder you like...)
- The val_labels.pkl are already in the `./data/nucla/statistics/` folder

## Graphing data extraction
- Find a good video sample using `data/visualisations/data_vis.py`
    - In "__main__" section, provide a `video_name` and if it's not suitable, possible video names will be provided

- Extract the optical flow using:
    ```bash
    srun --time=0:05:00 --gres=gpu:1 --mem-per-cpu=16G python data/visualisations/get_flow_samples.py --dataset ntu --sample_name [SAMPLE_NAME]
    ```
    - This creates a `.npz` file with keys `["pose", "rgb", "flow"]` and saves it as:
        - `data/visualisations/RAW/[SAMPLE_NAME].npz`

- 
