# UCF-101 Data Generation
First, prepare for optical flow and pose extraction:
- Download the data [link]()
- Update config **RGB** data path in [path_to_yaml]()
  - Default is `../Datasets/UCF-101/`

Next, create the annotations dictionary, that will be
```
python data_gen/ucf101/UCF-101_annotations.py
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
| `get_flowpose_samples.py` (0-20000,20000-40000,40000-60000) | 8.5 hours | 25g | ✔ |
| the code block below (to combine each extraction) | 10 minutes | 75g | ❌ |
| `seq_transformation.py` | 2 hours | 450g? | ❌ |

Edit the file `data_gen/NTU/ntu_gendata.sh` and comment out each section as you go.
Refer to the table above for resources needed for these extractions.

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