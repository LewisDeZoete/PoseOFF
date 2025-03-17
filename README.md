# UCF-101 Data Generation
First, prepare for optical flow and pose extraction:
- Download the data [link]()
- Update config **RGB** data path in [path_to_yaml]()
  - Default is `../Datasets/UCF-101/`

Next, create the annotations dictionary, that will be
```
python data_gen/UCF-101_annotations.py
```

Uncomment the appropriate line in [the extraction file](data_gen/UCF-101_extract.sh). Optical flow and pose extraction must precede flowpose extraction.

From the root directory, run batch the extraction file.
```
sbatch data_gen/UCF-101_extract.sh
```

Depending on whether you're extracting optical flow or poses, this may take a few hours. 

TODO: Make sure this section makes sense. Is there a better way to go about doing it?
NOTE: You must add this folder to your PYTHONPATH
```
export PYTHONPATH="${PYTHONPATH}:/path/to/this/folder/MS-G3D"
```
