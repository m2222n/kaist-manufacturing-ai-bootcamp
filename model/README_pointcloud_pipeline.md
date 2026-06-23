# STL → CAD Point Cloud Dataset

## Install

```bash
pip install numpy trimesh tqdm torch
```

## Convert STL folder

```bash
python stl_to_pointcloud_dataset.py \
  --input_dir ./stl_folder \
  --output_dir ./pc_dataset \
  --n_points 8192 \
  --edge_ratio 0.2 \
  --sharp_angle_deg 35 \
  --recursive \
  --export_ply
```

## Output

Each CAD produces one `.npz` and one `.json`.

Important arrays:

- `points`: canonical normalized xyz, shape `(N, 3)`
- `normals`: surface normals, shape `(N, 3)`
- `features`: `[points, normals]`, shape `(N, 6)`
- `points_raw`: sampled xyz in original CAD / assembly coordinate
- `nocs`: bbox-normalized object coordinates in `[0, 1]`
- `sample_source`: `0=surface`, `1=sharp edge`

Use `manifest.json` as the dataset index.

## PyTorch loader

```python
from cad_pointcloud_dataset import CADPointCloudDataset

train_set = CADPointCloudDataset('./pc_dataset/manifest.json', augment=True)
item = train_set[0]
print(item['features'].shape)  # torch.Size([8192, 6])
```
