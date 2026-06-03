# Mosaic3D Scene Graph Generation Pipeline

This is a clean wrapper around the working Mosaic3D full-floor workflow:

```text
point cloud
-> preprocessing
-> Point2Graph-style room proposals/refinement
-> room-first chunking
-> Mosaic3D per-chunk open-vocabulary segmentation
-> stitched object instances
-> scene graph JSON
-> optional interactive editing
-> edited scene graph JSON
```

The original Mosaic3D repository is not required to be modified after this copy is verified.

A precomputed Building 1B run is included at `outputs/building_1B_floor/`. Start there to inspect stitched objects, scene graphs, and saved interactive edits.

## Folder Layout

```text
configs/              Mosaic3D model configs used by inference
docker/               Docker image build/run files
floor_pipeline/       Full-floor processing, room detection, chunking, stitching, viewer server
labels/               Label lists and prompt ensembles
mosaic3d_studio/      Shared instance clustering and Three.js vendor files
scripts/              Entry-point wrappers and Mosaic3D point cloud inference script
src/                  Mosaic3D model/source modules needed by inference
checkpoints/          Local checkpoint placeholder, ignored by git
data/                 Local input data placeholder, ignored by git
outputs/              Pipeline outputs, ignored by git
```

## Requirements

Use Docker for Mosaic3D inference. The host Python only needs the pipeline utilities and viewer dependencies.

Host utilities:

```bash
python -m pip install -r requirements-host.txt
```

Docker image:

```bash
bash docker/docker_build.sh
```

The Docker image tag expected by the configs is:

```text
mosaic3d:latest
```

## Checkpoint

Use the converted Mosaic3D checkpoint:

```text
ckpt_raw/converted/spunet34c.ckpt
```

Checkpoint files are local runtime assets and are ignored by git.

## Main Full-Floor Command

Example for Building 1B:

```bash
cd /home/dongan/Downloads/test/real_man/scene_graph_generation

bash scripts/run_full_floor.sh \
  --input data/building_1B.ply \
  --work-dir outputs/building_1B_floor \
  --ckpt ckpt_raw/converted/spunet34c.ckpt \
  --labels-file labels/cv_dep_labels.txt \
  --config floor_pipeline/config.building_1B.json \
  --runner docker \
  --room-detection point2graph \
  --resume
```

Resume only Mosaic3D batch inference:

```bash
bash scripts/run_full_floor.sh \
  --input data/building_1B.ply \
  --work-dir outputs/building_1B_floor \
  --ckpt ckpt_raw/converted/spunet34c.ckpt \
  --labels-file labels/cv_dep_labels.txt \
  --config floor_pipeline/config.building_1B.json \
  --runner docker \
  --room-detection point2graph \
  --start-at batch \
  --resume
```

## Output Scene Graphs

Original stitched output:

```text
outputs/building_1B_floor/04_stitched/scene_graph.json
outputs/building_1B_floor/04_stitched/scene_graph_detailed.json
outputs/building_1B_floor/04_stitched/floor_instances.json
```

Interactive edited output:

```text
outputs/building_1B_floor/05_interactive_edits/session_*/scene_graph.json
outputs/building_1B_floor/05_interactive_edits/session_*/scene_graph_detailed.json
outputs/building_1B_floor/05_interactive_edits/session_*/floor_instances.json
```

`scene_graph.json` is the readable room/object tree format.

`scene_graph_detailed.json` is the full graph with rooms, objects, nodes, and edges.

## Interactive Viewer

Open a run:

```bash
bash scripts/run_label_viewer.sh \
  --work-dir outputs/building_1B_floor \
  --host 127.0.0.1 \
  --port 8902 \
  --max-points 6000000
```

Reopen the latest saved edits:

```bash
bash scripts/run_label_viewer.sh \
  --work-dir outputs/building_1B_floor \
  --load-latest-edits \
  --host 127.0.0.1 \
  --port 8902 \
  --max-points 6000000
```

The viewer can:

- edit object labels
- delete individual objects
- delete/restore a full class
- add manual segments
- save edited outputs

Main controls:

- **Color** switches between semantic and instance coloring.
- **Point Size** increases/decreases point visibility.
- **Labels** toggles label overlays in the 3D view.
- **Top** changes to a top-down view.
- **Reset** resets the camera.
- **Filter** narrows the object list by label or object name.
- **Min Points** hides small objects from the list.
- **Max Labels** limits how many label overlays are drawn.

Edit one object:

1. Click an object in the object list.
2. Change the text in **Edit Object -> Label**.
3. Use **Active in saved output** to keep or remove that object from saved outputs.
4. Click **Apply**.
5. Click **Save Outputs** when finished.

Delete one object:

1. Click the object in the list.
2. Uncheck **Active in saved output**.
3. Click **Apply**.
4. Click **Save Outputs**.

Delete or restore a full class:

1. Use **Bulk Class Edit -> Class**.
2. Select the class, for example `window`.
3. Click **Delete Class** or **Restore Class**.
4. Click **Save Outputs**.

Add a manual segment:

1. Click **Segment** in the bottom toolbar.
2. Click a point in the 3D view near the object to create a preview mask.
3. Set the **Manual Segment -> Label**.
4. Adjust **Radius m** and **Voxel m** if the preview is too small or too large.
5. Click **Accept Segment**.
6. Click **Save Outputs**.

After editing, click **Save Outputs**. The edited scene graph is saved under:

```text
05_interactive_edits/session_*/
```

Each save creates a timestamped session containing:

```text
floor_instances.json
scene_graph.json
scene_graph_detailed.json
edit_summary.json
```

Use `--load-latest-edits` to reopen the newest saved edit session instead of the original stitched output.

## Quick Smoke Tests

Syntax/import checks:

```bash
python -B -m py_compile \
  floor_pipeline/process_full_floor.py \
  floor_pipeline/scripts/*.py \
  floor_pipeline/label_viewer/server.py \
  scripts/infer_pointcloud.py
```

Dry-run Mosaic3D batch command generation after chunks exist:

```bash
bash scripts/run_full_floor.sh \
  --input /path/to/input.ply \
  --work-dir outputs/test_run \
  --ckpt ckpt_raw/converted/spunet34c.ckpt \
  --labels-file labels/cv_dep_labels.txt \
  --config floor_pipeline/config.building_1B.json \
  --runner print \
  --room-detection point2graph \
  --start-at batch \
  --resume
```

## Notes

- `runs/`, `outputs/`, point clouds, and checkpoints are intentionally ignored by git.
- CloudCompare is used for `.bin` conversion and optional spatial subsampling. If needed, set:

```bash
export CLOUDCOMPARE_BIN=/snap/bin/cloudcompare.CloudCompare
```
