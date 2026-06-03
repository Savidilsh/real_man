# Mosaic3D Full-Floor Pipeline

This file is the detailed pipeline reference copied from the working development repo. For clean setup and current commands, start with the root `README.md` in this repository.

This folder processes a large floor point cloud automatically with a room-first pipeline:

```text
preprocess point cloud
→ Point2Graph-style room proposals
→ geometry-refined room masks
→ full point-to-room ownership assignment
→ for each room:
     if room points <= 900k: run Mosaic3D on the full room
     else: split into the largest possible overlapping chunks
→ merge chunk predictions back into one global semantic map
→ globally merge object instances
→ assign each object to a room
→ write Point2Graph-style scene_graph.json
```

The important rule is: scene graphs are built **after** global stitching. Per-chunk scene graphs are not used, because they create duplicate objects and wrong object locations.

The public `zimingluo/Point2Graph` repository was inspected locally under `third_party/Point2Graph`. Its README states that the released code is the object detection/classification module, not a callable room-detection module. This pipeline therefore includes a **Point2Graph-style room layer** that follows the paper/project room method: Z slicing, XY grid projection, border-enhanced density map generation, and connected room-region extraction.

## What It Produces

For one full floor, the pipeline writes:

- `01_preprocess/floor_infer.npz`: downsampled inference cloud with global point IDs
- `02_rooms/rooms.json`: initial Point2Graph-style room proposals
- `02_rooms_refined/rooms.json`: cleaned, split/merged, wall-snapped rooms plus full point ownership used for chunking
- `02_chunks/chunk_*.npy`: full-room or split-room Mosaic3D inputs
- `03_inference/chunk_*/chunk_*_mosaic3d_predictions.npz`: per-chunk Mosaic3D predictions
- `04_stitched/floor_mosaic3d_stitched_predictions.npz`: one global stitched prediction
- `04_stitched/floor_mosaic3d_stitched_colorized.ply`: one global semantic coloured point cloud
- `04_stitched/floor_instances.json`: object instances from global connected-component clustering
- `04_stitched/floor_instances_colorized.ply`: object-instance visualization
- `04_stitched/scene_graph.json`: object locations and spatial relations
- `04_stitched/qa_flags.json`: failed chunks, unlabeled points, limited chunks

## Web Label QA Viewer

After stitching, open the browser QA viewer to inspect semantic labels, instance labels, rooms, and object locations together:

```bash
cd /home/dongan/ws_demo/Mosaic3D

bash mosaic3d_floor_pipeline/run_label_viewer.sh \
  --work-dir /home/dongan/ws_demo/Mosaic3D/mosaic3d_floor_pipeline/runs/robotics_floor \
  --host 127.0.0.1 \
  --port 8898 \
  --max-points 2000000
```

Then open:

```text
http://127.0.0.1:8898
```

If you are using SSH from another machine, forward the port:

```bash
ssh -L 8898:127.0.0.1:8898 server_user@server_ip
```

The viewer has:

- **Semantic** color mode: points colored by class label
- **Instance** color mode: each object instance gets its own color
- **Original** color mode: original point colors
- floating object labels at instance centroids
- object filtering by class/name/room
- click an object in the right panel to focus the camera on it

Use this for quick QA: if the label says `couch_1` but the highlighted/centered object is a desk, the semantic or instance output is wrong.

## One-Command Run

From the Mosaic3D repo root:

```bash
cd /home/dongan/ws_demo/Mosaic3D

python mosaic3d_floor_pipeline/process_full_floor.py \
  --input /home/dongan/ws_demo/Realman_SG/RoboticsInstitute-pointcloud-fine.las \
  --work-dir /home/dongan/ws_demo/Mosaic3D/mosaic3d_floor_pipeline/runs/robotics_floor \
  --ckpt ckpt_raw/converted/spunet34c.ckpt \
  --labels-file /home/dongan/ws_demo/Mosaic3D/labels/robotics_lab_recommended_labels.txt \
  --config /home/dongan/ws_demo/Mosaic3D/mosaic3d_floor_pipeline/config.example.json \
  --runner docker \
  --resume
```

Use `--runner python` only if the current Python environment already has the working Mosaic3D dependencies and CUDA setup.

## Step-By-Step Run

```bash
cd /home/dongan/ws_demo/Mosaic3D

python mosaic3d_floor_pipeline/scripts/preprocess_floor.py \
  --input /home/dongan/ws_demo/Realman_SG/RoboticsInstitute-pointcloud-fine.las \
  --work-dir mosaic3d_floor_pipeline/runs/robotics_floor \
  --voxel-size 0.03 \
  --preview-ply

python mosaic3d_floor_pipeline/scripts/detect_rooms_point2graph.py \
  --preprocess-manifest mosaic3d_floor_pipeline/runs/robotics_floor/01_preprocess/preprocess_manifest.json \
  --work-dir mosaic3d_floor_pipeline/runs/robotics_floor \
  --preview-ply

python mosaic3d_floor_pipeline/scripts/refine_room_proposals.py \
  --preprocess-manifest mosaic3d_floor_pipeline/runs/robotics_floor/01_preprocess/preprocess_manifest.json \
  --rooms-json mosaic3d_floor_pipeline/runs/robotics_floor/02_rooms/rooms.json \
  --work-dir mosaic3d_floor_pipeline/runs/robotics_floor \
  --preview-ply

python mosaic3d_floor_pipeline/scripts/make_chunks.py \
  --preprocess-manifest mosaic3d_floor_pipeline/runs/robotics_floor/01_preprocess/preprocess_manifest.json \
  --work-dir mosaic3d_floor_pipeline/runs/robotics_floor \
  --halo 0.8 \
  --room-full-max-points 900000 \
  --max-points 900000 \
  --room-polygons mosaic3d_floor_pipeline/runs/robotics_floor/02_rooms_refined/rooms.json

python mosaic3d_floor_pipeline/scripts/run_mosaic3d_batch.py \
  --chunks-manifest mosaic3d_floor_pipeline/runs/robotics_floor/02_chunks/chunks_manifest.json \
  --work-dir mosaic3d_floor_pipeline/runs/robotics_floor \
  --ckpt ckpt_raw/converted/spunet34c.ckpt \
  --labels-file labels/robotics_lab_recommended_labels.txt \
  --runner docker \
  --resume

python mosaic3d_floor_pipeline/scripts/stitch_scene_graph.py \
  --preprocess-manifest mosaic3d_floor_pipeline/runs/robotics_floor/01_preprocess/preprocess_manifest.json \
  --chunks-manifest mosaic3d_floor_pipeline/runs/robotics_floor/02_chunks/chunks_manifest.json \
  --batch-manifest mosaic3d_floor_pipeline/runs/robotics_floor/03_inference/batch_manifest.json \
  --work-dir mosaic3d_floor_pipeline/runs/robotics_floor \
  --object-relabel \
  --object-relabel-candidates all
```

## Exact Room-First Mosaic3D Policy

After room detection, the chunker follows this policy:

1. For each detected room, count the room points.
2. If the room has `<= room_full_max_points`, write one `room_full` Mosaic3D input containing the full room.
3. If the room is larger, recursively split the room along the densest/longest XY axis.
4. Splitting stops as soon as each chunk including halo context is `<= max_points`.
5. The split chunks are therefore the largest chunks the configured Mosaic3D point limit can process.

Each split-room chunk has:

- a **core** area where predictions are trusted most
- a **halo** area around it for Mosaic3D context
- a global point index for every point
- a center weight used during stitching

During stitching, if a point appears in multiple chunks:

```text
final_label = weighted vote(label, Mosaic3D confidence, distance from chunk border)
```

This reduces bad labels at chunk boundaries.

If `object_relabel` is enabled, the stitch stage also does a second object-level pass after masks are created:

```text
final instance mask
→ collect Mosaic3D logits for points inside that mask
→ keep the most reliable point observations first
   (high Mosaic3D confidence, near the chunk core, not halo/boundary)
→ average the remaining label probabilities over the whole object
→ reject geometrically implausible label candidates if enabled
   (for example a 20m-wide mask cannot be accepted as a door)
→ store top label candidates
→ optionally replace the object's inherited semantic label
```

This keeps the mask geometry unchanged. It only updates the object label fields in `floor_instances.json` and `scene_graph.json`.

The reliable-point behavior is controlled by:

```json
"object_relabel_top_point_fraction": 0.75,
"object_relabel_min_point_score": 0.0,
"object_relabel_min_center_weight": 0.35,
"object_relabel_geometry_filter": true
```

Default limits are set for your requested range:

```json
"room_full_max_points": 900000,
"max_points_per_chunk": 900000,
"mosaic3d_max_points": 1000000
```

If GPU memory fails, reduce these to `750000`, then `500000`.

## Point2Graph-Style Room Detection

By default, `process_full_floor.py` runs the built-in Point2Graph-style room detector after preprocessing:

```text
preprocess point cloud
→ estimate floor Z
→ use mostly wall-height points
→ project multiple height slices into XY grids
→ build persistent wall/border map
→ suppress low-only furniture-like clutter
→ combine with floor footprint map
→ extract connected room proposals
→ refine proposals geometrically
→ write refined rooms.json
```

The proposal output is:

```text
02_rooms/rooms.json
02_rooms/room_point_indices.npz
02_rooms/rooms_preview.ply
02_rooms/room_label_map.npy
02_rooms/room_debug_maps.npz
```

The refined room output is:

```text
02_rooms_refined/rooms.json
02_rooms_refined/room_point_indices.npz
02_rooms_refined/rooms_preview.ply
02_rooms_refined/room_label_map.npy
02_rooms_refined/room_owner_index.npy
02_rooms_refined/room_refine_debug_maps.npz
```

The refinement stage performs:

- cleanup of noisy proposal masks
- false-merge splitting using strong wall evidence inside a proposal
- false-split merging when adjacent regions have weak/no wall evidence
- polygon simplification
- snapping mostly horizontal/vertical polygon edges to nearby strong wall lines
- non-overlapping confident-room assignment for room-aware chunking
- full ownership assignment for every inference point:
  - confident room interior points stay with that room
  - large unowned free-space regions become `corridor_###` or `common_area_###`
  - remaining wall/border/noise points go to the nearest room/corridor/common-area owner

If you already have room polygons from another implementation, pass them instead:

```bash
python mosaic3d_floor_pipeline/process_full_floor.py \
  --input full_floor.ply \
  --work-dir mosaic3d_floor_pipeline/runs/floor_with_rooms \
  --room-polygons rooms.json \
  --resume
```

Expected `rooms.json` format:

```json
{
  "rooms": [
    {
      "room_id": "room_001",
      "room_type": "office",
      "polygon_xy": [[0.0, 0.0], [5.0, 0.0], [5.0, 4.0], [0.0, 4.0]]
    }
  ]
}
```

Room polygons are optional. Without them, the pipeline uses adaptive overlapping floor chunks and still builds a global scene graph under one `floor` room node.

To disable automatic room detection:

```bash
python mosaic3d_floor_pipeline/process_full_floor.py \
  --input full_floor.ply \
  --work-dir mosaic3d_floor_pipeline/runs/no_rooms \
  --room-detection none \
  --resume
```

Room detector tuning:

- increase `room_wall_persistence` if objects are incorrectly treated as walls
- decrease `room_wall_persistence` if walls are missed
- increase `room_wall_dilate_cells` if door gaps keep rooms connected
- increase `room_min_component_area` to ignore small false rooms
- decrease `room_grid_size` for sharper room boundaries, at higher memory cost
- increase `room_refine_split_min_area` if refinement creates too many tiny rooms
- decrease `room_refine_merge_max_wall_ratio` if rooms are incorrectly merged
- increase `room_refine_snap_distance` only if room edges are close to real wall lines but not snapped
- set `"room_refinement": false` to use raw `02_rooms/rooms.json` directly

## Scene Graph Structure

The final `04_stitched/scene_graph.json` is room/object hierarchy oriented:

```json
{
  "graph_type": "point2graph_style_scene_graph",
  "root": "floor",
  "hierarchy": {
    "floor": {
      "rooms": [
        {
          "room_id": "room_001",
          "room_type": "room",
          "objects": ["chair_1", "table_1"]
        }
      ]
    }
  },
  "objects": [
    {
      "id": "chair_1",
      "object_name": "chair_1",
      "class_name": "chair",
      "room_id": "room_001",
      "coordinates": {"x": 1.2, "y": 3.4},
      "position": {"x": 1.2, "y": 3.4, "z": 0.6}
    }
  ],
  "edges": [
    {"source": "floor", "target": "room_001", "relation": "contains"},
    {"source": "room_001", "target": "chair_1", "relation": "contains"}
  ]
}
```

## Tuning

If Mosaic3D runs out of memory:

- decrease `room_full_max_points` and `max_points_per_chunk`
- increase `voxel_size` to `0.04` or `0.05`

If objects are split at chunk borders:

- increase `halo_m` to `1.0`
- keep `max_points_per_chunk` as high as your GPU allows

If too many object instances are created:

- increase `instance_voxel_size`
- increase `instance_min_points`
- remove non-object labels from `instance_labels`

If nearby objects are merged:

- decrease `instance_voxel_size`
- increase the number of object labels in the Mosaic3D labels file
