import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';

const state = {
  metadata: null,
  pointArray: null,
  geometry: null,
  points: null,
  selectedInstanceId: null,
  labels: new Map(),
  instanceById: new Map(),
  filteredInstances: [],
  segmentMode: false,
  positivePoints: [],
  negativePoints: [],
  previewSampleHits: new Set(),
  previewSummary: null,
  previewCloud: null,
  pendingPreview: 0,
  editingLabelInstanceId: null,
};

const els = {
  viewport: document.querySelector('#viewport'),
  labelLayer: document.querySelector('#label-layer'),
  status: document.querySelector('#status'),
  summary: document.querySelector('#summary'),
  objectList: document.querySelector('#object-list'),
  selected: document.querySelector('#selected'),
  editPanel: document.querySelector('#edit-panel'),
  editLabel: document.querySelector('#edit-label'),
  editActive: document.querySelector('#edit-active'),
  applyLabel: document.querySelector('#apply-label'),
  saveEdits: document.querySelector('#save-edits'),
  saveStatus: document.querySelector('#save-status'),
  colorMode: document.querySelector('#color-mode'),
  pointSize: document.querySelector('#point-size'),
  showLabels: document.querySelector('#show-labels'),
  filter: document.querySelector('#filter'),
  minPoints: document.querySelector('#min-points'),
  maxLabels: document.querySelector('#max-labels'),
  bulkClass: document.querySelector('#bulk-class'),
  bulkDeleteClass: document.querySelector('#bulk-delete-class'),
  bulkRestoreClass: document.querySelector('#bulk-restore-class'),
  bulkStatus: document.querySelector('#bulk-status'),
  segmentModeButton: document.querySelector('#segment-mode'),
  segmentLabel: document.querySelector('#segment-label'),
  segmentRadius: document.querySelector('#segment-radius'),
  segmentVoxel: document.querySelector('#segment-voxel'),
  segmentSemantic: document.querySelector('#segment-semantic'),
  acceptSegment: document.querySelector('#accept-segment'),
  resetSegment: document.querySelector('#reset-segment'),
  segmentInfo: document.querySelector('#segment-info'),
  topView: document.querySelector('#top-view'),
  resetView: document.querySelector('#reset-view'),
};

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b1014);

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100000);
const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
els.viewport.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

const grid = new THREE.GridHelper(60, 60, 0x2f3d4d, 0x1c2734);
grid.rotation.x = Math.PI / 2;
grid.material.transparent = true;
grid.material.opacity = 0.42;
scene.add(grid);

const axes = new THREE.AxesHelper(1.5);
scene.add(axes);

function setStatus(text) {
  els.status.textContent = text;
}

function rgbCss(color) {
  const r = Math.round((color?.[0] ?? 0.5) * 255);
  const g = Math.round((color?.[1] ?? 0.5) * 255);
  const b = Math.round((color?.[2] ?? 0.5) * 255);
  return `rgb(${r}, ${g}, ${b})`;
}

function formatNumber(value) {
  return new Intl.NumberFormat().format(Number(value || 0));
}

async function getJson(url) {
  const response = await fetch(url);
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : { ok: false, error: await response.text() };
  if (!response.ok || payload.ok === false) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function postJson(url, body = {}) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const contentType = response.headers.get('content-type') || '';
  const payload = contentType.includes('application/json')
    ? await response.json()
    : { ok: false, error: await response.text() };
  if (!response.ok || payload.ok === false) {
    const message = payload.error
      ? payload.error.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()
      : `HTTP ${response.status}`;
    throw new Error(message || `HTTP ${response.status}`);
  }
  return payload;
}

function resize() {
  const rect = els.viewport.getBoundingClientRect();
  camera.aspect = Math.max(rect.width, 1) / Math.max(rect.height, 1);
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height, false);
}

function fitCamera(top = false) {
  if (!state.geometry) return;
  state.geometry.computeBoundingSphere();
  const sphere = state.geometry.boundingSphere;
  const radius = Math.max(sphere.radius, 1);
  controls.target.copy(sphere.center);
  if (top) {
    camera.position.set(sphere.center.x, sphere.center.y, sphere.center.z + radius * 1.8);
    camera.up.set(0, 1, 0);
  } else {
    camera.position.set(
      sphere.center.x + radius * 0.9,
      sphere.center.y - radius * 1.15,
      sphere.center.z + radius * 0.9,
    );
    camera.up.set(0, 0, 1);
  }
  camera.near = Math.max(radius / 1000, 0.01);
  camera.far = radius * 20;
  camera.updateProjectionMatrix();
  controls.update();
}

function buildGeometry(array, stride) {
  const count = array.length / stride;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const src = i * stride;
    const dst = i * 3;
    positions[dst] = array[src];
    positions[dst + 1] = array[src + 1];
    positions[dst + 2] = array[src + 2];
    colors[dst] = array[src + 6];
    colors[dst + 1] = array[src + 7];
    colors[dst + 2] = array[src + 8];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  return geometry;
}

function updateColors() {
  const { pointArray, metadata, geometry } = state;
  if (!pointArray || !metadata || !geometry) return;
  const stride = metadata.stride;
  const mode = els.colorMode.value;
  const offsets = { original: 3, semantic: 6, instance: 9 };
  const offset = offsets[mode] ?? 6;
  const colorAttr = geometry.getAttribute('color');
  const colors = colorAttr.array;
  const count = pointArray.length / stride;
  const selected = state.selectedInstanceId;

  for (let i = 0; i < count; i += 1) {
    const src = i * stride + offset;
    const dst = i * 3;
    const instanceId = Math.round(pointArray[i * stride + 13]);
    const instance = state.instanceById.get(instanceId);
    let r = pointArray[src];
    let g = pointArray[src + 1];
    let b = pointArray[src + 2];

    if (instance && !instance.active) {
      r *= 0.22;
      g *= 0.22;
      b *= 0.22;
    }

    if (selected !== null && selected !== undefined && instanceId >= 0) {
      if (instanceId === selected) {
        r = 1.0;
        g = 0.95;
        b = 0.05;
      } else if (mode === 'instance') {
        r *= 0.18;
        g *= 0.18;
        b *= 0.18;
      }
    }

    if (state.previewSampleHits.has(i)) {
      r = 1.0;
      g = 0.05;
      b = 0.86;
    }

    colors[dst] = r;
    colors[dst + 1] = g;
    colors[dst + 2] = b;
  }
  colorAttr.needsUpdate = true;
}

function setPointSize() {
  if (state.points) {
    state.points.material.size = Number(els.pointSize.value);
  }
  if (state.previewCloud) {
    state.previewCloud.material.size = Math.max(Number(els.pointSize.value) * 1.8, 5);
  }
}

function cleanText(value) {
  return String(value || '').trim().toLowerCase();
}

function classBase(value) {
  const key = cleanText(value)
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
  if (!key) return '';
  const parts = key.split(' ');
  if (parts.length > 1 && /^\d+$/.test(parts[parts.length - 1])) {
    parts.pop();
  }
  return parts.join(' ');
}

function matchesObjectFilter(item, text) {
  const terms = text
    .split(',')
    .map((part) => cleanText(part))
    .filter(Boolean);
  if (!terms.length) return true;

  const name = cleanText(item.name);
  const original = cleanText(item.original_label);
  const edited = cleanText(item.edited_label);
  const semantic = cleanText(item.semantic_label);
  const room = cleanText(item.room_id);

  return terms.some((term) => {
    if (semantic === term || room === term || name === term || edited === term || original === term) {
      return true;
    }
    if (name.startsWith(`${term}_`) || edited.startsWith(`${term}_`) || original.startsWith(`${term}_`)) {
      return true;
    }
    return name.includes(term)
      || edited.includes(term)
      || original.includes(term)
      || room.includes(term);
  });
}

function filterInstances() {
  if (!state.metadata) {
    state.filteredInstances = [];
    return;
  }
  const text = els.filter.value.trim().toLowerCase();
  const minPoints = Number(els.minPoints.value || 0);
  state.filteredInstances = state.metadata.instances
    .filter((item) => item.points >= minPoints)
    .filter((item) => matchesObjectFilter(item, text))
    .sort((a, b) => b.points - a.points);
}

function updateSummary() {
  if (!state.metadata) return;
  const visibleObjects = state.filteredInstances.length;
  const activeObjects = state.metadata.instances.filter((item) => item.active && item.points > 0).length;
  const manualObjects = state.metadata.instances.filter((item) => item.source === 'manual_segment').length;
  els.summary.textContent = `${formatNumber(state.metadata.sample_count)} shown from ${formatNumber(state.metadata.point_count)} points · ${formatNumber(visibleObjects)} visible of ${formatNumber(state.metadata.instances.length)} objects · ${formatNumber(activeObjects)} active · ${formatNumber(manualObjects)} manual · ${formatNumber(state.metadata.rooms.length)} rooms`;
}

function renderBulkClassOptions() {
  if (!state.metadata) return;

  const previous = els.bulkClass.value;
  const stats = new Map();
  for (const item of state.metadata.instances) {
    const labels = new Set([
      classBase(item.semantic_label),
      classBase(item.original_label),
      classBase(item.edited_label),
    ].filter(Boolean));
    for (const label of labels) {
      const row = stats.get(label) || { label, total: 0, active: 0, points: 0 };
      row.total += 1;
      row.points += Number(item.points || 0);
      if (item.active) row.active += 1;
      stats.set(label, row);
    }
  }

  const rows = [...stats.values()].sort((a, b) => {
    if (b.active !== a.active) return b.active - a.active;
    if (b.total !== a.total) return b.total - a.total;
    return a.label.localeCompare(b.label);
  });

  els.bulkClass.textContent = '';
  for (const row of rows) {
    const option = document.createElement('option');
    option.value = row.label;
    option.textContent = `${row.label} (${row.active}/${row.total}, ${formatNumber(row.points)} pts)`;
    els.bulkClass.append(option);
  }

  if (previous && rows.some((row) => row.label === previous)) {
    els.bulkClass.value = previous;
  }
  const hasOptions = rows.length > 0;
  els.bulkClass.disabled = !hasOptions;
  els.bulkDeleteClass.disabled = !hasOptions;
  els.bulkRestoreClass.disabled = !hasOptions;
}

function getSelectedInstance() {
  if (!state.metadata || state.selectedInstanceId === null || state.selectedInstanceId === undefined) {
    return null;
  }
  return state.instanceById.get(state.selectedInstanceId) || null;
}

function focusOnInstance(item) {
  const p = item.viewer_centroid;
  const target = new THREE.Vector3(p[0], p[1], p[2]);
  const currentDistance = Math.max(camera.position.distanceTo(controls.target), 1);
  controls.target.copy(target);
  camera.position.copy(target).add(new THREE.Vector3(
    currentDistance * 0.35,
    -currentDistance * 0.5,
    currentDistance * 0.3,
  ));
  camera.updateProjectionMatrix();
  controls.update();
}

async function commitInlineLabelEdit(item, value) {
  const requestedLabel = value.trim();
  if (!requestedLabel || requestedLabel === item.edited_label || requestedLabel === item.name) {
    renderLabels();
    return;
  }

  setStatus('renaming label');
  try {
    const payload = await postJson('/api/edit_instance', {
      instance_id: item.instance_id,
      edited_label: requestedLabel,
    });
    state.selectedInstanceId = item.instance_id;
    setMetadata(payload.metadata);
    els.saveStatus.textContent = 'Label applied. Save Outputs writes the edit session files.';
    setStatus('label renamed');
  } catch (error) {
    els.saveStatus.textContent = error.message;
    setStatus('label rename failed');
    renderLabels();
  }
}

async function deleteInlineLabel(item) {
  setStatus('deleting label');
  try {
    const payload = await postJson('/api/set_instance_active', {
      instance_id: item.instance_id,
      active: false,
    });
    state.editingLabelInstanceId = null;
    state.selectedInstanceId = item.instance_id;
    setMetadata(payload.metadata);
    await loadPointBuffer({ refit: false });
    els.saveStatus.textContent = 'Deleted from active output. Save Outputs writes the edit session files.';
    setStatus('label deleted');
  } catch (error) {
    els.saveStatus.textContent = error.message;
    setStatus('delete failed');
    renderLabels();
  }
}

async function setBulkClassActive(active) {
  const label = els.bulkClass.value;
  if (!label) return;
  if (!active) {
    const ok = window.confirm(`Delete all objects with class "${label}" from the active output?`);
    if (!ok) return;
  }

  setStatus(active ? 'restoring class' : 'deleting class');
  els.bulkDeleteClass.disabled = true;
  els.bulkRestoreClass.disabled = true;
  try {
    const payload = await postJson('/api/set_class_active', { label, active });
    setMetadata(payload.metadata);
    await loadPointBuffer({ refit: false });
    const action = active ? 'restored' : 'deleted';
    const message = `${formatNumber(payload.matched_count)} ${label} objects ${action}. Save Outputs writes the edit session files.`;
    els.bulkStatus.textContent = message;
    els.saveStatus.textContent = message;
    setStatus(`class ${action}`);
  } catch (error) {
    els.bulkStatus.textContent = error.message;
    els.saveStatus.textContent = error.message;
    setStatus(active ? 'class restore failed' : 'class delete failed');
    renderLabels();
  } finally {
    renderBulkClassOptions();
  }
}

function beginInlineLabelEdit(item, node) {
  if (state.segmentMode || state.editingLabelInstanceId !== null) return;
  state.editingLabelInstanceId = item.instance_id;
  state.selectedInstanceId = item.instance_id;
  renderSelectedDetails(false);
  updateColors();
  renderList();

  const original = item.edited_label || item.name || '';
  node.classList.add('editing');
  node.textContent = '';

  const editor = document.createElement('span');
  editor.className = 'label-editor-wrap';

  const input = document.createElement('input');
  input.className = 'label-editor';
  input.type = 'text';
  input.value = original;
  input.setAttribute('aria-label', 'Object label');

  const saveButton = document.createElement('button');
  saveButton.className = 'label-edit-action';
  saveButton.type = 'button';
  saveButton.textContent = 'Save';

  const deleteButton = document.createElement('button');
  deleteButton.className = 'label-edit-action danger';
  deleteButton.type = 'button';
  deleteButton.textContent = 'Delete';

  editor.append(input, saveButton, deleteButton);
  node.append(editor);

  let closed = false;
  let actionPointerDown = false;
  const finish = async (save) => {
    if (closed) return;
    closed = true;
    const value = input.value;
    state.editingLabelInstanceId = null;
    if (save) {
      await commitInlineLabelEdit(item, value);
    } else {
      renderLabels();
    }
  };

  const holdFocusForAction = (event) => {
    actionPointerDown = true;
    event.preventDefault();
    event.stopPropagation();
  };

  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      finish(true);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      finish(false);
    }
  });
  input.addEventListener('click', (event) => {
    event.stopPropagation();
  });
  input.addEventListener('pointerdown', (event) => {
    event.stopPropagation();
  });
  input.addEventListener('blur', () => {
    if (actionPointerDown) return;
    finish(true);
  });
  saveButton.addEventListener('pointerdown', holdFocusForAction);
  deleteButton.addEventListener('pointerdown', holdFocusForAction);
  saveButton.addEventListener('click', (event) => {
    event.preventDefault();
    event.stopPropagation();
    finish(true);
  });
  deleteButton.addEventListener('click', async (event) => {
    event.preventDefault();
    event.stopPropagation();
    if (closed) return;
    closed = true;
    state.editingLabelInstanceId = null;
    await deleteInlineLabel(item);
  });

  requestAnimationFrame(() => {
    input.focus();
    input.select();
  });
}

function renderSelectedDetails(focus = false) {
  const item = getSelectedInstance();
  if (!item) {
    els.selected.textContent = '';
    els.selected.hidden = true;
    els.editPanel.hidden = true;
    return;
  }

  if (focus) {
    focusOnInstance(item);
  }

  els.selected.hidden = false;
  els.editPanel.hidden = false;
  els.editLabel.value = item.edited_label || item.name || '';
  els.editActive.checked = Boolean(item.active);
  const candidates = (item.label_candidates || [])
    .slice(0, 5)
    .map((candidate) => `${candidate.label}:${Number(candidate.score || 0).toFixed(3)}`)
    .join(', ');
  const relabelText = item.relabel_changed
    ? `relabel=${item.pre_relabel_semantic_label || 'unknown'} -> ${item.semantic_label}`
    : `relabel=${item.relabel_changed === false ? 'unchanged' : 'not run'}`;
  const geometryText = item.geometry_corrected_label
    ? `\ngeometry=${item.pre_geometry_best_label || 'unknown'} -> ${item.semantic_label}`
    : '';
  els.selected.textContent = `${item.name}
class=${item.semantic_label} room=${item.room_id}
points=${formatNumber(item.points)} score=${item.mean_score.toFixed(3)} object_score=${Number(item.object_label_score || item.mean_score || 0).toFixed(3)} margin=${Number(item.object_label_margin || 0).toFixed(3)}
${relabelText}${geometryText}${candidates ? `\ncandidates=${candidates}` : ''}
source=${item.source} active=${item.active ? 'yes' : 'no'}
xyz=${item.centroid.map((v) => Number(v).toFixed(3)).join(', ')}`;
}

function selectInstance(instanceId, focus = true) {
  state.selectedInstanceId = instanceId;
  els.saveStatus.textContent = '';
  renderSelectedDetails(focus);
  els.colorMode.value = 'instance';
  updateColors();
  renderList();
  renderLabels();
}

function renderList() {
  els.objectList.textContent = '';
  const fragment = document.createDocumentFragment();
  for (const item of state.filteredInstances) {
    const row = document.createElement('button');
    row.className = 'object-row';
    if (item.instance_id === state.selectedInstanceId) row.classList.add('selected');
    if (!item.active) row.classList.add('inactive');
    if (item.source === 'manual_segment') row.classList.add('manual');
    if (item.qa_status === 'review') row.classList.add('qa-review');
    if (item.qa_status === 'likely_wrong') row.classList.add('qa-wrong');
    row.type = 'button';
    row.addEventListener('click', () => selectInstance(item.instance_id));

    const swatch = document.createElement('span');
    swatch.className = 'swatch';
    swatch.style.background = rgbCss(item.color);

    const main = document.createElement('span');
    main.className = 'object-main';
    const name = document.createElement('span');
    name.className = 'object-name';
    name.textContent = item.name;
    const meta = document.createElement('span');
    meta.className = 'object-meta';
    const metaParts = [item.semantic_label, item.room_id, `score ${item.mean_score.toFixed(3)}`];
    if (item.relabel_changed) {
      metaParts.push(`was ${item.pre_relabel_semantic_label || 'unknown'}`);
    }
    if (!item.active) metaParts.push('inactive');
    main.append(name, meta);
    meta.textContent = metaParts.join(' · ');

    const count = document.createElement('span');
    count.className = 'object-count';
    count.textContent = formatNumber(item.points);

    row.append(swatch, main, count);
    fragment.append(row);
  }
  els.objectList.append(fragment);
}

function rebuildLabelNodes() {
  els.labelLayer.textContent = '';
  state.labels.clear();
  const requestedMax = Number(els.maxLabels.value || 0);
  const labelCandidates = state.filteredInstances.filter((item) => item.active);
  const maxLabels = requestedMax <= 0 ? labelCandidates.length : Math.max(1, requestedMax);
  const selected = getSelectedInstance();
  const visible = labelCandidates.slice(0, maxLabels);
  if (selected && !visible.some((item) => item.instance_id === selected.instance_id)) {
    visible.unshift(selected);
  }
  for (const item of visible) {
    const node = document.createElement('div');
    node.className = 'label';
    node.textContent = item.name;
    node.style.borderColor = rgbCss(item.color);
    if (item.qa_status === 'review') node.classList.add('qa-review');
    if (item.qa_status === 'likely_wrong') node.classList.add('qa-wrong');
    if (item.qa_status && item.qa_status !== 'likely_correct') {
      node.textContent = `! ${item.name}`;
    }
    node.title = [item.name, ...(item.qa_reasons || [])].filter(Boolean).join('\n');
    node.tabIndex = 0;
    if (item.instance_id === state.selectedInstanceId) node.classList.add('selected');
    if (!item.active) node.classList.add('inactive');
    node.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      beginInlineLabelEdit(item, node);
    });
    node.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === 'F2') {
        event.preventDefault();
        event.stopPropagation();
        beginInlineLabelEdit(item, node);
      }
    });
    els.labelLayer.append(node);
    state.labels.set(item.instance_id, { node, item });
  }
}

function renderLabels() {
  if (!state.metadata) return;
  const enabled = els.showLabels.checked;
  els.labelLayer.style.display = enabled ? 'block' : 'none';
  if (!enabled) return;
  rebuildLabelNodes();
}

function updateLabelPositions() {
  if (!state.metadata || !els.showLabels.checked) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  const vector = new THREE.Vector3();
  for (const { node, item } of state.labels.values()) {
    const p = item.viewer_centroid;
    vector.set(p[0], p[1], p[2]).project(camera);
    const visible = vector.z > -1 && vector.z < 1;
    if (!visible) {
      node.style.display = 'none';
      continue;
    }
    const x = (vector.x * 0.5 + 0.5) * width;
    const y = (-vector.y * 0.5 + 0.5) * height;
    node.style.display = 'block';
    node.style.left = `${x}px`;
    node.style.top = `${y}px`;
  }
}

function updateFilters() {
  filterInstances();
  updateSummary();
  renderList();
  renderLabels();
}

function setMetadata(metadata) {
  state.metadata = metadata;
  state.instanceById = new Map(metadata.instances.map((item) => [item.instance_id, item]));
  if (state.selectedInstanceId !== null && !state.instanceById.has(state.selectedInstanceId)) {
    state.selectedInstanceId = null;
  }
  filterInstances();
  updateSummary();
  renderBulkClassOptions();
  renderList();
  renderSelectedDetails(false);
  renderLabels();
  updateColors();
  renderSegmentInfo();
}

function setPreviewCloud(points) {
  if (!points || !points.length) {
    if (state.previewCloud) {
      scene.remove(state.previewCloud);
      state.previewCloud.geometry.dispose();
      state.previewCloud.material.dispose();
      state.previewCloud = null;
    }
    return;
  }

  const positions = new Float32Array(points.length * 3);
  for (let i = 0; i < points.length; i += 1) {
    const dst = i * 3;
    positions[dst] = points[i][0];
    positions[dst + 1] = points[i][1];
    positions[dst + 2] = points[i][2];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const material = new THREE.PointsMaterial({
    color: 0xff0ad8,
    size: Math.max(Number(els.pointSize.value) * 1.8, 5),
    sizeAttenuation: false,
    depthTest: true,
    depthWrite: false,
  });

  if (state.previewCloud) {
    scene.remove(state.previewCloud);
    state.previewCloud.geometry.dispose();
    state.previewCloud.material.dispose();
  }
  state.previewCloud = new THREE.Points(geometry, material);
  scene.add(state.previewCloud);
}

async function loadPointBuffer({ refit = false } = {}) {
  const response = await fetch('/api/points.bin');
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  state.pointArray = new Float32Array(buffer);
  const geometry = buildGeometry(state.pointArray, state.metadata.stride);

  if (state.points) {
    state.points.geometry.dispose();
    state.points.geometry = geometry;
  } else {
    state.points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: Number(els.pointSize.value),
        vertexColors: true,
        sizeAttenuation: false,
        depthTest: true,
        depthWrite: true,
      }),
    );
    scene.add(state.points);
  }
  state.geometry = geometry;
  updateColors();
  renderLabels();
  if (refit) {
    fitCamera(false);
  }
}

async function loadData() {
  setStatus('loading metadata');
  const metadata = await getJson('/api/metadata');
  setMetadata(metadata);

  setStatus('loading point buffer');
  await loadPointBuffer({ refit: true });
  setStatus('ready');
}

async function applySelectedChanges() {
  const item = getSelectedInstance();
  if (!item) return;

  const requestedLabel = els.editLabel.value.trim();
  const requestedActive = Boolean(els.editActive.checked);
  setStatus('applying edit');

  try {
    const labelPayload = await postJson('/api/edit_instance', {
      instance_id: item.instance_id,
      edited_label: requestedLabel,
    });
    setMetadata(labelPayload.metadata);

    const updated = getSelectedInstance();
    if (updated && requestedActive !== Boolean(updated.active)) {
      const activePayload = await postJson('/api/set_instance_active', {
        instance_id: item.instance_id,
        active: requestedActive,
      });
      setMetadata(activePayload.metadata);
      await loadPointBuffer({ refit: false });
    }

    els.saveStatus.textContent = 'Applied. Save Outputs writes the edit session files.';
    setStatus('edit applied');
  } catch (error) {
    els.saveStatus.textContent = error.message;
    setStatus('edit failed');
  }
}

async function saveEdits() {
  setStatus('saving edits');
  els.saveEdits.disabled = true;
  try {
    const payload = await postJson('/api/save_edits');
    setMetadata(payload.metadata);
    const summary = payload.summary;
    els.saveStatus.textContent = `Saved ${formatNumber(summary.active_instance_count)} active objects and ${formatNumber(summary.manual_segment_count)} manual objects to ${summary.edit_dir}`;
    setStatus('saved outputs');
  } catch (error) {
    els.saveStatus.textContent = error.message;
    setStatus('save failed');
  } finally {
    els.saveEdits.disabled = false;
  }
}

function setSegmentMode(enabled) {
  state.segmentMode = Boolean(enabled);
  els.segmentModeButton.classList.toggle('active', state.segmentMode);
  els.segmentModeButton.setAttribute('aria-pressed', String(state.segmentMode));
  setStatus(state.segmentMode ? 'segment mode' : 'ready');
}

function worldPointFromSample(sampleIndex) {
  const stride = state.metadata.stride;
  const src = sampleIndex * stride;
  return [
    state.pointArray[src] + state.metadata.center[0],
    state.pointArray[src + 1] + state.metadata.center[1],
    state.pointArray[src + 2] + state.metadata.center[2],
  ];
}

function pickSample(event) {
  if (!state.points || !state.pointArray || !state.metadata) return null;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / Math.max(rect.width, 1)) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / Math.max(rect.height, 1)) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  raycaster.params.Points.threshold = Math.max(0.08, Math.min(0.5, Number(els.pointSize.value) * 0.035));
  const hits = raycaster.intersectObject(state.points, false);
  if (!hits.length) return null;
  const sampleIndex = hits[0].index;
  const instanceId = Math.round(state.pointArray[sampleIndex * state.metadata.stride + 13]);
  return {
    sampleIndex,
    instanceId,
    point: worldPointFromSample(sampleIndex),
  };
}

function renderSegmentInfo() {
  const preview = state.previewSummary;
  const pos = state.positivePoints.length;
  const neg = state.negativePoints.length;
  if (preview) {
    els.segmentInfo.textContent = `Preview ${formatNumber(preview.points)} pts, seed ${preview.seed_label}, ${pos} positive, ${neg} negative`;
  } else if (pos || neg) {
    els.segmentInfo.textContent = `${pos} positive, ${neg} negative, no preview`;
  } else {
    els.segmentInfo.textContent = 'No segment preview.';
  }
  els.acceptSegment.disabled = !preview;
  els.resetSegment.disabled = !(preview || pos || neg);
}

async function requestSegmentPreview() {
  if (!state.positivePoints.length) {
    renderSegmentInfo();
    return;
  }

  const previewId = state.pendingPreview + 1;
  state.pendingPreview = previewId;
  setStatus('previewing segment');
  try {
    const payload = await postJson('/api/segment_preview', {
      positive_points: state.positivePoints,
      negative_points: state.negativePoints,
      radius: Number(els.segmentRadius.value || 1.5),
      voxel_size: Number(els.segmentVoxel.value || 0.12),
      semantic_mode: els.segmentSemantic.value,
    });
    if (previewId !== state.pendingPreview) return;
    state.previewSampleHits = new Set(payload.sample_hits || []);
    state.previewSummary = payload.preview || null;
    setPreviewCloud(payload.preview_points || []);
    setMetadata(payload.metadata);
    setStatus('segment preview ready');
  } catch (error) {
    if (previewId !== state.pendingPreview) return;
    state.previewSampleHits = new Set();
    state.previewSummary = null;
    setPreviewCloud([]);
    renderSegmentInfo();
    updateColors();
    els.segmentInfo.textContent = error.message;
    setStatus('segment preview failed');
  }
}

async function handleSegmentClick(event, negative = false) {
  if (!state.segmentMode) return;
  event.preventDefault();
  event.stopPropagation();

  const hit = pickSample(event);
  if (!hit) {
    setStatus('no point picked');
    return;
  }
  if (negative && !state.positivePoints.length) {
    setStatus('positive point required first');
    return;
  }

  if (negative) {
    state.negativePoints.push(hit.point);
  } else {
    state.positivePoints.push(hit.point);
  }
  renderSegmentInfo();
  await requestSegmentPreview();
}

async function acceptSegment() {
  if (!state.previewSummary) return;
  setStatus('accepting segment');
  els.acceptSegment.disabled = true;
  try {
    const label = els.segmentLabel.value.trim() || state.previewSummary.seed_label || 'manual_object';
    const payload = await postJson('/api/segment_accept', { label });
    state.previewSampleHits = new Set();
    state.previewSummary = null;
    setPreviewCloud([]);
    state.positivePoints = [];
    state.negativePoints = [];
    setMetadata(payload.metadata);
    await loadPointBuffer({ refit: false });
    if (payload.new_instance) {
      selectInstance(payload.new_instance.instance_id, true);
    }
    renderSegmentInfo();
    setStatus('segment accepted');
  } catch (error) {
    els.segmentInfo.textContent = error.message;
    setStatus('segment accept failed');
  }
}

async function resetSegment() {
  state.pendingPreview += 1;
  state.previewSampleHits = new Set();
  state.previewSummary = null;
  setPreviewCloud([]);
  state.positivePoints = [];
  state.negativePoints = [];
  renderSegmentInfo();
  updateColors();
  setStatus('resetting segment');
  try {
    const payload = await postJson('/api/segment_reset');
    setMetadata(payload.metadata);
    setStatus('segment reset');
  } catch (error) {
    els.segmentInfo.textContent = error.message;
    setStatus('segment reset failed');
  }
}

function onViewportClick(event) {
  if (!state.segmentMode) return;
  handleSegmentClick(event, event.shiftKey);
}

function onViewportContextMenu(event) {
  if (!state.segmentMode) return;
  handleSegmentClick(event, true);
}

function animate() {
  controls.update();
  updateLabelPositions();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

window.addEventListener('resize', resize);
renderer.domElement.addEventListener('click', onViewportClick);
renderer.domElement.addEventListener('contextmenu', onViewportContextMenu);

els.colorMode.addEventListener('change', updateColors);
els.pointSize.addEventListener('input', setPointSize);
els.showLabels.addEventListener('change', renderLabels);
els.filter.addEventListener('input', updateFilters);
els.minPoints.addEventListener('input', updateFilters);
els.maxLabels.addEventListener('input', renderLabels);
els.bulkDeleteClass.addEventListener('click', () => setBulkClassActive(false));
els.bulkRestoreClass.addEventListener('click', () => setBulkClassActive(true));
els.applyLabel.addEventListener('click', applySelectedChanges);
els.saveEdits.addEventListener('click', saveEdits);
els.segmentModeButton.addEventListener('click', () => setSegmentMode(!state.segmentMode));
els.acceptSegment.addEventListener('click', acceptSegment);
els.resetSegment.addEventListener('click', resetSegment);
els.segmentRadius.addEventListener('change', requestSegmentPreview);
els.segmentVoxel.addEventListener('change', requestSegmentPreview);
els.segmentSemantic.addEventListener('change', requestSegmentPreview);
els.topView.addEventListener('click', () => fitCamera(true));
els.resetView.addEventListener('click', () => fitCamera(false));

resize();
animate();
loadData().catch((error) => {
  console.error(error);
  setStatus(`error: ${error.message}`);
});
