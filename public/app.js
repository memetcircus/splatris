/**
 * Splatris Spatial AI Platform — Sprint 5: End-to-End Pipeline & WebXR Dynamic Loader
 */

// --- Global State & Configuration ---
const STATE = {
  activeJobId: null,
  pollInterval: null,
  audioEnabled: true,
  autoOrbit: false,
  splatMesh: null,
  audioEmitterMesh: null,
  triangleMeshGroup: null,
  triangleMeshChunks: [],
  trainingMethod: '3dgs', // '3dgs' (default) | 'triangle'
  activeModelUrl: '/uploads/models/scene.splat',
  isRaycastQuerying: false,
  currentSceneContext: {
    videoId: 'default-scene',
    metadata: null,
    nemotronDecisions: null,
    totalFrames: 12000,
  },
};

// --- Web Audio Spatial API Setup ---
let audioCtx = null;
let pannerNode = null;
let audioListener = null;
let threeAudioListener = null;

function initSpatialAudio() {
  if (threeAudioListener && audioCtx) return;
  try {
    if (!threeAudioListener) {
      threeAudioListener = new THREE.AudioListener();
      camera.add(threeAudioListener);
    }
    audioCtx = threeAudioListener.context;
    audioListener = threeAudioListener;

    if (!pannerNode && audioCtx) {
      pannerNode = audioCtx.createPanner();
      pannerNode.panningModel = 'HRTF';
      pannerNode.distanceModel = 'inverse';
      pannerNode.refDistance = 1;
      pannerNode.maxDistance = 10000;
      pannerNode.rolloffFactor = 1;
      pannerNode.coneInnerAngle = 360;
      pannerNode.connect(audioCtx.destination);
    }
  } catch (err) {
    console.warn('Web Audio Spatial API initialization note:', err.message);
  }
}

function updateSpatialEmitterPosition(x, y, z) {
  if (pannerNode && audioCtx) {
    if (pannerNode.positionX) {
      pannerNode.positionX.setValueAtTime(x, audioCtx.currentTime);
      pannerNode.positionY.setValueAtTime(y, audioCtx.currentTime);
      pannerNode.positionZ.setValueAtTime(z, audioCtx.currentTime);
    } else {
      pannerNode.setPosition(x, y, z);
    }
  }

  // Move 3D visual emitter
  if (STATE.audioEmitterMesh) {
    STATE.audioEmitterMesh.position.set(x, y, z);
    STATE.audioEmitterMesh.visible = true;
  }

  const coordLabel = document.getElementById('emitter-coords');
  if (coordLabel) {
    coordLabel.textContent = `Position: (${x.toFixed(1)}, ${y.toFixed(1)}, ${z.toFixed(1)})`;
  }
}

function triggerSpatialChime(freq = 440, duration = 0.5) {
  if (!audioCtx || !pannerNode) return;
  if (audioCtx.state === 'suspended') audioCtx.resume();

  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();

  osc.type = 'sine';
  osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(freq * 1.5, audioCtx.currentTime + duration);

  gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);

  osc.connect(gain);
  gain.connect(pannerNode);

  osc.start();
  osc.stop(audioCtx.currentTime + duration);
}

// --- Three.js & WebXR Scene Setup ---
const canvas = document.getElementById('webgl-canvas');
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x07090e, 0.04);

const camera = new THREE.PerspectiveCamera(65, window.innerWidth / window.innerHeight, 0.01, 1000.0);
camera.near = 0.01;
camera.far = 1000.0;
camera.updateProjectionMatrix();
camera.position.set(0, 1.25, 0.4);

// Dedicated CameraRig group to allow smooth thumbstick locomotion & turning in WebXR
const cameraRig = new THREE.Group();
cameraRig.name = 'CameraRig';
cameraRig.position.set(0, 0, 0);
scene.add(cameraRig);
cameraRig.add(camera);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;

// --- WebXR Mobile VR (Meta Quest 3 Jitter Fix) Layer Optimizations ---
if (typeof window !== 'undefined' && 'XRWebGLLayer' in window) {
  const NativeXRWebGLLayer = window.XRWebGLLayer;
  window.XRWebGLLayer = function(session, gl, options = {}) {
    // 1. Lower framebuffer scale factor to 0.7 (stereo fill-rate bottleneck fix)
    // 3. Disable MSAA / anti-aliasing during VR presentation
    const xrOptimizedOptions = Object.assign({}, options, {
      antialias: false,
      framebufferScaleFactor: 0.7
    });
    console.log('[WebXR Optimization] Initializing XRWebGLLayer (framebufferScaleFactor: 0.7, antialias: false)');
    const layer = new NativeXRWebGLLayer(session, gl, xrOptimizedOptions);
    // 2. Enable Fixed Foveated Rendering (FFR) at 1.0 (maximum) on Meta Quest 3
    if ('fixedFoveation' in layer) {
      try {
        layer.fixedFoveation = 1.0;
        console.log('[WebXR Optimization] XRWebGLLayer.fixedFoveation set to 1.0 (Maximum)');
      } catch (err) {
        console.warn('[WebXR] Could not assign fixedFoveation to layer:', err);
      }
    }
    return layer;
  };
  window.XRWebGLLayer.prototype = NativeXRWebGLLayer.prototype;
  if (NativeXRWebGLLayer.getNativeFramebufferScaleFactor) {
    window.XRWebGLLayer.getNativeFramebufferScaleFactor = NativeXRWebGLLayer.getNativeFramebufferScaleFactor.bind(NativeXRWebGLLayer);
  }
}

let controller0 = null;
let controller1 = null;

if ('xr' in navigator) {
  renderer.xr.enabled = true;
  if (renderer.xr.setFramebufferScaleFactor) {
    renderer.xr.setFramebufferScaleFactor(0.7);
  }
  if (renderer.xr.setFoveation) {
    renderer.xr.setFoveation(1.0);
  }
  try {
    renderer.xr.setReferenceSpaceType('local-floor');
  } catch (e) {
    console.warn('Could not set local-floor reference space:', e);
  }

  // VR Controllers setup attached to cameraRig with pointing beams & triggers
  controller0 = renderer.xr.getController(0);
  controller1 = renderer.xr.getController(1);
  cameraRig.add(controller0);
  cameraRig.add(controller1);

  const rayGeo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -2.5),
  ]);
  const rayMat = new THREE.LineBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.8 });
  controller0.add(new THREE.Line(rayGeo, rayMat));
  controller1.add(new THREE.Line(rayGeo, rayMat));

  const onXRSelect = (e) => {
    // Explicitly resume audio context inside VR session to satisfy browser autoplay policy
    initSpatialAudio();
    if (threeAudioListener && threeAudioListener.context && threeAudioListener.context.state === 'suspended') {
      threeAudioListener.context.resume().catch((err) => console.warn('[WebXR] Audio resume error:', err));
    } else if (audioCtx && audioCtx.state === 'suspended') {
      audioCtx.resume().catch((err) => console.warn('[WebXR] Audio resume error:', err));
    }
    if ('speechSynthesis' in window && window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }

    // Prevent multi-trigger API spam & pipeline lockup in WebXR mode
    if (STATE.isRaycastQuerying) {
      console.log('[Splatris] WebXR raycast trigger ignored: inspection query already in progress');
      return;
    }

    const controller = e.target;
    const tempMatrix = new THREE.Matrix4();
    tempMatrix.identity().extractRotation(controller.matrixWorld);
    const origin = new THREE.Vector3().setFromMatrixPosition(controller.matrixWorld);
    const direction = new THREE.Vector3(0, 0, -1).applyMatrix4(tempMatrix).normalize();

    // Check in-VR HUD Exit Button
    if (typeof checkXRExitButtonHit === 'function' && checkXRExitButtonHit(origin, direction)) {
      return;
    }

    if (typeof performRaycastSelectionXR === 'function') {
      performRaycastSelectionXR(origin, direction);
    }
  };
  controller0.addEventListener('select', onXRSelect);
  controller1.addEventListener('select', onXRSelect);
}

// Camera Controls: OrbitControls (turntable) and TrackballControls (full 3-axis free rotation)
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.minDistance = 0.05;
controls.maxDistance = 25.0; // Avoid hitting ceiling boundary on zoom
controls.maxPolarAngle = Math.PI - 0.05;
controls.target.set(0, 1.2, 0);

let trackballControls = null;
if (typeof THREE.TrackballControls === 'function') {
  trackballControls = new THREE.TrackballControls(camera, renderer.domElement);
  trackballControls.rotateSpeed = 2.8; // Responsive 3-axis rotation across pitch, yaw, and roll
  trackballControls.zoomSpeed = 1.2;
  trackballControls.panSpeed = 0.8;
  trackballControls.noZoom = false;
  trackballControls.noPan = false;
  trackballControls.staticMoving = false;
  trackballControls.dynamicDampingFactor = 0.18;
  trackballControls.target.set(0, 1.2, 0);
}

// Default mode is Grounded Orbit ('orbit') for stable, level-horizon navigation
STATE.cameraControlMode = 'orbit';

function setCameraControlMode(mode) {
  STATE.cameraControlMode = mode;
  const btn = document.getElementById('btn-toggle-trackball');
  if (mode === 'trackball' && trackballControls) {
    controls.enabled = false;
    trackballControls.enabled = true;
    trackballControls.target.copy(controls.target);
    if (btn) {
      btn.classList.add('active');
      const text = btn.querySelector('span');
      if (text) text.textContent = '3-Axis Free';
    }
  } else {
    if (trackballControls) {
      trackballControls.enabled = false;
      controls.target.copy(trackballControls.target);
    }
    camera.up.set(0, 1, 0); // Restore level horizon
    controls.enabled = true;
    if (btn) {
      btn.classList.remove('active');
      const text = btn.querySelector('span');
      if (text) text.textContent = '3-Axis Free';
    }
  }
}
setCameraControlMode(STATE.cameraControlMode);

function updateCameraTargets(x, y, z) {
  if (typeof x === 'object' && x !== null) {
    controls.target.copy(x);
    if (trackballControls) trackballControls.target.copy(x);
  } else {
    controls.target.set(x, y, z);
    if (trackballControls) trackballControls.target.set(x, y, z);
  }
  controls.update();
  if (trackballControls && trackballControls.enabled) trackballControls.update();
}

function setControlsEnabled(enabled) {
  if (STATE.cameraControlMode === 'trackball' && trackballControls) {
    trackballControls.enabled = enabled;
  } else {
    controls.enabled = enabled;
  }
}

// Spatial 3D Target Reticle Marker (Double-Click / Raycast Selection)
let raycastTargetMesh = null;
function createRaycastTargetMesh() {
  const group = new THREE.Group();
  group.name = 'SpatialRaycastTargetMarker';

  // Compact, non-intrusive target dot & concentric rings (clamped in render loop)
  const dotGeo = new THREE.SphereGeometry(0.012, 16, 16);
  const dotMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, transparent: true, opacity: 0.95 });
  const dot = new THREE.Mesh(dotGeo, dotMat);
  group.add(dot);

  const ringGeo = new THREE.RingGeometry(0.022, 0.032, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x67e8f9,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.85
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  group.add(ring);

  const ring2Geo = new THREE.RingGeometry(0.040, 0.048, 32);
  const ring2Mat = new THREE.MeshBasicMaterial({
    color: 0x8b5cf6,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.6
  });
  const ring2 = new THREE.Mesh(ring2Geo, ring2Mat);
  group.add(ring2);

  group.visible = false;
  scene.add(group);
  raycastTargetMesh = group;
  return group;
}
createRaycastTargetMesh();

// Dedicated Splat Turntable Pivot Container (Rotates around vertical Y axis without wobble)
const splatPivotGroup = new THREE.Group();
splatPivotGroup.name = 'SplatPivotGroup';
scene.add(splatPivotGroup);
STATE.splatPivotGroup = splatPivotGroup;
STATE.turntableActive = false;
STATE.isObjectScan = false;
STATE.pivotTargetY = 1.2;

// Lighting
const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0x8b5cf6, 1.2);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);

const cyanPointLight = new THREE.PointLight(0x06b6d4, 1.5, 20);
cyanPointLight.position.set(-4, 2, -3);
scene.add(cyanPointLight);

// Spatial Grid
const gridHelper = new THREE.GridHelper(20, 40, 0x8b5cf6, 0x1f293d);
gridHelper.position.y = 0;
scene.add(gridHelper);

// 3D Audio Emitter Visual Marker
function createAudioEmitterMarker() {
  const group = new THREE.Group();

  const coreGeo = new THREE.SphereGeometry(0.06, 16, 16);
  const coreMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
  const core = new THREE.Mesh(coreGeo, coreMat);
  group.add(core);

  const ringGeo = new THREE.RingGeometry(0.09, 0.13, 32);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x67e8f9,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.7,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  group.add(ring);

  group.position.set(0, 2.2, -1.8);
  scene.add(group);
  STATE.audioEmitterMesh = group;
}
createAudioEmitterMarker();

// --- 3D Gaussian Splatting Dedicated Engine (@mkkellogg/gaussian-splats-3d) ---
// --- 3D Gaussian Splatting Dedicated Engine (@mkkellogg/gaussian-splats-3d) ---
let activeLoadSessionId = 0;

// --- Stereoscopic Per-Eye Matrix & Viewport Sync for WebXR (Double Vision Fix) ---
function setupStereoscopicPerEyeRendering(splatMesh) {
  if (!splatMesh || splatMesh._stereoHooked) return;
  splatMesh._stereoHooked = true;
  splatMesh.onBeforeRender = function(renderer, scene, eyeCam, geometry, material) {
    if (renderer && renderer.xr && renderer.xr.isPresenting && eyeCam) {
      // 1. Accurate per-eye Viewport Configuration from WebXR
      const vp = eyeCam.viewport;
      let eyeW = 0, eyeH = 0;
      if (vp && vp.z > 0 && vp.w > 0) {
        eyeW = vp.z;
        eyeH = vp.w;
      } else {
        const xrSession = renderer.xr.getSession();
        const baseLayer = xrSession ? xrSession.renderState.baseLayer : null;
        if (baseLayer) {
          eyeW = baseLayer.framebufferWidth / 2;
          eyeH = baseLayer.framebufferHeight;
        } else {
          const s = renderer.getSize(new THREE.Vector2());
          eyeW = s.x / 2;
          eyeH = s.y;
        }
      }

      // 2. Per-Eye Matrix, Focal Length, and Viewport Uniforms Application
      const activeMat = material || this.material;
      if (eyeW > 0 && eyeH > 0 && activeMat && activeMat.uniforms) {
        const proj00 = eyeCam.projectionMatrix.elements[0];
        const proj11 = eyeCam.projectionMatrix.elements[5];
        const fx = proj00 * 0.5 * eyeW;
        const fy = proj11 * 0.5 * eyeH;

        if (activeMat.uniforms.viewport) {
          activeMat.uniforms.viewport.value.set(eyeW, eyeH);
        }
        if (activeMat.uniforms.basisViewport) {
          activeMat.uniforms.basisViewport.value.set(1.0 / eyeW, 1.0 / eyeH);
        }
        if (activeMat.uniforms.focal) {
          activeMat.uniforms.focal.value.set(fx, fy);
        }
        if (activeMat.uniforms.inverseFocalAdjustment) {
          activeMat.uniforms.inverseFocalAdjustment.value = 1.0;
        }
        activeMat.uniformsNeedUpdate = true;
      }
    }
  };
}

// --- WebXR Chunked Triangle Mesh Engine ---
function disposeTriangleMeshChunks() {
  if (STATE.triangleMeshGroup) {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.remove(STATE.triangleMeshGroup);
    } else {
      scene.remove(STATE.triangleMeshGroup);
    }
  }
  if (Array.isArray(STATE.triangleMeshChunks)) {
    for (const chunk of STATE.triangleMeshChunks) {
      if (chunk.geometry) chunk.geometry.dispose();
      if (chunk.material) {
        if (Array.isArray(chunk.material)) {
          chunk.material.forEach((m) => m.dispose());
        } else {
          chunk.material.dispose();
        }
      }
    }
  }
  STATE.triangleMeshChunks = [];
  STATE.triangleMeshGroup = null;
}

function parseAndChunkMonolithicPLY(buffer, gridDim = 4) {
  const decoder = new TextDecoder('ascii');
  const bytes = new Uint8Array(buffer);
  let headerStr = '';
  let headerEnd = -1;
  for (let i = 0; i < Math.min(bytes.length, 4096); i++) {
    if (bytes[i] === 10) { // newline
      const line = decoder.decode(bytes.subarray(0, i));
      if (line.includes('end_header')) {
        headerEnd = i + 1;
        headerStr = line;
        break;
      }
    }
  }

  let numVerts = 0;
  let numFaces = 0;
  for (const l of headerStr.split('\n')) {
    if (l.startsWith('element vertex')) numVerts = parseInt(l.split(' ').pop(), 10);
    if (l.startsWith('element face')) numFaces = parseInt(l.split(' ').pop(), 10);
  }

  const vDataView = new DataView(buffer, headerEnd);
  const nTri = numFaces || Math.floor(numVerts / 3);

  // Pass 1: Compute scene bounds from triangle centroids across 100% of triangles (Zero Decimation)
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

  const centroidsX = new Float32Array(nTri);
  const centroidsY = new Float32Array(nTri);
  const centroidsZ = new Float32Array(nTri);

  for (let i = 0; i < nTri; i++) {
    const o0 = i * 48; // 3 verts * 16 bytes = 48 bytes per triangle
    const x0 = vDataView.getFloat32(o0, true);
    const y0 = vDataView.getFloat32(o0 + 4, true);
    const z0 = vDataView.getFloat32(o0 + 8, true);

    const x1 = vDataView.getFloat32(o0 + 16, true);
    const y1 = vDataView.getFloat32(o0 + 20, true);
    const z1 = vDataView.getFloat32(o0 + 24, true);

    const x2 = vDataView.getFloat32(o0 + 32, true);
    const y2 = vDataView.getFloat32(o0 + 36, true);
    const z2 = vDataView.getFloat32(o0 + 40, true);

    const cx = (x0 + x1 + x2) / 3.0;
    const cy = (y0 + y1 + y2) / 3.0;
    const cz = (z0 + z1 + z2) / 3.0;

    centroidsX[i] = cx;
    centroidsY[i] = cy;
    centroidsZ[i] = cz;

    if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
    if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
    if (cz < minZ) minZ = cz; if (cz > maxZ) maxZ = cz;
  }

  const spanX = Math.max(0.001, maxX - minX);
  const spanY = Math.max(0.001, maxY - minY);
  const spanZ = Math.max(0.001, maxZ - minZ);

  const nx = gridDim;
  const ny = Math.max(2, Math.floor(gridDim / 2));
  const nz = gridDim;
  const numCells = nx * ny * nz;

  const cellCounts = new Int32Array(numCells);
  const triKeys = new Int32Array(nTri);

  for (let i = 0; i < nTri; i++) {
    const gx = Math.min(nx - 1, Math.max(0, Math.floor(((centroidsX[i] - minX) / spanX) * nx)));
    const gy = Math.min(ny - 1, Math.max(0, Math.floor(((centroidsY[i] - minY) / spanY) * ny)));
    const gz = Math.min(nz - 1, Math.max(0, Math.floor(((centroidsZ[i] - minZ) / spanZ) * nz)));
    const key = gx + gy * nx + gz * (nx * ny);
    triKeys[i] = key;
    cellCounts[key]++;
  }

  // Pre-allocate typed arrays for each active spatial chunk
  const chunkBuffers = new Map();
  for (let c = 0; c < numCells; c++) {
    const count = cellCounts[c];
    if (count > 0) {
      chunkBuffers.set(c, {
        pos: new Float32Array(count * 9),
        col: new Uint8Array(count * 12),
        ptr: 0,
      });
    }
  }

  // Pass 2: Transfer all 100% full-resolution vertex positions and colors into spatial chunks
  for (let i = 0; i < nTri; i++) {
    const key = triKeys[i];
    const chunk = chunkBuffers.get(key);
    const pIdx = chunk.ptr * 9;
    const cIdx = chunk.ptr * 12;

    const o = i * 48;
    // Vertex 0
    chunk.pos[pIdx] = vDataView.getFloat32(o, true);
    chunk.pos[pIdx + 1] = vDataView.getFloat32(o + 4, true);
    chunk.pos[pIdx + 2] = vDataView.getFloat32(o + 8, true);
    chunk.col[cIdx] = vDataView.getUint8(o + 12);
    chunk.col[cIdx + 1] = vDataView.getUint8(o + 13);
    chunk.col[cIdx + 2] = vDataView.getUint8(o + 14);
    chunk.col[cIdx + 3] = vDataView.getUint8(o + 15);

    // Vertex 1
    chunk.pos[pIdx + 3] = vDataView.getFloat32(o + 16, true);
    chunk.pos[pIdx + 4] = vDataView.getFloat32(o + 20, true);
    chunk.pos[pIdx + 5] = vDataView.getFloat32(o + 24, true);
    chunk.col[cIdx + 4] = vDataView.getUint8(o + 28);
    chunk.col[cIdx + 5] = vDataView.getUint8(o + 29);
    chunk.col[cIdx + 6] = vDataView.getUint8(o + 30);
    chunk.col[cIdx + 7] = vDataView.getUint8(o + 31);

    // Vertex 2
    chunk.pos[pIdx + 6] = vDataView.getFloat32(o + 32, true);
    chunk.pos[pIdx + 7] = vDataView.getFloat32(o + 36, true);
    chunk.pos[pIdx + 8] = vDataView.getFloat32(o + 40, true);
    chunk.col[cIdx + 8] = vDataView.getUint8(o + 44);
    chunk.col[cIdx + 9] = vDataView.getUint8(o + 45);
    chunk.col[cIdx + 10] = vDataView.getUint8(o + 46);
    chunk.col[cIdx + 11] = vDataView.getUint8(o + 47);

    chunk.ptr++;
  }

  const meshes = [];
  const sharedMat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    side: THREE.DoubleSide,
    transparent: true,
    alphaTest: 0.15,
    depthWrite: true,
  });

  for (const [chunkId, chunk] of chunkBuffers.entries()) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(chunk.pos, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(chunk.col, 4, true));
    geo.computeBoundingSphere();
    geo.computeBoundingBox();

    const mesh = new THREE.Mesh(geo, sharedMat);
    mesh.name = `TriangleChunk_${chunkId}`;
    mesh.frustumCulled = true; // WebXR Camera Frustum Culling
    meshes.push(mesh);
  }

  return { meshes, triCount: nTri };
}

async function loadChunkedTriangleMesh(url) {
  const sessionId = ++activeLoadSessionId;
  const countPill = document.getElementById('stat-splat-count');
  const unitPill = document.getElementById('stat-splat-unit');
  const modelNamePill = document.getElementById('stat-model-name');
  const overlay = document.getElementById('splat-loader-overlay');
  const overlayTitle = document.getElementById('splat-loader-title');
  const overlaySubtitle = document.getElementById('splat-loader-subtitle');
  const overlayBar = document.getElementById('splat-loader-bar');

  if (countPill) countPill.textContent = 'Loading Mesh...';
  if (unitPill) unitPill.textContent = '';
  if (modelNamePill) modelNamePill.textContent = url.split('/').pop();

  if (overlay) {
    overlay.classList.remove('hidden');
    if (overlayTitle) overlayTitle.textContent = 'Loading Chunked Triangle Mesh...';
    if (overlaySubtitle) overlaySubtitle.textContent = 'Streaming WebXR spatial chunks...';
    if (overlayBar) overlayBar.style.width = '15%';
  }

  // Clean up previous 3DGS DropInViewer and existing mesh chunks
  if (STATE.dropInViewer) {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.remove(STATE.dropInViewer);
    } else {
      scene.remove(STATE.dropInViewer);
    }
    try {
      await STATE.dropInViewer.dispose();
    } catch (e) {}
    STATE.dropInViewer = null;
    STATE.splatMesh = null;
  }
  disposeTriangleMeshChunks();

  STATE.activeModelUrl = url;

  // Determine pre-baked .chunks.bin or monolithic .ply
  const baseName = url.replace(/\.(splat|ply|chunks\.bin)$/i, '');
  const candidateChunksUrl = STATE.activeScene?.chunksUrl || `${baseName}.chunks.bin`;
  let buffer = null;
  let isPrebakedChunks = false;

  try {
    const chunkCheck = await fetch(candidateChunksUrl, { method: 'HEAD' });
    if (chunkCheck.ok) {
      if (overlaySubtitle) overlaySubtitle.textContent = 'Streaming pre-partitioned WebXR chunks container...';
      const res = await fetch(candidateChunksUrl);
      if (res.ok) {
        buffer = await res.arrayBuffer();
        isPrebakedChunks = true;
      }
    }
  } catch (e) {}

  if (!buffer) {
    const plyUrl = `${baseName}.ply`;
    if (overlaySubtitle) overlaySubtitle.textContent = 'Downloading high-resolution mesh PLY...';
    const res = await fetch(plyUrl);
    if (!res.ok) {
      throw new Error(`Failed to load triangle mesh: ${res.status}`);
    }
    buffer = await res.arrayBuffer();
  }

  if (sessionId !== activeLoadSessionId) return;

  if (overlayBar) overlayBar.style.width = '65%';
  if (overlaySubtitle) overlaySubtitle.textContent = 'Building spatial octree chunks for WebXR frustum culling...';

  if (!STATE.splatPivotGroup) {
    STATE.splatPivotGroup = new THREE.Group();
    STATE.splatPivotGroup.name = 'SplatPivotGroup';
    scene.add(STATE.splatPivotGroup);
  }
  STATE.splatPivotGroup.rotation.set(0, 0, 0);
  STATE.splatPivotGroup.position.set(0, 0, 0);

  const meshGroup = new THREE.Group();
  meshGroup.name = 'TriangleMeshChunksGroup';
  // Triangle Splatting uses raw COLMAP coordinates (ground plane in X-Z, -Y is UP) -> 180 deg X flip to align upright
  meshGroup.rotation.set(Math.PI, 0, 0);

  let totalTriangles = 0;
  const chunkMeshes = [];

  if (isPrebakedChunks) {
    const dataView = new DataView(buffer);
    const magic = String.fromCharCode(dataView.getUint8(0), dataView.getUint8(1), dataView.getUint8(2), dataView.getUint8(3));
    if (magic !== 'TCHK') {
      throw new Error('Invalid chunk container magic: expected TCHK');
    }
    const version = dataView.getUint32(4, true);
    const numChunks = dataView.getUint32(8, true);
    let offset = 12;

    const sharedMat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      transparent: true,
      alphaTest: 0.15,
      depthWrite: true,
    });

    for (let c = 0; c < numChunks; c++) {
      const chunkId = dataView.getUint32(offset, true);
      const cx = dataView.getFloat32(offset + 4, true);
      const cy = dataView.getFloat32(offset + 8, true);
      const cz = dataView.getFloat32(offset + 12, true);
      const radius = dataView.getFloat32(offset + 16, true);

      const minX = dataView.getFloat32(offset + 20, true);
      const minY = dataView.getFloat32(offset + 24, true);
      const minZ = dataView.getFloat32(offset + 28, true);
      const maxX = dataView.getFloat32(offset + 32, true);
      const maxY = dataView.getFloat32(offset + 36, true);
      const maxZ = dataView.getFloat32(offset + 40, true);

      const nTri = dataView.getUint32(offset + 44, true);
      offset += 48;

      const posByteLen = nTri * 9 * 4;
      const colByteLen = nTri * 12;

      const posArray = new Float32Array(buffer, offset, nTri * 9);
      offset += posByteLen;

      const colArray = new Uint8Array(buffer, offset, nTri * 12);
      offset += colByteLen;

      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(posArray, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(colArray, 4, true));

      // Tight bounding sphere & box enable instant per-eye WebXR camera frustum culling!
      geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(cx, cy, cz), radius);
      geo.boundingBox = new THREE.Box3(new THREE.Vector3(minX, minY, minZ), new THREE.Vector3(maxX, maxY, maxZ));

      const chunkMesh = new THREE.Mesh(geo, sharedMat);
      chunkMesh.name = `TriangleChunk_${chunkId}`;
      chunkMesh.frustumCulled = true; // CRITICAL: Quest 3 camera frustum discards chunks outside the eye FOV!
      meshGroup.add(chunkMesh);
      chunkMeshes.push(chunkMesh);
      totalTriangles += nTri;
    }
  } else {
    const { meshes, triCount } = parseAndChunkMonolithicPLY(buffer, 4);
    totalTriangles = triCount;
    for (const m of meshes) {
      meshGroup.add(m);
      chunkMeshes.push(m);
    }
  }

  STATE.triangleMeshGroup = meshGroup;
  STATE.triangleMeshChunks = chunkMeshes;
  STATE.splatPivotGroup.add(meshGroup);
  STATE.splatMesh = STATE.splatPivotGroup;

  // Auto-align and ground the mesh on the floor at Y=0
  const box = new THREE.Box3().setFromObject(meshGroup);
  if (!box.isEmpty()) {
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const rawHeight = Math.max(0.01, size.y);
    const autoScale = rawHeight > 0.05 ? 1.0 / rawHeight : 1.0;
    meshGroup.scale.set(autoScale, autoScale, autoScale);
    meshGroup.position.set(-center.x * autoScale, -box.min.y * autoScale, -center.z * autoScale);
    STATE.splatPivotGroup.position.set(0, 1.0, 0); // table/pedestal height
  }

  if (overlayBar) overlayBar.style.width = '100%';
  if (overlay) overlay.classList.add('hidden');

  if (countPill) countPill.textContent = totalTriangles.toLocaleString();
  if (unitPill) unitPill.textContent = `Triangles (${chunkMeshes.length} Chunks)`;

  showToast(`✅ Loaded ${totalTriangles.toLocaleString()} triangles across ${chunkMeshes.length} spatial chunks (WebXR Frustum Culled)`, 'success', 3500);
}

async function loadGaussianSplatScene(url, isDynamicReplacement = false) {
  const sessionId = ++activeLoadSessionId;

  // Always clean up previous triangle mesh chunks
  disposeTriangleMeshChunks();

  // If a .chunks.bin or .ply was directly requested, check if high-fidelity .splat exists
  if (url && (url.toLowerCase().endsWith('.chunks.bin') || url.toLowerCase().endsWith('.ply'))) {
    const companionSplat = url.replace(/\.(chunks\.bin|ply)$/i, '.splat');
    try {
      const splatCheck = await fetch(companionSplat, { method: 'HEAD' });
      if (splatCheck.ok) {
        console.log(`[Splatris] Found companion high-fidelity radiance splat: ${companionSplat}. Rendering via DropInViewer.`);
        url = companionSplat;
      } else {
        console.log(`[Splatris] Loading pure triangle mesh container: ${url}`);
        return await loadChunkedTriangleMesh(url);
      }
    } catch {
      return await loadChunkedTriangleMesh(url);
    }
  }

  // Determine if active scene is Triangle Splatting
  let isTriangle = STATE.trainingMethod === 'triangle' || STATE.activeScene?.trainingMethod === 'triangle';
  if (!isTriangle && url) {
    const modelId = url.split('/').pop().replace(/\.(splat|ply|chunks\.bin)$/, '');
    const found = STATE.recentModels?.find((m) => m.name === modelId || m.jobId === modelId || (m.filename && url.includes(m.filename)));
    if (found?.trainingMethod === 'triangle') {
      isTriangle = true;
    }
  }
  if (!isTriangle && url && (url.includes('scene.splat') || url.includes('scene.ply')) && STATE.recentModels && STATE.recentModels.length > 0) {
    if (STATE.recentModels[0].trainingMethod === 'triangle') {
      isTriangle = true;
    }
  }

  const isPly = url.toLowerCase().endsWith('.ply');
  const countPill = document.getElementById('stat-splat-count');
  const unitPill = document.getElementById('stat-splat-unit');
  const modelNamePill = document.getElementById('stat-model-name');
  const overlay = document.getElementById('splat-loader-overlay');
  const overlayTitle = document.getElementById('splat-loader-title');
  const overlaySubtitle = document.getElementById('splat-loader-subtitle');
  const overlayBar = document.getElementById('splat-loader-bar');

  if (countPill) countPill.textContent = 'Connecting...';
  if (unitPill) unitPill.textContent = '';
  if (modelNamePill) modelNamePill.textContent = url.split('/').pop();

  if (overlay) {
    overlay.classList.remove('hidden');
    if (overlayTitle) overlayTitle.textContent = isPly ? 'Loading 3D Triangle Splats...' : 'Loading 3D Gaussian Splats...';
    if (overlaySubtitle) overlaySubtitle.textContent = `Connecting to ${url.split('/').pop()}...`;
    if (overlayBar) overlayBar.style.width = '5%';
  }

  // 1. Verify @mkkellogg/gaussian-splats-3d availability
  const GS = window.GaussianSplats3D || window['Gaussian Splats 3D'];
  const isGSAvailable = Boolean(GS && (GS.DropInViewer || GS.Viewer));
  console.log('[Splatris] GaussianSplats3D Library Available:', isGSAvailable);

  if (!isGSAvailable) {
    const errorMsg = 'FATAL: @mkkellogg/gaussian-splats-3d failed to load on window. Cannot render scene.';
    console.error('[Splatris]', errorMsg);
    if (countPill) countPill.textContent = 'Engine Error';
    if (overlay) overlay.classList.add('hidden');
    alert(errorMsg);
    throw new Error(errorMsg);
  }

  // 2. Clean up previous DropInViewer / Scene objects cleanly
  if (STATE.dropInViewer) {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.remove(STATE.dropInViewer);
    } else {
      scene.remove(STATE.dropInViewer);
    }
    try {
      await STATE.dropInViewer.dispose();
      console.log('[Splatris] Disposed previous GaussianSplats3D instance');
    } catch (e) {
      console.warn('[Splatris] Previous instance disposal notice:', e);
    }
    STATE.dropInViewer = null;
    STATE.splatMesh = null;
  }

  // 3. Initialize DropInViewer with WebXR / Meta Quest 3 optimized settings
  console.log('[Splatris] Initializing GaussianSplats3D.DropInViewer instance...');
  const DropInClass = GS.DropInViewer;
  const dropInViewer = new DropInClass({
    gpuAcceleratedSort: false, // Maximum stability across all WebXR / Quest 3 drivers
    sharedMemoryForWorkers: false,
    integerBasedSort: false,
    dynamicScene: false, // CRITICAL: Static scene enables spatial octree culling & angular thresholding, eliminating sort stalls when looking around!
    splatRenderMode: GS.SplatRenderMode ? GS.SplatRenderMode.ThreeD : 0
  });

  // Ensure dedicated turntable pivot container exists
  if (!STATE.splatPivotGroup) {
    STATE.splatPivotGroup = new THREE.Group();
    STATE.splatPivotGroup.name = 'SplatPivotGroup';
    scene.add(STATE.splatPivotGroup);
  }

  // Reset turntable rotation and positions
  STATE.splatPivotGroup.rotation.set(0, 0, 0);
  STATE.splatPivotGroup.position.set(0, 0, 0);
  dropInViewer.position.set(0, 0, 0);
  dropInViewer.scale.set(1, 1, 1);
  STATE.splatPivotGroup.add(dropInViewer);
  STATE.dropInViewer = dropInViewer;
  STATE.splatMesh = STATE.splatPivotGroup;

  // 4. Add splat scene in a single request with live progress tracking
  const format = isPly 
    ? (GS.SceneFormat ? GS.SceneFormat.Ply : 2) 
    : (GS.SceneFormat ? GS.SceneFormat.Splat : 0);

  // Determine if active scene is Triangle Splatting
  isTriangle = isTriangle || STATE.trainingMethod === 'triangle' || STATE.activeScene?.trainingMethod === 'triangle';
  if (!isTriangle && url) {
    const modelId = url.split('/').pop().replace(/\.(splat|ply)$/, '');
    const found = STATE.recentModels?.find((m) => m.name === modelId || m.jobId === modelId || (m.filename && url.includes(m.filename)));
    if (found?.trainingMethod === 'triangle') {
      isTriangle = true;
    }
  }
  if (!isTriangle && url && url.includes('scene.splat') && STATE.recentModels && STATE.recentModels.length > 0) {
    if (STATE.recentModels[0].trainingMethod === 'triangle') {
      isTriangle = true;
    }
  }

  // Nerfstudio (3DGS): uses Z-up coordinates -> needs -90 deg X rotation (quaternion [-0.7071068, 0, 0, 0.7071068])
  // Triangle Splatting (raw COLMAP): ground plane is in X-Z and -Y is UP -> needs 180 deg X flip [1, 0, 0, 0] to align upright in Three.js
  const sceneRotation = isTriangle
    ? [1, 0, 0, 0]
    : [-0.7071068, 0, 0, 0.7071068];

  console.log(`[Splatris] Loading splat scene via DropInViewer.addSplatScene('${url}', format=${format}, method=${isTriangle ? 'triangle' : '3dgs'})...`);

  try {
    await dropInViewer.addSplatScene(url, {
      splatAlphaRemovalThreshold: 20, // Cull near-zero opacity fog/noise (alpha < 20/255) to reduce overdraw and memory pressure on Quest 3 mobile XR2 GPU
      showLoadingUI: false,
      position: [0, 0, 0],
      rotation: sceneRotation,
      scale: [1, 1, 1],
      format: format,
      progressiveLoad: false,
      onProgress: (percentComplete, percentCompleteLabel) => {
        if (sessionId !== activeLoadSessionId) return;
        const pct = Math.min(100, Math.max(0, Math.round(percentComplete || 0)));
        if (overlay) {
          if (overlayBar) overlayBar.style.width = `${Math.max(6, pct)}%`;
          if (overlaySubtitle) {
            if (pct < 100) {
              overlaySubtitle.textContent = `Downloading ${pct}% • High-fidelity radiance field`;
            } else {
              overlaySubtitle.textContent = 'Compiling Anisotropic Splat Covariance...';
            }
          }
        }
        if (countPill) {
          if (pct < 100) {
            countPill.textContent = `Downloading ${pct}%`;
          } else {
            countPill.textContent = 'Compiling...';
          }
        }
      }
    });

    if (sessionId !== activeLoadSessionId) {
      console.log(`[Splatris] Load session #${sessionId} superseded by newer session #${activeLoadSessionId}. Discarding.`);
      return;
    }

    console.log(`[Splatris] GaussianSplats3D scene successfully compiled with anisotropic covariance shaders and active for rendering!`);

    // 5. Compute spatial bounding box, ground floor at Y=0, and center pivot from in-memory splats
    const splatMesh = dropInViewer.splatMesh;
    const splatCount = (splatMesh && splatMesh.getSplatCount) ? splatMesh.getSplatCount() : (isPly ? 45000 : 307837);
    STATE.currentSplatCount = splatCount;

    let autoScale = 1.0;
    let translateX = 0;
    let translateY = 0;
    let translateZ = 0;
    let roomHeight = 2.4;

    if (splatMesh && splatCount > 0) {
      const sampleStep = Math.max(1, Math.floor(splatCount / 3000));
      const tempCenter = new THREE.Vector3();
      const xs = [], ys = [], zs = [];

      for (let i = 0; i < splatCount; i += sampleStep) {
        splatMesh.getSplatCenter(i, tempCenter, true);
        if (Number.isFinite(tempCenter.x) && Number.isFinite(tempCenter.y) && Number.isFinite(tempCenter.z)) {
          xs.push(tempCenter.x);
          ys.push(tempCenter.y);
          zs.push(tempCenter.z);
        }
      }

      if (xs.length > 20) {
        xs.sort((a, b) => a - b);
        ys.sort((a, b) => a - b);
        zs.sort((a, b) => a - b);

        const p = (arr, q) => arr[Math.floor(arr.length * q)];
        const floorY = p(ys, 0.05);
        const ceilingY = p(ys, 0.95);
        const rawHeight = Math.max(0.01, ceilingY - floorY);

        const centerX = (p(xs, 0.05) + p(xs, 0.95)) / 2;
        const centerY = (p(ys, 0.05) + p(ys, 0.95)) / 2;
        const centerZ = (p(zs, 0.05) + p(zs, 0.95)) / 2;

        let isObjectScan = 
          STATE.trainingMethod === 'image2splat' || 
          STATE.trainingMethod === 'triangle' || 
          url.toLowerCase().includes('image2splat') || 
          url.toLowerCase().includes('tank') ||
          (STATE.activeScene?.trainingMethod === 'triangle') ||
          (STATE.activeScene?.trainingMethod === 'image2splat') ||
          (STATE.activeScene?.originalFilename && STATE.activeScene.originalFilename.toLowerCase().includes('tank'));

        if (!isObjectScan && STATE.recentModels) {
          const match = STATE.recentModels.find(m => url.includes(m.filename) || (m.name && url.includes(m.name)));
          if (match) {
            if (match.trainingMethod === 'triangle' || match.trainingMethod === 'image2splat') {
              isObjectScan = true;
            } else if (match.originalFilename && match.originalFilename.toLowerCase().includes('tank')) {
              isObjectScan = true;
            } else if (match.deviceName && match.deviceName.includes('Instant Object')) {
              isObjectScan = true;
            }
          }
        }
        STATE.isObjectScan = isObjectScan;

        if (isObjectScan) {
          autoScale = rawHeight > 0.05 ? 1.0 / rawHeight : 1.0;
        } else {
          autoScale = (rawHeight > 0.05 && rawHeight < 1.8) ? 2.4 / rawHeight : 1.0;
        }
        roomHeight = rawHeight * autoScale;

        translateX = -centerX * autoScale;
        translateY = -floorY * autoScale;
        translateZ = -centerZ * autoScale;

        console.log(`[Splatris] Spatial Alignment (isObjectScan=${isObjectScan}): Floor=${floorY.toFixed(2)}m, Ceiling=${ceilingY.toFixed(2)}m, Height=${roomHeight.toFixed(2)}m, Scale=${autoScale.toFixed(2)}x`);
      }
    }

    const targetCenterY = STATE.isObjectScan ? 1.0 : Math.max(1.1, Math.min(1.4, roomHeight * 0.52));
    STATE.pivotTargetY = targetCenterY;

    if (STATE.isObjectScan) {
      STATE.splatPivotGroup.position.set(0, targetCenterY, 0);
      dropInViewer.position.set(translateX, -targetCenterY + translateY, translateZ);
    } else {
      STATE.splatPivotGroup.position.set(0, 0, 0);
      dropInViewer.position.set(translateX, translateY, translateZ);
    }

    dropInViewer.scale.set(autoScale, autoScale, autoScale);

    // Ensure all splat meshes and internal geometry are rendered double-sided and never backface-culled
    const ensureDoubleSidedAndNoCulling = (obj) => {
      if (!obj) return;
      obj.frustumCulled = false;
      if (obj.material) {
        obj.material.side = THREE.DoubleSide;
        obj.material.depthTest = true;
        obj.material.depthWrite = false;
        obj.material.needsUpdate = true;
      }
      if (obj.children) {
        obj.children.forEach(ensureDoubleSidedAndNoCulling);
      }
    };
    ensureDoubleSidedAndNoCulling(STATE.splatPivotGroup);
    ensureDoubleSidedAndNoCulling(dropInViewer);
    if (dropInViewer.splatMesh) {
      ensureDoubleSidedAndNoCulling(dropInViewer.splatMesh);
      setupStereoscopicPerEyeRendering(dropInViewer.splatMesh);
    }
    if (dropInViewer.callbackMesh) {
      ensureDoubleSidedAndNoCulling(dropInViewer.callbackMesh);
    }

    STATE.splatPivotGroup.frustumCulled = false;
    dropInViewer.frustumCulled = false;
    dropInViewer.updateMatrixWorld(true);
    STATE.splatPivotGroup.updateMatrixWorld(true);

    camera.near = 0.01;
    camera.far = 1000.0;
    camera.updateProjectionMatrix();

    if (countPill) countPill.textContent = splatCount.toLocaleString();
    if (unitPill) unitPill.textContent = isPly ? 'Triangles' : 'Gaussians';
    if (modelNamePill) modelNamePill.textContent = url.split('/').pop();

    if (overlay) {
      if (overlayBar) overlayBar.style.width = '100%';
      if (overlaySubtitle) overlaySubtitle.textContent = `${splatCount.toLocaleString()} ${isPly ? 'Triangles' : 'Gaussians'} Ready`;
      setTimeout(() => {
        if (sessionId === activeLoadSessionId && overlay) {
          overlay.classList.add('hidden');
        }
      }, 500);
    }

    // 6. Spawn Camera Centered directly on the Object's Vertical Axis with upright horizon
    camera.up.set(0, 1, 0);
    if (STATE.isObjectScan) {
      const viewDist = Math.max(1.8, Math.min(2.8, (roomHeight || 1.0) * 1.8));
      camera.position.set(0, targetCenterY + 0.35, viewDist);
      updateCameraTargets(0, targetCenterY, 0);
      controls.minDistance = 0.1;
      controls.maxDistance = 25.0;
      controls.maxPolarAngle = Math.PI - 0.05;
      showToast('3D Object View: Drag to rotate around model, or use Turntable Spin.', 'info', 4500);
    } else {
      const eyeHeight = targetCenterY;
      STATE.eyeHeight = eyeHeight;
      camera.position.set(0, eyeHeight + 0.2, 1.8);
      updateCameraTargets(0, eyeHeight, 0);
      controls.minDistance = 0.05;
      controls.maxDistance = 25.0;
      controls.maxPolarAngle = Math.PI - 0.05;
    }

    // Sync active state on recent scenes chips
    document.querySelectorAll('.scene-chip').forEach((chip) => {
      const chipUrl = chip.dataset.url;
      const isMatch = chipUrl === url || (url.includes(chipUrl) && !chipUrl.includes('scene.splat'));
      chip.classList.toggle('active', isMatch);
    });

    // Sync active scene metadata for smart re-processing & image isolation
    const modelBase = url.split('/').pop().replace(/\.(splat|ply)$/, '');
    if (STATE.recentModels) {
      const found = STATE.recentModels.find(m => url.endsWith(m.filename) || m.jobId === modelBase || m.name === modelBase);
      if (found) {
        STATE.activeScene = found;
        updatePipelineUIForActiveScene();
      } else {
        safeFetchJson(`/api/models/meta/${modelBase}`).then(res => {
          if (res?.success && res.meta) {
            STATE.activeScene = res.meta;
            updatePipelineUIForActiveScene();
          }
        }).catch(() => {});
      }
    } else {
      safeFetchJson(`/api/models/meta/${modelBase}`).then(res => {
        if (res?.success && res.meta) {
          STATE.activeScene = res.meta;
          updatePipelineUIForActiveScene();
        }
      }).catch(() => {});
    }

    if (isDynamicReplacement) {
      const btnWebXR = document.getElementById('btn-webxr');
      if (btnWebXR) btnWebXR.classList.add('highlight');
    }
  } catch (loadErr) {
    if (sessionId !== activeLoadSessionId) return;
    console.error('[Splatris] Error loading Gaussian Splat scene:', loadErr);
    if (countPill) countPill.textContent = 'Render Error';
    if (unitPill) unitPill.textContent = '';
    if (overlay) {
      if (overlaySubtitle) overlaySubtitle.textContent = `Error: ${loadErr.message || 'Stream interrupted'}`;
      setTimeout(() => overlay.classList.add('hidden'), 2500);
    }
    showToast(`Failed to load splat model: ${loadErr.message || 'Stream error'}`, 'error', 6000);
  }
}

// --- Animation Loop ---
let lastTime = performance.now();
let lastFreeRoamTime = performance.now();
let frameCount = 0;
const fpsLabel = document.getElementById('stat-fps');

function animate() {
  renderer.setAnimationLoop(() => {
    const now = performance.now();
    frameCount++;
    if (now - lastTime >= 1000) {
      if (fpsLabel) fpsLabel.textContent = frameCount;
      frameCount = 0;
      lastTime = now;
    }

    const dt = Math.min((now - lastFreeRoamTime) / 1000, 0.1);
    lastFreeRoamTime = now;

    if (renderer.xr.isPresenting) {
      window.__splatrisXrActive = true;

      // Ensure Fixed Foveated Rendering (FFR) remains locked at 1.0 (Maximum) on Meta Quest 3
      if (renderer.xr.setFoveation) {
        renderer.xr.setFoveation(1.0);
      }
      const xrSession = renderer.xr.getSession();
      if (xrSession && xrSession.renderState && xrSession.renderState.baseLayer) {
        if ('fixedFoveation' in xrSession.renderState.baseLayer) {
          xrSession.renderState.baseLayer.fixedFoveation = 1.0;
        }
      }

      // 1. Disable Frustum Culling on main 3DGS container & Triangle Splatting mesh
      if (STATE.splatPivotGroup) STATE.splatPivotGroup.frustumCulled = false;
      if (STATE.dropInViewer) {
        STATE.dropInViewer.frustumCulled = false;
        if (STATE.dropInViewer.splatMesh) STATE.dropInViewer.splatMesh.frustumCulled = false;
      }

      // Ensure Head-locked HUD remains attached to camera at top-left FOV
      if (xrHudMesh && xrHudMesh.visible && xrHudMesh.parent !== camera) {
        camera.add(xrHudMesh);
        xrHudMesh.position.set(-0.5, 0.3, -1.5);
        xrHudMesh.rotation.set(-0.06, 0.22, 0);
      }

      // 2. Increase Camera Far Plane to 1000.0 so environment never clips out during room-scale walk
      camera.near = 0.01;
      camera.far = 1000.0;
      camera.updateProjectionMatrix();

      const xrCam = renderer.xr.getCamera();
      if (xrCam) {
        xrCam.near = 0.01;
        xrCam.far = 1000.0;
        if (xrCam.cameras && xrCam.cameras.length > 0) {
          xrCam.cameras.forEach((c) => {
            c.near = 0.01;
            c.far = 1000.0;
            // Note: Never call c.updateProjectionMatrix() on XR eye cameras as that destroys asymmetric stereoscopic matrices
          });
        }
      }

      // Reduce screen-space splat size by 20% to drastically cut overdraw on mobile XR2 GPU
      if (STATE.dropInViewer && STATE.dropInViewer.splatMesh) {
        if (STATE.dropInViewer.splatMesh.splatScale !== 0.8) {
          if (STATE.dropInViewer.splatMesh.setSplatScale) {
            STATE.dropInViewer.splatMesh.setSplatScale(0.8);
          } else if (STATE.dropInViewer.splatMesh.material && STATE.dropInViewer.splatMesh.material.uniforms && STATE.dropInViewer.splatMesh.material.uniforms.splatScale) {
            STATE.dropInViewer.splatMesh.material.uniforms.splatScale.value = 0.8;
          }
        }
      }

      // --- WebXR 6DoF Thumbstick Locomotion & Turning ---
      const session = renderer.xr.getSession();
      if (session && session.inputSources) {
        const xrCam = renderer.xr.getCamera();
        const xrForward = new THREE.Vector3();
        xrCam.getWorldDirection(xrForward);
        xrForward.y = 0;
        xrForward.normalize();

        const xrRight = new THREE.Vector3();
        xrRight.crossVectors(xrForward, new THREE.Vector3(0, 1, 0)).normalize();

        for (const source of session.inputSources) {
          if (!source.gamepad) continue;
          const gp = source.gamepad;
          const axes = gp.axes || [];
          const btns = gp.buttons || [];

          // Exit WebXR on Secondary Button (Button B on right controller or Button Y on left controller)
          if (btns[5] && btns[5].pressed && !STATE.isExitingXR) {
            STATE.isExitingXR = true;
            session.end().finally(() => { STATE.isExitingXR = false; });
            return;
          }

          const hand = source.handedness;
          const deadzone = 0.15;

          // Left Controller: Smooth Locomotion (move forward/backward & strafe)
          if (hand === 'left') {
            const stickX = axes.length >= 4 ? axes[2] : (axes[0] || 0);
            const stickY = axes.length >= 4 ? axes[3] : (axes[1] || 0);

            const moveSpeed = 2.4 * dt;
            if (Math.abs(stickY) > deadzone) {
              // stickY < 0 is forward
              cameraRig.position.addScaledVector(xrForward, -stickY * moveSpeed);
            }
            if (Math.abs(stickX) > deadzone) {
              cameraRig.position.addScaledVector(xrRight, stickX * moveSpeed);
            }
          }
          // Right Controller: Turning & Elevation
          else if (hand === 'right') {
            const stickX = axes.length >= 4 ? axes[2] : (axes[0] || 0);
            const stickY = axes.length >= 4 ? axes[3] : (axes[1] || 0);

            // Smooth yaw turning
            if (Math.abs(stickX) > deadzone) {
              const turnSpeed = 2.0 * dt;
              cameraRig.rotation.y -= stickX * turnSpeed;
            }

            // Elevation (fly up/down)
            if (Math.abs(stickY) > deadzone) {
              const elevSpeed = 1.8 * dt;
              cameraRig.position.y -= stickY * elevSpeed;
              if (cameraRig.position.y < -0.8) cameraRig.position.y = -0.8;
            }
          }
        }
      }
    } else {
      window.__splatrisXrActive = false;
      if (camera.far !== 1000.0 || camera.near !== 0.01) {
        camera.near = 0.01;
        camera.far = 1000.0;
        camera.updateProjectionMatrix();
      }

      // Ensure desktop splat scale remains exactly 1.0
      if (STATE.dropInViewer && STATE.dropInViewer.splatMesh) {
        if (STATE.dropInViewer.splatMesh.splatScale !== 1.0) {
          if (STATE.dropInViewer.splatMesh.setSplatScale) {
            STATE.dropInViewer.splatMesh.setSplatScale(1.0);
          } else if (STATE.dropInViewer.splatMesh.material && STATE.dropInViewer.splatMesh.material.uniforms && STATE.dropInViewer.splatMesh.material.uniforms.splatScale) {
            STATE.dropInViewer.splatMesh.material.uniforms.splatScale.value = 1.0;
          }
        }
      }

      if (STATE.isFreeRoam) {
        // --- 6DoF 3D Fly & Free Roam Navigation (Flight along Mouse Gaze Vector) ---
        const speed = 2.6; // Flight speed in meters per second
        const moveVector = new THREE.Vector3();
        const forward = new THREE.Vector3();
        camera.getWorldDirection(forward);
        forward.normalize(); // Direct 3D gaze vector where mouse is pointing (including pitch up/down)

        // Right vector is perpendicular to forward and world Up for clean horizontal strafing
        const right = new THREE.Vector3();
        right.crossVectors(forward, new THREE.Vector3(0, 1, 0)).normalize();
        if (right.lengthSq() < 0.01) {
          right.crossVectors(forward, camera.up).normalize();
        }

        if (moveForward) moveVector.add(forward);
        if (moveBackward) moveVector.sub(forward);
        if (moveRight) moveVector.add(right);
        if (moveLeft) moveVector.sub(right);

        if (moveVector.lengthSq() > 0) {
          moveVector.normalize();
          camera.position.addScaledVector(moveVector, speed * dt);
        }
        if (moveUp) camera.position.y += speed * dt;
        if (moveDown) camera.position.y -= speed * dt;
      } else {
        if (STATE.cameraControlMode === 'trackball' && trackballControls && trackballControls.enabled) {
          trackballControls.update();
        } else if (controls.enabled) {
          controls.update();
        }
      }
    }

    // Smooth turntable spin around the vertical Y axis
    if (STATE.splatPivotGroup && (STATE.autoOrbit || STATE.turntableActive) && !renderer.xr.isPresenting) {
      STATE.splatPivotGroup.rotation.y += 0.008; // ~0.5 RPM smooth vertical spin
    }

    // Pulsing 3D target inspection reticle animation (strictly clamped screen-space size)
    if (raycastTargetMesh && raycastTargetMesh.visible) {
      const r1 = raycastTargetMesh.children[1];
      const r2 = raycastTargetMesh.children[2];
      if (r1) r1.rotation.z += 0.025;
      if (r2) r2.rotation.z -= 0.018;

      const activeCam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;
      const camPos = new THREE.Vector3();
      activeCam.getWorldPosition(camPos);

      // Keep reticle billboarded directly toward the active camera
      raycastTargetMesh.lookAt(camPos);

      // Clamp reticle scale so it remains a small, non-intrusive targeting dot/circle
      // regardless of the camera's distance, zoom level, or FOV
      const dist = Math.max(0.12, camPos.distanceTo(raycastTargetMesh.position));
      const fovFactor = (activeCam.isPerspectiveCamera && activeCam.fov)
        ? Math.tan(THREE.MathUtils.degToRad(activeCam.fov * 0.5))
        : 0.45;

      const baseScale = dist * fovFactor * 0.35;
      const clampedScale = Math.min(0.75, Math.max(0.18, baseScale));
      const pulse = 1.0 + Math.sin(now * 0.007) * 0.12;
      const finalScale = clampedScale * pulse;
      raycastTargetMesh.scale.set(finalScale, finalScale, finalScale);
    }

    if (STATE.audioEmitterMesh) {
      const ring = STATE.audioEmitterMesh.children[1];
      if (ring) {
        const scale = 1 + Math.sin(now * 0.005) * 0.25;
        ring.scale.set(scale, scale, scale);
      }
    }

    const activeCamera = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;

    if (audioListener && audioCtx && audioCtx.state === 'running') {
      const headPos = new THREE.Vector3();
      activeCamera.getWorldPosition(headPos);

      if (audioListener.positionX) {
        audioListener.positionX.setValueAtTime(headPos.x, audioCtx.currentTime);
        audioListener.positionY.setValueAtTime(headPos.y, audioCtx.currentTime);
        audioListener.positionZ.setValueAtTime(headPos.z, audioCtx.currentTime);
      } else if (audioListener.setPosition) {
        audioListener.setPosition(headPos.x, headPos.y, headPos.z);
      }

      // 3D Spatial Audio Orientation (HRTF) tracking headset / view rotation
      const forwardDir = new THREE.Vector3();
      activeCamera.getWorldDirection(forwardDir);
      const camQuat = new THREE.Quaternion();
      activeCamera.getWorldQuaternion(camQuat);
      const upDir = new THREE.Vector3(0, 1, 0).applyQuaternion(camQuat);

      if (audioListener.forwardX) {
        audioListener.forwardX.setValueAtTime(forwardDir.x, audioCtx.currentTime);
        audioListener.forwardY.setValueAtTime(forwardDir.y, audioCtx.currentTime);
        audioListener.forwardZ.setValueAtTime(forwardDir.z, audioCtx.currentTime);
        audioListener.upX.setValueAtTime(upDir.x, audioCtx.currentTime);
        audioListener.upY.setValueAtTime(upDir.y, audioCtx.currentTime);
        audioListener.upZ.setValueAtTime(upDir.z, audioCtx.currentTime);
      } else if (audioListener.setOrientation) {
        audioListener.setOrientation(forwardDir.x, forwardDir.y, forwardDir.z, upDir.x, upDir.y, upDir.z);
      }
    }

    if (STATE.dropInViewer && STATE.dropInViewer.viewer) {
      if (STATE.dropInViewer.splatMesh) {
        setupStereoscopicPerEyeRendering(STATE.dropInViewer.splatMesh);
      }
      try {
        const isXR = Boolean(renderer && renderer.xr && renderer.xr.isPresenting);
        if (isXR) {
          const xrCam = renderer.xr.getCamera();
          // Decouple sorting from pure rotation in WebXR:
          // Depth order of splats does not change significantly when rotating from a fixed position.
          // ONLY trigger a resort when the user physically translates beyond 0.20m (20cm) in space.
          if (!STATE.lastXRSortPos) {
            STATE.lastXRSortPos = new THREE.Vector3();
            xrCam.getWorldPosition(STATE.lastXRSortPos);
            STATE.dropInViewer.viewer.update(renderer, xrCam);
          } else {
            const currentXRPos = new THREE.Vector3();
            xrCam.getWorldPosition(currentXRPos);
            const moveDist = currentXRPos.distanceTo(STATE.lastXRSortPos);
            if (moveDist >= 0.20) {
              STATE.lastXRSortPos.copy(currentXRPos);
              STATE.dropInViewer.viewer.update(renderer, xrCam);
            }
          }
        } else {
          STATE.lastXRSortPos = null;
          STATE.dropInViewer.viewer.update(renderer, camera);
        }
      } catch (e) {}
    }

    renderer.render(scene, camera);
  });
}

function onWindowResize() {
  if (!camera || !renderer) return;
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.near = 0.01;
  camera.far = 1000.0;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  if (trackballControls) {
    trackballControls.handleResize();
  }
}
window.addEventListener('resize', onWindowResize);

// --- Toast Notification Helper ---
function showToast(message, type = 'info', durationMs = 4500) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast-notification ${type}`;

  let iconHtml = 'ℹ️';
  if (type === 'error') iconHtml = '⚠️';
  else if (type === 'warning') iconHtml = '⚡';
  else if (type === 'success') iconHtml = '✅';

  toast.innerHTML = `
    <span class="toast-icon">${iconHtml}</span>
    <span class="toast-text">${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('hide');
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 300);
  }, durationMs);
}

// --- Collapsible Left & Right Panels (Immersive Viewport Mode) ---
const btnToggleLeft = document.getElementById('btn-toggle-left-panel');
const btnToggleRight = document.getElementById('btn-toggle-right-panel');
const leftPanel = document.getElementById('left-panel');
const rightPanel = document.getElementById('right-panel');

if (leftPanel) {
  leftPanel.addEventListener('transitionend', () => onWindowResize());
}
if (rightPanel) {
  rightPanel.addEventListener('transitionend', () => onWindowResize());
}

if (btnToggleLeft && leftPanel) {
  btnToggleLeft.addEventListener('click', () => {
    const isCollapsed = leftPanel.classList.toggle('collapsed');
    btnToggleLeft.classList.toggle('active', !isCollapsed);
    btnToggleLeft.setAttribute('title', isCollapsed ? 'Expand Space Ingestion Pipeline (Left Panel)' : 'Collapse Space Ingestion Pipeline (Left Panel)');
    onWindowResize();
  });
}

if (btnToggleRight && rightPanel) {
  btnToggleRight.addEventListener('click', () => {
    const isCollapsed = rightPanel.classList.toggle('collapsed');
    btnToggleRight.classList.toggle('active', !isCollapsed);
    btnToggleRight.setAttribute('title', isCollapsed ? 'Expand Spatial Audio RAG (Right Panel)' : 'Collapse Spatial Audio RAG (Right Panel)');
    onWindowResize();
  });
}

// --- Step Tracker UI Helper ---
function setPipelineStep(activeStepIndex, options = {}) {
  for (let i = 1; i <= 5; i++) {
    const el = document.getElementById(`step-${i}`);
    if (!el) continue;
    el.classList.remove('active', 'completed', 'bypassed');

    if (options.bypassColmap && (i === 1 || i === 2)) {
      el.classList.add('bypassed');
    } else if (i < activeStepIndex) {
      el.classList.add('completed');
    } else if (i === activeStepIndex) {
      el.classList.add('active');
    }
  }
}

// --- Sprint 5: End-to-End Automated Pipeline ---
const dropzone = document.getElementById('video-dropzone');
const fileInput = document.getElementById('video-file-input');
const dropTitle = document.getElementById('dropzone-title');
const dropSubtitle = document.getElementById('dropzone-subtitle');
const jobPill = document.getElementById('job-status-pill');

const nemotronCard = document.getElementById('nemotron-reasoning-card');
const nemotronMatcherBadge = document.getElementById('nemotron-matcher-badge');
const nemotronItersVal = document.getElementById('nemotron-iters-val');
const nemotronFramesVal = document.getElementById('nemotron-frames-val');
const nemotronReasoningText = document.getElementById('nemotron-reasoning-text');

const progressCard = document.getElementById('job-progress-card');
const progressBar = document.getElementById('progress-bar');
const progressStage = document.getElementById('progress-stage-text');
const progressPct = document.getElementById('progress-pct');
const progressJobId = document.getElementById('progress-job-id');
const progressTimer = document.getElementById('progress-timer');
const btnAbortJob = document.getElementById('btn-abort-job');
const btnCancelUpload = document.getElementById('btn-cancel-upload');

let currentUploadXhr = null;

// Dropzone interactions
if (dropzone && fileInput) {
  dropzone.addEventListener('click', () => fileInput.click());

  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });

  ['dragleave', 'dragend'].forEach((ev) => {
    dropzone.addEventListener(ev, () => dropzone.classList.remove('drag-over'));
  });

  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      executeEndToEndPipeline(files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      executeEndToEndPipeline(files[0]);
    }
  });
}

// Abort & Reset Pipeline Button Handler
if (btnAbortJob) {
  btnAbortJob.addEventListener('click', async (e) => {
    e.preventDefault();
    e.stopPropagation();
    await abortCurrentPipeline();
  });
}

// Cancel Video Upload Button Handler
if (btnCancelUpload) {
  btnCancelUpload.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (currentUploadXhr) {
      console.log('[Splatris] User cancelled video upload.');
      currentUploadXhr.abort();
      currentUploadXhr = null;
    }
    btnCancelUpload.classList.add('hidden');
    progressCard.classList.add('hidden');
    dropTitle.textContent = 'Drop a walkthrough video or single object photo';
    dropSubtitle.textContent = 'Supports MP4/MOV walkthroughs (max 1GB) or JPG/PNG photos for Instant 3D scan';
    jobPill.className = 'badge idle';
    jobPill.textContent = 'Idle';
    progressBar.style.width = '0%';
    progressPct.textContent = '0%';
    if (fileInput) fileInput.value = '';
    showToast('Upload cancelled by user.', 'info', 4000);
  });
}

// --- Training Method Segmented Control (Standard 3DGS vs Triangle Splatting vs Instant Object) ---
const btnMethod3dgs = document.getElementById('btn-method-3dgs');
const btnMethodTriangle = document.getElementById('btn-method-triangle');
const btnMethodImage2splat = document.getElementById('btn-method-image2splat');
const methodSelectedBadge = document.getElementById('method-selected-badge');
const instantObjectFallbackCard = document.getElementById('instant-object-fallback-card');
const inputManualObjectName = document.getElementById('input-manual-object-name');
const btnConfirmObjectName = document.getElementById('btn-confirm-object-name');

// --- Feature Update: Pipeline Caching & Fast Re-Processing UI Controls ---
const reprocessBanner = document.getElementById('reprocess-banner');
const reprocessSceneName = document.getElementById('reprocess-scene-name');
const reprocessTargetName = document.getElementById('reprocess-target-name');
const reprocessDescText = document.getElementById('reprocess-desc-text');
const btnReprocessAction = document.getElementById('btn-reprocess-action');
const btnReprocessLabel = document.getElementById('btn-reprocess-label');
const btnReprocessUploadNew = document.getElementById('btn-reprocess-upload-new');

// --- Feature Update: Preemptible GPU Suspended & Offline Banner Controls ---
const gpuSuspendedBanner = document.getElementById('gpu-suspended-banner');
const gpuSuspendedDetail = document.getElementById('gpu-suspended-detail');
const btnResumeTraining = document.getElementById('btn-resume-training');
const btnResumeLabel = document.getElementById('btn-resume-label');
const btnAbortSuspended = document.getElementById('btn-abort-suspended');

function showGpuSuspendedBanner(detailMsg) {
  if (gpuSuspendedBanner) {
    gpuSuspendedBanner.classList.remove('hidden');
    gpuSuspendedBanner.style.removeProperty('display');
    gpuSuspendedBanner.style.display = 'block';
  }
  if (gpuSuspendedDetail && detailMsg) {
    gpuSuspendedDetail.textContent = detailMsg;
  }
}

function hideGpuSuspendedBanner() {
  if (gpuSuspendedBanner) {
    gpuSuspendedBanner.classList.add('hidden');
    gpuSuspendedBanner.style.setProperty('display', 'none', 'important');
  }
}

function updateButtonsForActiveJob(isActive) {
  if (btnReprocessAction) {
    if (isActive) {
      btnReprocessAction.disabled = true;
      btnReprocessAction.classList.add('btn-disabled');
      if (btnReprocessLabel) btnReprocessLabel.textContent = 'Training in Progress...';
    } else {
      btnReprocessAction.disabled = false;
      btnReprocessAction.classList.remove('btn-disabled');
      if (btnReprocessLabel) {
        btnReprocessLabel.textContent = STATE.trainingMethod === 'triangle'
          ? 'Re-process as Triangle Splatting (Fast)'
          : 'Re-process as Standard 3DGS (Fast)';
      }
    }
  }
}

function updatePipelineUIForActiveScene() {
  const active = STATE.activeScene;

  // Robust detection of whether active scene is a single photo (Image2Splat / Instant Object)
  const isImageScene = Boolean(
    active?.isImage ||
    active?.trainingMethod === 'image2splat' ||
    (active?.deviceName && active.deviceName.toLowerCase().includes('instant object')) ||
    (active?.deviceName && active.deviceName.toLowerCase().includes('single image')) ||
    (active?.originalFilename && /\.(jpe?g|png|webp)$/i.test(active.originalFilename)) ||
    (STATE.activeModelUrl && STATE.activeModelUrl.toLowerCase().includes('image2splat'))
  );

  const btn3dgs = document.getElementById('btn-method-3dgs');
  const btnTriangle = document.getElementById('btn-method-triangle');
  const btnImage = document.getElementById('btn-method-image2splat');
  const videoDropzone = document.getElementById('video-dropzone');

  if (isImageScene) {
    // 1. VISIBILITY STATE RESET:
    // When the user selects a single photo (Image2Splat) from Recent Scenes,
    // the UI must completely hide (display: none) the #reprocess-banner.
    // This banner must ONLY be visible when the active scene is a Video-based model.
    if (reprocessBanner) {
      reprocessBanner.classList.add('hidden');
      reprocessBanner.style.setProperty('display', 'none', 'important');
    }

    // 2. IMAGE ISOLATION:
    // If the active scene was generated from a single photo (Instant Object / Image2Splat),
    // disable or gray-out the "Standard 3DGS" and "Triangle Splatting" radio buttons entirely.
    if (btn3dgs) {
      btn3dgs.classList.add('method-disabled');
      btn3dgs.setAttribute('disabled', 'true');
      btn3dgs.setAttribute('title', 'Single images cannot be processed into video-based pipelines. Image2Splat is isolated.');
    }
    if (btnTriangle) {
      btnTriangle.classList.add('method-disabled');
      btnTriangle.setAttribute('disabled', 'true');
      btnTriangle.setAttribute('title', 'Single images cannot be processed into video-based pipelines. Image2Splat is isolated.');
    }

    // Force active radio selection to image2splat
    if (STATE.trainingMethod !== 'image2splat') {
      setTrainingMethod('image2splat', false);
    }

    if (videoDropzone) {
      videoDropzone.classList.remove('hidden');
      videoDropzone.style.display = '';
    }
    return;
  }

  // VIDEO SCENE:
  // Enable 3dgs and triangle buttons
  if (btn3dgs) {
    btn3dgs.classList.remove('method-disabled');
    btn3dgs.removeAttribute('disabled');
    btn3dgs.setAttribute('title', 'Standard 3D Gaussian Splatting (Nerfstudio Splatfacto)');
  }
  if (btnTriangle) {
    btnTriangle.classList.remove('method-disabled');
    btnTriangle.removeAttribute('disabled');
    btnTriangle.setAttribute('title', 'Triangle Splatting: Mesh-based differentiable rasterization for WebXR');
  }

  // Determine active model's method vs selected method
  const activeMethod = active?.trainingMethod || (STATE.activeModelUrl?.endsWith('.ply') ? 'triangle' : '3dgs');
  const selectedMethod = STATE.trainingMethod;

  // Video Re-processing Check:
  // Banner must ONLY be visible when active scene is a Video and user switches between 3dgs and triangle
  const isSwitchingPipeline = (selectedMethod !== activeMethod) && (selectedMethod === 'triangle' || selectedMethod === '3dgs');

  if (isSwitchingPipeline && !STATE.userWantsNewUpload) {
    // Dynamic Name Binding: Render actual filename of active scene (e.g. tank_internet.mp4)
    const activeFilename = active?.originalFilename || (active?.name ? `${active.name}.mp4` : 'Walkthrough.mp4');
    const targetLabel = selectedMethod === 'triangle' ? 'Triangle Splatting (WebXR Optimized)' : 'Standard 3DGS';

    if (reprocessSceneName) {
      reprocessSceneName.textContent = activeFilename;
    }
    if (reprocessTargetName) {
      reprocessTargetName.textContent = selectedMethod === 'triangle' ? 'Triangle Splatting' : 'Standard 3DGS';
    }
    if (reprocessDescText) {
      reprocessDescText.innerHTML = `This scene (<strong>${escapeHtml(activeFilename)}</strong>) is already processed. You can instantly convert it to <strong>${escapeHtml(targetLabel)}</strong> without waiting for a new upload or initial processing.`;
    }
    if (btnReprocessLabel) {
      btnReprocessLabel.textContent = selectedMethod === 'triangle'
        ? 'Re-process as Triangle Splatting (Fast)'
        : 'Re-process as Standard 3DGS (Fast)';
    }

    if (reprocessBanner) {
      reprocessBanner.classList.remove('hidden');
      reprocessBanner.style.removeProperty('display');
      reprocessBanner.style.display = 'block';
    }
    if (videoDropzone) {
      videoDropzone.classList.add('hidden');
      videoDropzone.style.display = 'none';
    }
  } else {
    // Normal mode: hide banner, show dropzone
    if (reprocessBanner) {
      reprocessBanner.classList.add('hidden');
      reprocessBanner.style.setProperty('display', 'none', 'important');
    }
    if (videoDropzone) {
      videoDropzone.classList.remove('hidden');
      videoDropzone.style.display = '';
    }
  }
}

async function triggerFastReprocess() {
  const active = STATE.activeScene;
  if (!active) {
    showToast('No active video scene found to re-process.', 'warning', 3500);
    return;
  }

  const isImageScene = Boolean(
    active?.isImage ||
    active?.trainingMethod === 'image2splat' ||
    (active?.deviceName && active.deviceName.includes('Single Image'))
  );
  if (isImageScene) {
    showToast('Single image scenes cannot be re-processed into video pipelines.', 'error', 4500);
    return;
  }

  const sceneId = active.jobId || active.sceneId || active.name || (STATE.activeModelUrl?.split('/').pop().replace(/\.(splat|ply)$/, ''));
  const targetMethod = STATE.trainingMethod; // 'triangle' or '3dgs'

  showToast(`⚡ Quick Convert: Re-processing as ${targetMethod === 'triangle' ? 'Triangle Splatting' : 'Standard 3DGS'} (Using existing processed data)...`, 'info', 4000);

  // Set pipeline step 4 with bypass
  setPipelineStep(4, { bypassColmap: true });
  const step1Text = document.querySelector('#step-1 span:last-child');
  const step2Text = document.querySelector('#step-2 span:last-child');
  const step4Text = document.querySelector('#step-4 span:last-child');
  if (step1Text) step1Text.textContent = '1. Upload (Ready)';
  if (step2Text) step2Text.textContent = '2. Processed';
  if (step4Text) step4Text.textContent = targetMethod === 'triangle' ? '4. Mesh Train' : '4. 3DGS Train';

  const progressCard = document.getElementById('job-progress-card');
  const progressStage = document.getElementById('progress-stage-text');
  const progressPct = document.getElementById('progress-pct');
  const progressBar = document.getElementById('progress-bar');
  const progressJobId = document.getElementById('progress-job-id');

  if (progressCard) progressCard.classList.remove('hidden');
  if (progressStage) progressStage.textContent = '⚡ Quick Convert: Re-processing scene into new pipeline...';
  if (progressPct) progressPct.textContent = '35%';
  if (progressBar) progressBar.style.width = '35%';

  if (reprocessBanner) {
    reprocessBanner.classList.add('hidden');
    reprocessBanner.style.setProperty('display', 'none', 'important');
  }
  if (dropzone) {
    dropzone.classList.add('hidden');
    dropzone.style.display = 'none';
  }

  try {
    const res = await safeFetchJson('/api/spatial/job/reprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sceneId, trainingMethod: targetMethod }),
    });

    if (!res.success || !res.jobId) {
      throw new Error(res.error || 'Failed to start fast re-processing job.');
    }

    if (progressJobId) progressJobId.textContent = `job: ${res.jobId.slice(0, 8)}...`;
    STATE.activeJobId = res.jobId;
    localStorage.setItem('splatris_active_job_id', res.jobId);
    localStorage.setItem('splatris_active_scene_id', sceneId);
    updateButtonsForActiveJob(true);

    if (res.isExisting) {
      showToast('Re-attached to active training job for this scene.', 'info', 3000);
    }

    const uploadData = {
      videoId: sceneId,
      originalName: active.originalFilename || `${sceneId}.mp4`,
      metadata: { device: active.deviceName || 'Camera' },
      totalFrames: active.splatCount || 12000,
    };
    const decisions = { iterations: 10000, colmapMatcher: 'vocab_tree' };

    await pollJobUntilReady(res.jobId, uploadData, decisions);
  } catch (err) {
    console.error('[Reprocess] Error:', err);
    showToast(`Re-processing error: ${err.message}`, 'error', 5000);
    if (progressCard) progressCard.classList.add('hidden');
    updateButtonsForActiveJob(false);
    updatePipelineUIForActiveScene();
  }
}

if (btnReprocessAction) {
  btnReprocessAction.addEventListener('click', (e) => {
    e.preventDefault();
    triggerFastReprocess();
  });
}

if (btnReprocessUploadNew) {
  btnReprocessUploadNew.addEventListener('click', (e) => {
    e.preventDefault();
    STATE.userWantsNewUpload = true;
    if (reprocessBanner) {
      reprocessBanner.classList.add('hidden');
      reprocessBanner.style.setProperty('display', 'none', 'important');
    }
    const videoDropzone = document.getElementById('video-dropzone');
    if (videoDropzone) {
      videoDropzone.classList.remove('hidden');
      videoDropzone.style.display = '';
    }
    updatePipelineUIForActiveScene();
  });
}

// Preemptible GPU Suspended & Offline Banner Listeners
if (btnResumeTraining) {
  btnResumeTraining.addEventListener('click', async () => {
    const jobId = STATE.activeJobId || localStorage.getItem('splatris_active_job_id');
    const sceneId = STATE.activeScene?.jobId || STATE.activeScene?.sceneId || localStorage.getItem('splatris_active_scene_id');

    if (!jobId && !sceneId) {
      showToast('No active or suspended job found to resume.', 'warning', 3500);
      return;
    }

    if (btnResumeLabel) btnResumeLabel.textContent = 'Reconnecting to GPU...';
    btnResumeTraining.disabled = true;

    try {
      const res = await safeFetchJson('/api/spatial/job/resume', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId, sceneId }),
      });

      if (!res.success) {
        throw new Error(res.error || 'Could not resume training job.');
      }

      showToast('⚡ Training resumed on Nebius GPU (using cached poses)...', 'success', 4000);
      hideGpuSuspendedBanner();

      if (progressStage) progressStage.textContent = res.stage || 'Resuming training on Nebius GPU...';
      if (jobPill) {
        jobPill.className = 'badge processing';
        jobPill.textContent = 'Training';
      }

      const uploadData = {
        videoId: sceneId || jobId,
        originalName: STATE.activeScene?.originalFilename || 'Walkthrough.mp4',
        metadata: { device: STATE.activeScene?.deviceName || 'Mobile Camera' },
        totalFrames: STATE.activeScene?.splatCount || 10000,
      };
      pollJobUntilReady(res.jobId || jobId, uploadData, { colmapMatcher: 'vocab_tree' });
    } catch (err) {
      console.error('[ResumeJob] Error:', err);
      showToast(`Cannot resume yet: ${err.message}`, 'error', 5500);
      if (gpuSuspendedDetail) {
        gpuSuspendedDetail.textContent = err.message;
      }
    } finally {
      if (btnResumeLabel) btnResumeLabel.textContent = 'Resume Training';
      btnResumeTraining.disabled = false;
    }
  });
}

if (btnAbortSuspended) {
  btnAbortSuspended.addEventListener('click', async () => {
    const jobId = STATE.activeJobId || localStorage.getItem('splatris_active_job_id');
    if (jobId) {
      await safeFetchJson('/api/spatial/job/abort', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId }),
      }).catch(() => {});
    }
    localStorage.removeItem('splatris_active_job_id');
    localStorage.removeItem('splatris_active_scene_id');
    hideGpuSuspendedBanner();
    resetPipelineUI({ notify: false });
    showToast('Job cancelled and reset.', 'info', 3000);
  });
}

function setTrainingMethod(method, triggerUpdate = true) {
  if (['3dgs', 'triangle', 'image2splat'].includes(method)) {
    STATE.trainingMethod = method;
  } else {
    STATE.trainingMethod = '3dgs';
  }

  const is3dgs = STATE.trainingMethod === '3dgs';
  const isTriangle = STATE.trainingMethod === 'triangle';
  const isImage2splat = STATE.trainingMethod === 'image2splat';

  if (btnMethod3dgs) {
    btnMethod3dgs.classList.toggle('active', is3dgs);
    btnMethod3dgs.setAttribute('aria-checked', is3dgs.toString());
  }
  if (btnMethodTriangle) {
    btnMethodTriangle.classList.toggle('active', isTriangle);
    btnMethodTriangle.setAttribute('aria-checked', isTriangle.toString());
  }
  if (btnMethodImage2splat) {
    btnMethodImage2splat.classList.toggle('active', isImage2splat);
    btnMethodImage2splat.setAttribute('aria-checked', isImage2splat.toString());
  }

  if (methodSelectedBadge) {
    if (isImage2splat) {
      methodSelectedBadge.textContent = 'Instant Object (Image2Splat)';
    } else if (isTriangle) {
      methodSelectedBadge.textContent = 'Triangle Splatting';
    } else {
      methodSelectedBadge.textContent = 'Standard 3DGS';
    }
  }

  // The universal dropzone accepts BOTH video and image files regardless of active mode
  const fileInput = document.getElementById('video-file-input');
  const dTitle = document.getElementById('dropzone-title');
  const dSubtitle = document.getElementById('dropzone-subtitle');

  if (fileInput) {
    fileInput.accept = 'video/mp4,video/quicktime,image/jpeg,image/png';
  }

  if (dTitle && dSubtitle) {
    if (isImage2splat) {
      dTitle.textContent = 'Drop single object photo (JPG or PNG)';
      dSubtitle.textContent = 'Instant 3D Gaussian Splatting via Image2Splat • Or drop a video for room scale';
    } else if (isTriangle) {
      dTitle.textContent = 'Drop space walkthrough video (30–60s, MP4/MOV)';
      dSubtitle.textContent = 'Mesh-based Triangle Splatting • Or drop a photo for Instant Object';
    } else {
      dTitle.textContent = 'Drop a walkthrough video or single object photo';
      dSubtitle.textContent = 'Supports MP4/MOV walkthroughs (max 1GB) or JPG/PNG photos for Instant 3D scan';
    }
  }

  // Update pipeline step labels
  const step2Text = document.querySelector('#step-2 span:last-child');
  const step4Text = document.querySelector('#step-4 span:last-child');
  if (step2Text) step2Text.textContent = isImage2splat ? '2. Depth Lift' : '2. Extract';
  if (step4Text) step4Text.textContent = isImage2splat ? '4. Image2Splat' : (isTriangle ? '4. Mesh Train' : '4. 3DGS Train');

  if (btnAuditProceedLabel) {
    btnAuditProceedLabel.textContent = isImage2splat
      ? 'Proceed to Instant 3DGS'
      : (isTriangle ? 'Proceed to Triangle Training' : 'Proceed to 3DGS Training');
  }

  console.log(`[Splatris] Training pipeline method set to: ${STATE.trainingMethod}`);

  if (triggerUpdate) {
    updatePipelineUIForActiveScene();
  }
}

if (btnMethod3dgs) {
  btnMethod3dgs.addEventListener('click', (e) => {
    e.preventDefault();
    if (btnMethod3dgs.classList.contains('method-disabled')) {
      showToast('Single image scenes cannot be processed into video-based pipelines. Image2Splat is isolated.', 'warning', 4000);
      return;
    }
    STATE.userWantsNewUpload = false;
    setTrainingMethod('3dgs');
  });
}
if (btnMethodTriangle) {
  btnMethodTriangle.addEventListener('click', (e) => {
    e.preventDefault();
    if (btnMethodTriangle.classList.contains('method-disabled')) {
      showToast('Single image scenes cannot be processed into video-based pipelines. Image2Splat is isolated.', 'warning', 4000);
      return;
    }
    STATE.userWantsNewUpload = false;
    setTrainingMethod('triangle');
  });
}
if (btnMethodImage2splat) {
  btnMethodImage2splat.addEventListener('click', (e) => {
    e.preventDefault();
    STATE.userWantsNewUpload = false;
    setTrainingMethod('image2splat');
  });
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// --- Pre-Flight AI Video Quality Audit Modal Controller ---
const qualityAuditModal = document.getElementById('quality-audit-modal');
const btnCloseAuditModal = document.getElementById('btn-close-audit-modal');
const btnAuditReupload = document.getElementById('btn-audit-reupload');
const btnAuditBypass = document.getElementById('btn-audit-bypass');
const btnAuditProceed = document.getElementById('btn-audit-proceed');
const btnAuditProceedLabel = document.getElementById('btn-audit-proceed-label');

const auditScoreNum = document.getElementById('audit-score-num');
const auditGaugeFill = document.getElementById('audit-gauge-fill');
const auditStatusBadge = document.getElementById('audit-status-badge');
const auditModelBadge = document.getElementById('audit-model-badge');
const auditSummaryTitle = document.getElementById('audit-summary-title');
const auditSummaryDesc = document.getElementById('audit-summary-desc');

const pillarBlur = document.getElementById('pillar-blur');
const pillarBlurStatus = document.getElementById('pillar-blur-status');
const pillarParallax = document.getElementById('pillar-parallax');
const pillarParallaxStatus = document.getElementById('pillar-parallax-status');
const pillarLighting = document.getElementById('pillar-lighting');
const pillarLightingStatus = document.getElementById('pillar-lighting-status');
const pillarTexture = document.getElementById('pillar-texture');
const pillarTextureStatus = document.getElementById('pillar-texture-status');

const auditIssuesSection = document.getElementById('audit-issues-section');
const auditIssuesCount = document.getElementById('audit-issues-count');
const auditIssuesList = document.getElementById('audit-issues-list');
const auditGuidanceList = document.getElementById('audit-guidance-list');

function showQualityAuditModal(audit) {
  return new Promise((resolve) => {
    if (!qualityAuditModal) {
      return resolve({ action: 'proceed' });
    }

    const issues = Array.isArray(audit.issues_detected) ? audit.issues_detected : [];
    const isVerticalVideo = Boolean(
      audit.aspect_ratio_rejected ||
      (audit.laplacian_stats && audit.laplacian_stats.isPortrait) ||
      issues.some(i => i.type?.includes('Vertical Video') || i.detail?.includes('Landscape mode'))
    );

    let score;
    if (isVerticalVideo) {
      score = 0;
    } else if (typeof audit.readiness_score === 'number' && !isNaN(audit.readiness_score)) {
      score = Math.max(0, Math.min(100, Math.round(audit.readiness_score)));
    } else {
      const totalSampled = audit.laplacian_stats?.sampleCount || 12;
      const uniqueFlagged = new Set(issues.map(i => i.timestamp?.trim()).filter(Boolean)).size;
      const healthyFrames = Math.max(0, totalSampled - uniqueFlagged);
      score = Math.round((healthyFrames / totalSampled) * 100);
    }
    const isApproved = !isVerticalVideo && (audit.status === 'APPROVED' || audit.status === 'PASSED') && score >= 75;

    // 1. Update Gauge & Score
    if (auditScoreNum) auditScoreNum.textContent = `${score}%`;
    if (auditGaugeFill) {
      const circumference = 314.159;
      const offset = circumference * (1 - score / 100);
      auditGaugeFill.style.strokeDashoffset = offset;
      auditGaugeFill.className = 'gauge-fill' + (score >= 75 ? '' : score >= 50 ? ' warning' : ' danger');
    }

    // 2. Status Badge & Summary
    if (auditStatusBadge) {
      auditStatusBadge.textContent = isApproved ? 'APPROVED' : 'REJECTED';
      auditStatusBadge.className = `badge-status ${isApproved ? 'approved' : 'rejected'}`;
    }
    if (auditModelBadge) {
      const contextLabel = isVerticalVideo ? ' • Vertical Video Blocked' : audit.detected_context ? ` • ${audit.detected_context.replace(/_/g, ' ')}` : '';
      auditModelBadge.textContent = `${audit.evaluated_by || 'openbmb/MiniCPM-V-4_5 (Nebius)'}${contextLabel}`;
    }

    if (auditSummaryTitle) {
      auditSummaryTitle.textContent = isVerticalVideo
        ? 'Invalid Format: Vertical Video Detected'
        : isApproved 
          ? 'Ready for 3D Gaussian Reconstruction'
          : 'Quality Gatekeeper Notice: High Failure Risk';
    }
    if (auditSummaryDesc) {
      auditSummaryDesc.textContent = isVerticalVideo
        ? '3D Gaussian Splatting requires a wide horizontal field of view. Please rotate your camera to Landscape mode (horizontal) and capture the space again.'
        : isApproved
          ? 'The video exhibits stable translation, minimal motion blur, and balanced lighting suitable for COLMAP sparse point cloud initialization.'
          : 'Audit flagged potential issues that may cause COLMAP feature tracking loss or disconnected floating splat artifacts. Review detected issues below.';
    }

    // 3. Four Photogrammetry Pillars
    const issuesText = issues.map(i => `${i.type} ${i.detail}`).join(' ').toLowerCase();

    // Blur Pillar
    const hasBlur = issuesText.includes('blur') || issuesText.includes('pan');
    if (pillarBlur && pillarBlurStatus) {
      pillarBlur.className = `pillar-card ${isVerticalVideo ? 'warn' : hasBlur ? (score < 50 ? 'fail' : 'warn') : 'pass'}`;
      pillarBlurStatus.textContent = hasBlur ? 'Motion Blur / Rapid Pan' : 'Sharp Frames (Laplacian OK)';
    }

    // Parallax Pillar
    const hasParallaxLoss = isVerticalVideo || issuesText.includes('parallax') || issuesText.includes('rotation') || issuesText.includes('fixed') || issuesText.includes('spin');
    if (pillarParallax && pillarParallaxStatus) {
      pillarParallax.className = `pillar-card ${hasParallaxLoss ? (score < 50 ? 'fail' : 'warn') : 'pass'}`;
      pillarParallaxStatus.textContent = isVerticalVideo ? 'Narrow Vertical FOV' : hasParallaxLoss ? 'Parallax Loss / Axis Spin' : 'Translational Orbit';
    }

    // Lighting Pillar
    const hasLightingIssue = issuesText.includes('lighting') || issuesText.includes('exposure') || issuesText.includes('flare') || issuesText.includes('dark');
    if (pillarLighting && pillarLightingStatus) {
      pillarLighting.className = `pillar-card ${isVerticalVideo ? 'pass' : hasLightingIssue ? (score < 50 ? 'fail' : 'warn') : 'pass'}`;
      pillarLightingStatus.textContent = hasLightingIssue ? 'Exposure Shift / Glare' : 'Balanced Exposure';
    }

    // Texture Pillar
    const hasTextureIssue = issuesText.includes('texture') || issuesText.includes('blank') || issuesText.includes('monolithic') || issuesText.includes('keypoint');
    if (pillarTexture && pillarTextureStatus) {
      pillarTexture.className = `pillar-card ${isVerticalVideo ? 'warn' : hasTextureIssue ? (score < 50 ? 'fail' : 'warn') : 'pass'}`;
      pillarTextureStatus.textContent = hasTextureIssue ? 'Low Visual Texture' : 'Rich Keypoints';
    }

    // 4. Detected Issues List (with Visual Frame Thumbnails & Lightbox Inspection)
    if (auditIssuesSection && auditIssuesList && auditIssuesCount) {
      if (isVerticalVideo) {
        // Strict Aspect Ratio Rejection UI Notice: Override standard issues list with critical blocker card
        auditIssuesSection.classList.remove('hidden');
        auditIssuesCount.textContent = 'Critical Blocker';
        auditIssuesList.innerHTML = `
          <div class="critical-blocker-card">
            <div class="critical-blocker-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="7" y="2" width="10" height="20" rx="2" ry="2"></rect>
                <line x1="12" y1="18" x2="12.01" y2="18"></line>
              </svg>
            </div>
            <div class="critical-blocker-content">
              <div class="critical-blocker-title">Invalid Format: Vertical Video Detected</div>
              <div class="critical-blocker-desc">
                3D Gaussian Splatting requires a wide horizontal field of view. Please rotate your camera to Landscape mode (horizontal) and capture the space again.
              </div>
              <div class="critical-blocker-hint">
                <span>📐</span>
                <span><strong>Required Camera Mode:</strong> Landscape (Horizontal) orientation</span>
              </div>
            </div>
          </div>
        `;
      } else if (issues.length > 0) {
        auditIssuesSection.classList.remove('hidden');
        auditIssuesCount.textContent = `${issues.length} ${issues.length === 1 ? 'issue' : 'issues'}`;
        auditIssuesList.innerHTML = issues.map((issue, idx) => `
          <div class="issue-card" data-idx="${idx}">
            ${issue.thumbnail_url ? `
              <div class="issue-thumb-wrapper" data-thumb="${encodeURIComponent(issue.thumbnail_url)}" data-type="${escapeHtml(issue.type || '')}" data-time="${escapeHtml(issue.timestamp || '')}" data-detail="${escapeHtml(issue.detail || '')}" title="Click to inspect flagged frame">
                <img class="issue-thumb" src="${issue.thumbnail_url}" alt="Frame at ${escapeHtml(issue.timestamp || '')}" loading="lazy" />
                <span class="thumb-zoom-icon">🔍</span>
              </div>
            ` : ''}
            <div class="issue-main">
              <div class="issue-header-row">
                <span class="issue-timestamp">${escapeHtml(issue.timestamp || '00:00')}</span>
                <span class="issue-type">${escapeHtml(issue.type || 'Quality Issue')}</span>
              </div>
              <span class="issue-desc">${escapeHtml(issue.detail || '')}</span>
            </div>
          </div>
        `).join('');

        // Attach click listeners to all thumbnail wrappers to open lightbox
        auditIssuesList.querySelectorAll('.issue-thumb-wrapper').forEach((wrap) => {
          wrap.addEventListener('click', () => {
            const thumbUrl = decodeURIComponent(wrap.getAttribute('data-thumb') || '');
            const type = wrap.getAttribute('data-type') || '';
            const timestamp = wrap.getAttribute('data-time') || '';
            const detail = wrap.getAttribute('data-detail') || '';
            openFrameLightbox(thumbUrl, type, timestamp, detail);
          });
        });
      } else {
        auditIssuesSection.classList.add('hidden');
        auditIssuesList.innerHTML = '';
      }
    }

    // 5. Context-Aware Director Guidance
    if (auditGuidanceList) {
      const guidanceItems = Array.isArray(audit.director_guidance) && audit.director_guidance.length > 0
        ? audit.director_guidance
        : [
            'Maintain translational movement around the subject; avoid rotating the recording device on a fixed central axis.',
            'Tilt the recording device ~15° downwards toward surface boundaries and textured planes.',
            'Ensure consistent illumination across all viewing angles without harsh specular glare or auto-exposure jumps.'
          ];

      auditGuidanceList.innerHTML = guidanceItems.map(item => `
        <div class="guidance-card">
          <span class="guidance-icon">💡</span>
          <span class="guidance-text">${escapeHtml(item)}</span>
        </div>
      `).join('');
    }

    // 6. Action Buttons & Manual Consent Logic (No Auto-Start)
    const cleanupAndResolve = (decision) => {
      qualityAuditModal.classList.add('hidden');
      cleanupListeners();
      resolve(decision);
    };

    const onProceed = () => {
      if (btnAuditProceed && btnAuditProceed.disabled) return;
      cleanupAndResolve({ action: 'proceed' });
    };
    const onBypass = () => {
      if (isVerticalVideo) {
        window.alert(
          'Invalid Format: Vertical Video Detected\n\n' +
          '3D Gaussian Splatting requires a wide horizontal field of view. Please rotate your camera to Landscape mode (horizontal) and capture the space again.'
        );
        return;
      }
      const confirmed = window.confirm(
        `Nebius Server Credit Warning:\n\nThis video scored only ${score}% readiness. Running COLMAP on a video with motion blur or axis-spin rotation wastes Nebius server credits and frequently produces sparse, floating splat artifacts.\n\nAre you sure you want to bypass and force training?`
      );
      if (confirmed) {
        cleanupAndResolve({ action: 'bypass' });
      }
    };
    const onReupload = () => cleanupAndResolve({ action: 'reupload' });
    const onClose = () => cleanupAndResolve({ action: 'cancel' });

    function cleanupListeners() {
      if (btnAuditProceed) btnAuditProceed.removeEventListener('click', onProceed);
      if (btnAuditBypass) btnAuditBypass.removeEventListener('click', onBypass);
      if (btnAuditReupload) btnAuditReupload.removeEventListener('click', onReupload);
      if (btnCloseAuditModal) btnCloseAuditModal.removeEventListener('click', onClose);
    }

    if (btnAuditProceed) btnAuditProceed.addEventListener('click', onProceed);
    if (btnAuditBypass) btnAuditBypass.addEventListener('click', onBypass);
    if (btnAuditReupload) btnAuditReupload.addEventListener('click', onReupload);
    if (btnCloseAuditModal) btnCloseAuditModal.addEventListener('click', onClose);

    if (isApproved) {
      // When PASSED (>= 75%): Enable Proceed as primary action, hide Bypass button.
      // Explicit manual click required: Halt on results screen without auto-start or timeout.
      if (btnAuditProceed) {
        btnAuditProceed.disabled = false;
        btnAuditProceed.classList.remove('hidden');
        btnAuditProceed.title = 'Proceed with 3D Gaussian Splatting training';
        if (btnAuditProceedLabel) btnAuditProceedLabel.textContent = 'Proceed to 3DGS Training';
      }
      if (btnAuditBypass) {
        btnAuditBypass.classList.add('hidden');
      }
    } else {
      // When REJECTED (< 75%): Disable Proceed button (opacity reduced, cursor not-allowed)
      if (btnAuditProceed) {
        btnAuditProceed.disabled = true;
        btnAuditProceed.classList.remove('hidden');
        btnAuditProceed.title = isVerticalVideo
          ? 'Proceed disabled: Vertical video cannot be reconstructed into 3D Gaussian Splats.'
          : 'Proceed disabled: Readiness score must be at least 75% to prevent reconstruction failure.';
        if (btnAuditProceedLabel) btnAuditProceedLabel.textContent = 'Proceed to 3DGS Training';
      }
      if (btnAuditBypass) {
        if (isVerticalVideo) {
          btnAuditBypass.classList.add('hidden');
        } else {
          btnAuditBypass.classList.remove('hidden');
          btnAuditBypass.title = 'Force training despite quality risks. Warning: High risk of wasted Nebius server credits (Nebius L40S) and sparse floating splats.';
        }
      }
    }

    // Reveal Modal
    qualityAuditModal.classList.remove('hidden');
  });
}

/**
 * Automates:
 * 1. Upload Video & Pre-Flight Quality Gatekeeper (POST /api/spatial/upload)
 * 2. Dynamic Overlap Frame Extraction (clamp(duration * 2.2, 90, 220))
 * 3. Nemotron Reasoning (POST /api/agent/analyze-frames)
 * 4. Nebius GPU 3DGS Training (POST /api/spatial/job/start)
 * 5. Dynamic 3DGS Scene Replacement in WebXR
 * 6. Spatial Audio RAG Context Synchronization
 */
// Defensive fetch helper to prevent SyntaxError: Unexpected token '<' on HTML error pages
async function safeFetchJson(url, options = {}) {
  const res = await fetch(url, options);
  const contentType = res.headers.get('content-type') || '';
  if (!res.ok) {
    let errMsg = `Request failed (${res.status} ${res.statusText})`;
    if (contentType.includes('application/json')) {
      try {
        const errJson = await res.json();
        errMsg = errJson.error || errJson.message || errMsg;
      } catch (e) {}
    } else {
      const errText = await res.text();
      if (errText) errMsg = `${errMsg}: ${errText.slice(0, 150)}`;
    }
    throw new Error(errMsg);
  }
  if (!contentType.includes('application/json')) {
    const text = await res.text();
    throw new Error(`Expected JSON but received ${contentType || 'non-JSON'}: ${text.slice(0, 100)}`);
  }
  return await res.json();
}

/**
 * Uploads video with real-time percentage and byte progress tracking via XMLHttpRequest.
 */
function uploadVideoWithProgress(url, formData, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    currentUploadXhr = xhr;
    xhr.open('POST', url, true);

    if (xhr.upload && onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          onProgress(e.loaded, e.total);
        }
      };
    }

    xhr.onload = () => {
      currentUploadXhr = null;
      const contentType = xhr.getResponseHeader('content-type') || '';
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          resolve(data);
        } catch {
          reject(new Error('Invalid response from server during upload'));
        }
      } else if (xhr.status === 413) {
        reject(new Error('Payload Too Large (HTTP 413): File exceeds the maximum allowed upload limit (1GB / 1024MB).'));
      } else {
        try {
          const errData = JSON.parse(xhr.responseText);
          reject(new Error(errData.error || errData.message || `Upload failed (HTTP ${xhr.status})`));
        } catch {
          reject(new Error(`Upload failed with HTTP ${xhr.status}: ${xhr.statusText}`));
        }
      }
    };

    xhr.onerror = () => {
      currentUploadXhr = null;
      reject(new Error('Network connection error during upload.'));
    };
    xhr.ontimeout = () => {
      currentUploadXhr = null;
      reject(new Error('Upload timed out.'));
    };
    xhr.onabort = () => {
      currentUploadXhr = null;
      const abortErr = new Error('Upload cancelled by user');
      abortErr.name = 'AbortError';
      reject(abortErr);
    };
    xhr.send(formData);
  });
}

// Pre-flight browser metadata check for video aspect ratio (width vs height)
function checkVideoOrientationInBrowser(file) {
  return new Promise((resolve) => {
    try {
      const video = document.createElement('video');
      video.preload = 'metadata';
      const objUrl = URL.createObjectURL(file);
      video.onloadedmetadata = () => {
        URL.revokeObjectURL(objUrl);
        const w = video.videoWidth || 0;
        const h = video.videoHeight || 0;
        resolve({ isPortrait: w > 0 && h > 0 && h >= w, width: w, height: h });
      };
      video.onerror = () => {
        URL.revokeObjectURL(objUrl);
        resolve({ isPortrait: false, width: 0, height: 0 });
      };
      video.src = objUrl;
    } catch (e) {
      resolve({ isPortrait: false, width: 0, height: 0 });
    }
  });
}

/**
 * Fully resets the ingestion pipeline UI to its initial upload state.
 */
function resetPipelineUI(options = {}) {
  const { notify = false, toastMessage = 'Pipeline reset to initial upload state.' } = options;

  // 1. Clear background polling interval and active job
  if (STATE.pollInterval) {
    clearInterval(STATE.pollInterval);
    STATE.pollInterval = null;
  }
  STATE.activeJobId = null;
  localStorage.removeItem('splatris_active_job_id');
  localStorage.removeItem('splatris_active_scene_id');
  hideGpuSuspendedBanner();
  updateButtonsForActiveJob(false);

  // 2. Reset step tracker to Step 1 (Upload)
  setPipelineStep(1);

  // 3. Reset status pill to idle
  if (jobPill) {
    jobPill.className = 'badge idle';
    jobPill.textContent = 'Idle';
  }

  // 4. Clear and hide progress bar & card
  if (progressBar) progressBar.style.width = '0%';
  if (progressPct) progressPct.textContent = '0%';
  if (progressStage) progressStage.textContent = 'Ready for video input';
  if (progressJobId) progressJobId.textContent = 'job: —';
  if (progressTimer) progressTimer.textContent = '0s';
  if (progressCard) progressCard.classList.add('hidden');

  // 5. Hide and reset Abort button
  if (btnAbortJob) {
    btnAbortJob.classList.add('hidden');
    btnAbortJob.disabled = false;
    btnAbortJob.innerHTML = `
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
      </svg>
      <span>Abort & Reset Job</span>
    `;
  }

  // 6. Hide Nemotron reasoning card
  if (nemotronCard) nemotronCard.classList.add('hidden');

  // 7. Reset dropzone copy and file input
  if (dropTitle) {
    dropTitle.textContent = 'Drop a walkthrough video or single object photo';
  }
  if (dropSubtitle) {
    dropSubtitle.textContent = 'Supports MP4/MOV walkthroughs (max 1GB) or JPG/PNG photos for Instant 3D scan';
  }
  if (fileInput) {
    fileInput.value = '';
  }

  if (notify) {
    showToast(toastMessage, 'info', 4500);
  }
}

/**
 * Aborts the running 3DGS training job on Nebius GPU, releases GPU resources, and resets the pipeline UI.
 */
async function abortCurrentPipeline() {
  const targetJobId = STATE.activeJobId;
  const confirmed = window.confirm(
    "Nebius GPU Compute Kill Switch:\n\nAre you sure you want to abort the running 3DGS training job?\n\nThis will immediately terminate the active COLMAP / Splatfacto process on the Nebius GPU server, release GPU compute credits, and purge intermediate files."
  );
  if (!confirmed) return;

  // Immediate UI feedback on the abort button
  if (btnAbortJob) {
    btnAbortJob.disabled = true;
    btnAbortJob.innerHTML = `
      <svg class="spin-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
        <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor"></path>
      </svg>
      <span>Aborting GPU Process...</span>
    `;
  }

  // Clear polling interval immediately
  if (STATE.pollInterval) {
    clearInterval(STATE.pollInterval);
    STATE.pollInterval = null;
  }

  if (progressStage) {
    progressStage.textContent = 'Terminating Nebius GPU process and releasing resources...';
  }

  if (targetJobId) {
    try {
      await fetch(`/api/spatial/job/${targetJobId}/abort`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ jobId: targetJobId }),
      });
      console.log(`[Splatris] Successfully sent abort request for job ${targetJobId}`);
    } catch (err) {
      console.warn('[Splatris] Abort API call error:', err);
    }
  }

  // Full UI Reset back to initial upload state
  resetPipelineUI({
    notify: true,
    toastMessage: 'Job aborted. Nebius GPU resources released and pipeline reset to initial state.',
  });
}

// --- Autonomous Vision RAG Integration (Nemotron Vision + Tavily) ---
async function runAutonomousVisionRAG(videoId) {
  try {
    console.log('[VisionRAG] Triggering Nemotron Vision object identification for videoId:', videoId);
    if (instantObjectFallbackCard) instantObjectFallbackCard.classList.add('hidden');

    const res = await safeFetchJson('/api/spatial/instant-object/identify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoId }),
    });

    if (res.isConfident && res.objectName) {
      showToast(`AI Identified: "${res.objectName}" (${Math.round((res.confidence || 0.9) * 100)}% match)`, 'success', 5000);
      if (res.tavilyData) {
        streamTavilyDataToRagPanel(res.objectName, res.tavilyData);
      }
    } else {
      // Vision model was not confident or failed
      // Fallback: pause Tavily search and prompt user with text input field
      console.warn('[VisionRAG] Vision model paused. Prompting user for object identification:', res);
      displayFallbackObjectInput(videoId, res.suggestedName);
    }
  } catch (err) {
    console.warn('[VisionRAG] Vision RAG error:', err.message);
    displayFallbackObjectInput(videoId, '');
  }
}

function displayFallbackObjectInput(videoId, suggestedName) {
  if (!instantObjectFallbackCard) return;
  instantObjectFallbackCard.classList.remove('hidden');
  if (inputManualObjectName) {
    inputManualObjectName.value = suggestedName || '';
    inputManualObjectName.focus();
  }

  const handleConfirm = async () => {
    const objectName = inputManualObjectName?.value?.trim();
    if (!objectName) {
      showToast('Please enter an object name or description.', 'warning', 3000);
      return;
    }
    instantObjectFallbackCard.classList.add('hidden');
    showToast(`Searching Tavily for "${objectName}"...`, 'info', 4000);

    try {
      const tavilyRes = await safeFetchJson('/api/spatial/instant-object/tavily-enrich', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ objectName, videoId }),
      });
      if (tavilyRes.success && tavilyRes.tavilyData) {
        showToast(`Tavily knowledge fetched for "${objectName}"!`, 'success', 4000);
        streamTavilyDataToRagPanel(objectName, tavilyRes.tavilyData);
      }
    } catch (err) {
      console.error('[VisionRAG] Fallback Tavily enrichment error:', err);
      showToast(`Could not fetch details for ${objectName}: ${err.message}`, 'error', 4000);
    }
  };

  if (btnConfirmObjectName) {
    btnConfirmObjectName.onclick = handleConfirm;
  }
  if (inputManualObjectName) {
    inputManualObjectName.onkeydown = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        handleConfirm();
      }
    };
  }
}

function streamTavilyDataToRagPanel(objectName, tavilyData) {
  const ragLog = document.getElementById('rag-chat-log');
  if (!ragLog) return;

  const msgDiv = document.createElement('div');
  msgDiv.className = 'message ai-msg';

  let sourcesHtml = '';
  if (tavilyData.results && tavilyData.results.length > 0) {
    sourcesHtml = `
      <div class="rag-sources" style="margin-top: 8px; font-size: 11px; opacity: 0.85;">
        <strong>Verified Grounding Sources:</strong>
        <ul style="margin: 4px 0 0 16px; padding: 0;">
          ${tavilyData.results.slice(0, 3).map(r => `<li><a href="${escapeHtml(r.url)}" target="_blank" style="color: var(--accent-cyan, #06b6d4); text-decoration: underline;">${escapeHtml(r.title || r.url)}</a></li>`).join('')}
        </ul>
      </div>
    `;
  }

  msgDiv.innerHTML = `
    <div class="msg-header">
      <span class="agent-avatar" style="background: linear-gradient(135deg, #f59e0b, #ec4899);">AI</span>
      <span class="agent-name">Autonomous Spatial Guide • Instant Object Scan</span>
      <span class="badge ai" style="background: rgba(245, 158, 11, 0.25); color: #fbbf24;">Tavily + Nemotron</span>
    </div>
    <div class="msg-content">
      <div style="font-weight: 600; font-size: 14px; margin-bottom: 6px; color: #38bdf8;">
        Identified 3D Object: ${escapeHtml(objectName)}
      </div>
      <p style="margin: 0; line-height: 1.5;">${escapeHtml(tavilyData.summary || '')}</p>
      ${sourcesHtml}
    </div>
  `;

  ragLog.appendChild(msgDiv);
  ragLog.scrollTop = ragLog.scrollHeight;

  // Speak concise summary through Web Audio
  if (tavilyData.summary) {
    const speakText = tavilyData.summary.split('.').slice(0, 2).join('.') + '.';
    speakSpatialAudio(speakText, { x: 0, y: 1.2, z: -1.5 });
  }

  // Dynamically update RAG preset chips for this object
  const ragPresets = document.getElementById('rag-presets');
  if (ragPresets && tavilyData.suggestedQuestions && tavilyData.suggestedQuestions.length > 0) {
    ragPresets.innerHTML = '';
    tavilyData.suggestedQuestions.forEach(q => {
      const chip = document.createElement('button');
      chip.className = 'chip-query';
      chip.textContent = q.length > 38 ? q.slice(0, 35) + '...' : q;
      chip.title = q;
      chip.dataset.query = q;
      chip.addEventListener('click', (e) => {
        e.preventDefault();
        const ragInput = document.getElementById('rag-input');
        if (ragInput) {
          ragInput.value = q;
          document.getElementById('rag-form')?.dispatchEvent(new Event('submit', { cancelable: true }));
        }
      });
      ragPresets.appendChild(chip);
    });
  }
}

async function executeEndToEndPipeline(file) {
  initSpatialAudio();
  if (btnAbortJob) btnAbortJob.classList.add('hidden');

  const fileName = (file?.name || '').toLowerCase();
  const fileMime = (file?.type || '').toLowerCase();
  const isImageFile = fileName.endsWith('.jpg') || fileName.endsWith('.jpeg') || fileName.endsWith('.png') || fileMime.startsWith('image/');
  const isVideoFile = fileName.endsWith('.mp4') || fileName.endsWith('.mov') || fileMime.startsWith('video/');

  if (!isImageFile && !isVideoFile) {
    showToast('Unsupported format. Please upload an MP4/MOV video or JPG/PNG image.', 'error', 5000);
    return;
  }

  // --- Smart Routing Logic ---
  // IF VIDEO (.mp4, .mov): Ensure pipeline is set to a video-compatible mode (Standard 3DGS or Triangle Splatting).
  // If it is currently on Instant Object, switch it to Standard 3DGS.
  if (isVideoFile) {
    if (STATE.trainingMethod === 'image2splat') {
      setTrainingMethod('3dgs');
      showToast('Walkthrough video detected — switched to Standard 3DGS pipeline', 'info', 3000);
    }
  }

  // IF IMAGE (.jpg, .jpeg, .png): Automatically switch the Training Pipeline toggle to Instant Object (Single Image).
  if (isImageFile) {
    if (STATE.trainingMethod !== 'image2splat') {
      setTrainingMethod('image2splat');
      showToast('Single photo detected — switched to Instant Object Scan mode', 'info', 3000);
    }
  }

  const isImageMode = STATE.trainingMethod === 'image2splat' || isImageFile;

  // File size check: Max 1024 MB (1 GB)
  const MAX_FILE_SIZE_MB = 1024;
  const fileSizeMb = file.size / (1024 * 1024);
  if (fileSizeMb > MAX_FILE_SIZE_MB) {
    showToast(`File exceeds max file size of 1GB / ${MAX_FILE_SIZE_MB}MB (${fileSizeMb.toFixed(1)}MB). Please compress or crop before uploading.`, 'error', 6000);
    return;
  }

  // Pre-Flight Metadata Check in Browser: ONLY for video pipelines (landscape vs portrait).
  // Images BYPASS this entirely!
  if (isVideoFile) {
    const orientation = await checkVideoOrientationInBrowser(file);
    if (orientation.isPortrait) {
      const portraitAudit = {
        readiness_score: 0,
        status: 'REJECTED',
        aspect_ratio_rejected: true,
        detected_context: 'vertical_video',
        evaluated_by: 'Browser Pre-Flight (Orientation Guard)',
        issues_detected: [
          {
            timestamp: '00:00',
            type: 'Invalid Format: Vertical Video Detected',
            detail: '3D Gaussian Splatting requires a wide horizontal field of view. Please rotate your camera to Landscape mode (horizontal) and capture the space again.',
            is_critical_blocker: true,
          }
        ],
        director_guidance: [
          'Rotate your recording device 90° into Landscape mode (horizontal) before filming.',
          'Maintain a wide horizontal field of view to allow COLMAP to identify overlapping feature points across frames.',
          'Capture a slow, continuous perimeter loop around the subject or space in 16:9 landscape aspect ratio.'
        ],
        laplacian_stats: {
          width: orientation.width,
          height: orientation.height,
          isPortrait: true,
        }
      };

      dropTitle.textContent = file.name;
      dropSubtitle.textContent = `Rejected: Vertical Video (${orientation.width}x${orientation.height})`;
      jobPill.className = 'badge error';
      jobPill.textContent = 'Rejected';
      showToast('Vertical video detected. 3DGS photogrammetry requires landscape footage.', 'error', 6000);
      await showQualityAuditModal(portraitAudit);
      return;
    }
  }

  dropTitle.textContent = file.name;
  dropSubtitle.textContent = `${fileSizeMb.toFixed(1)} MB • Processing Pipeline...`;
  jobPill.className = 'badge processing';
  jobPill.textContent = 'Ingesting';

  try {
    // --- Step 1: Upload ---
    setPipelineStep(1);
    const formData = new FormData();
    formData.append(isImageMode ? 'image' : 'video', file);

    progressCard.classList.remove('hidden');
    progressStage.textContent = `Uploading ${fileSizeMb.toFixed(1)} MB ${isImageMode ? 'photo' : 'video'}...`;
    progressBar.style.width = '5%';
    progressPct.textContent = '5%';

    if (btnCancelUpload) {
      btnCancelUpload.classList.remove('hidden');
      btnCancelUpload.disabled = false;
    }

    const uploadData = await uploadVideoWithProgress('/api/spatial/upload', formData, (loaded, total) => {
      const pct = Math.min(100, Math.round((loaded / total) * 100));
      const loadedMb = (loaded / (1024 * 1024)).toFixed(1);
      const totalMb = (total / (1024 * 1024)).toFixed(1);
      const scaledPct = Math.round(5 + (pct * 0.20)); // Scales from 5% to 25%
      progressBar.style.width = `${scaledPct}%`;
      progressPct.textContent = `${pct}%`;
      if (pct < 100) {
        progressStage.textContent = `Uploading: ${pct}% (${loadedMb} / ${totalMb} MB)...`;
      } else {
        progressStage.textContent = isImageMode
          ? 'Upload complete! Ingesting photo for Instant Object Scan...'
          : 'Upload complete! Running Pre-Flight AI Quality Audit & Extracting Frames...';
      }
    });
    if (btnCancelUpload) btnCancelUpload.classList.add('hidden');
    if (!uploadData.success) throw new Error(uploadData.error || 'Upload failed');

    // --- Action for IMAGES: Trigger Autonomous Vision RAG & Bypass Quality Audit Entirely ---
    if (isImageMode) {
      runAutonomousVisionRAG(uploadData.videoId);
    }

    // --- Action for VIDEOS: Duration Check & Pre-Flight AI Video Quality Gatekeeper Intercept ---
    if (isVideoFile) {
      const duration = uploadData.metadata?.durationSeconds || 0;
      if (duration > 90) {
        showToast(`Notice: Video duration is ${Math.round(duration)}s. Recommended length is 30–60s for optimal reconstruction speed.`, 'warning', 6000);
      }

      if (uploadData.audit) {
        const score = Math.round(uploadData.audit.readiness_score || 0);
        progressStage.textContent = `Pre-Flight AI Audit: Score ${score}% (${uploadData.audit.status})`;
        
        const auditDecision = await showQualityAuditModal(uploadData.audit);
        if (auditDecision.action === 'reupload' || auditDecision.action === 'cancel') {
          dropTitle.textContent = 'Drop a walkthrough video or single object photo';
          dropSubtitle.textContent = 'Supports MP4/MOV walkthroughs (max 1GB) or JPG/PNG photos for Instant 3D scan';
          jobPill.className = 'badge idle';
          jobPill.textContent = 'Idle';
          progressCard.classList.add('hidden');
          showToast('Video upload cancelled. Follow director guidance when recording for optimal results.', 'info', 4500);
          return;
        }
        if (auditDecision.action === 'bypass') {
          showToast('Proceeding with training via manual bypass. GPU compute credit warning acknowledged.', 'warning', 5000);
        }
      }
    }

    // --- Fast Path: Instant Object Scan (Image2Splat) ---
    if (isImageMode) {
      setPipelineStep(2);
      progressStage.textContent = 'Single object photo ingested; lifting 3D depth cloud...';
      progressBar.style.width = '30%';
      progressPct.textContent = '30%';

      setPipelineStep(3);
      progressStage.textContent = 'Configuring Image2Splat instant inference parameters...';
      progressBar.style.width = '45%';
      progressPct.textContent = '45%';

      nemotronCard.classList.remove('hidden');
      nemotronMatcherBadge.textContent = 'instant_inference';
      nemotronItersVal.textContent = 'Instant (Image2Splat)';
      nemotronFramesVal.textContent = '1 / 1';
      nemotronReasoningText.textContent = '"Bypassing COLMAP SfM feature matching. Deploying direct single-image Image2Splat depth unprojection on Nebius L40S."';

      setPipelineStep(4);
      jobPill.className = 'badge processing';
      jobPill.textContent = 'Generating';
      progressStage.textContent = 'Running Nebius GPU Image2Splat inference runner...';
      progressBar.style.width = '55%';
      progressPct.textContent = '55%';

      if (btnAbortJob) {
        btnAbortJob.classList.remove('hidden');
        btnAbortJob.disabled = false;
      }

      const jobData = await safeFetchJson('/api/spatial/job/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          videoId: uploadData.videoId,
          originalFilename: uploadData.originalName || uploadData.filename,
          deviceName: 'Instant Object (Single Image)',
          uploadTimestamp: new Date().toISOString(),
          colmapMatcher: 'none',
          recommendedIterations: 1,
          filteredFrameIndices: [0],
          training_method: 'image2splat',
        }),
      });
      if (!jobData.success) throw new Error(jobData.error || 'Failed to start instant object reconstruction');

      STATE.activeJobId = jobData.jobId;
      progressJobId.textContent = `job: ${jobData.jobId.slice(0, 8)}...`;
      await pollJobUntilReady(jobData.jobId, uploadData, { colmapMatcher: 'none' });
      return;
    }

    // --- Step 2: Dynamic Overlap Frame Extraction ---
    setPipelineStep(2);
    const targetMsg = uploadData.targetFrames ? ` (target: ${uploadData.targetFrames} overlap-preserving frames)` : '';
    progressStage.textContent = `Extracted ${uploadData.totalFrames} frames${targetMsg}`;
    progressBar.style.width = '30%';
    progressPct.textContent = '30%';

    // --- Step 3: Nemotron Reasoning ---
    setPipelineStep(3);
    progressStage.textContent = 'Querying nvidia/Nemotron-3-Ultra for 3DGS calibration...';
    progressBar.style.width = '45%';
    progressPct.textContent = '45%';

    const analyzeData = await safeFetchJson('/api/agent/analyze-frames', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        videoId: uploadData.videoId,
        totalFrames: uploadData.totalFrames,
        metadata: uploadData.metadata,
        framePaths: uploadData.frames,
      }),
    });
    if (!analyzeData.success) throw new Error(analyzeData.error || 'Nemotron analysis failed');

    const decisions = analyzeData.decisions;

    // Render Nemotron Decisions in Left Panel Card
    nemotronCard.classList.remove('hidden');
    nemotronMatcherBadge.textContent = decisions.colmapMatcher || 'vocab_tree';
    nemotronItersVal.textContent = (decisions.recommendedIterations || 10000).toLocaleString();
    const keptCount = decisions.filteredFrameIndices ? decisions.filteredFrameIndices.length : uploadData.totalFrames;
    nemotronFramesVal.textContent = `${keptCount}/${uploadData.totalFrames}`;
    nemotronReasoningText.textContent = `"${decisions.reasoning || 'Calibrated optimal baseline and iteration count for space capture.'}"`;

    // --- Step 4: Nebius GPU Reconstruction Training ---
    setPipelineStep(4);
    jobPill.className = 'badge processing';
    jobPill.textContent = 'Training';
    const isTriangle = STATE.trainingMethod === 'triangle';
    progressStage.textContent = isTriangle
      ? 'Initiating Nebius GPU Triangle Splatting runner...'
      : 'Initiating Nebius GPU 3DGS training runner...';
    progressBar.style.width = '20%';
    progressPct.textContent = '20%';

    // Reveal prominent red Abort Job button during TRAINING state
    if (btnAbortJob) {
      btnAbortJob.classList.remove('hidden');
      btnAbortJob.disabled = false;
    }

    const jobData = await safeFetchJson('/api/spatial/job/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        videoId: uploadData.videoId,
        originalFilename: uploadData.originalName || uploadData.filename,
        deviceName: uploadData.deviceName || (uploadData.metadata && uploadData.metadata.deviceName),
        uploadTimestamp: uploadData.uploadTimestamp || (uploadData.metadata && uploadData.metadata.uploadTimestamp),
        colmapMatcher: decisions.colmapMatcher,
        recommendedIterations: decisions.recommendedIterations || 10000,
        filteredFrameIndices: decisions.filteredFrameIndices,
        training_method: STATE.trainingMethod || '3dgs',
      }),
    });
    if (!jobData.success) throw new Error(jobData.error || 'Failed to start reconstruction job');

    STATE.activeJobId = jobData.jobId;
    localStorage.setItem('splatris_active_job_id', jobData.jobId);
    localStorage.setItem('splatris_active_scene_id', uploadData.videoId);
    updateButtonsForActiveJob(true);
    progressJobId.textContent = `job: ${jobData.jobId.slice(0, 8)}...`;

    if (jobData.isExisting) {
      showToast('Re-attached to active training job for this scene.', 'info', 3000);
    }

    // Poll until completed
    await pollJobUntilReady(jobData.jobId, uploadData, decisions);
  } catch (err) {
    if (btnAbortJob) btnAbortJob.classList.add('hidden');
    if (btnCancelUpload) btnCancelUpload.classList.add('hidden');
    if (err.name === 'AbortError' || (err.message && err.message.toLowerCase().includes('abort')) || (err.message && err.message.toLowerCase().includes('cancel'))) {
      console.log('[Splatris] Pipeline or upload aborted, skipping error modal.');
      return;
    }
    console.error('Pipeline error:', err);
    jobPill.className = 'badge idle';
    jobPill.textContent = 'Error';
    let cleanMsg = err.message || 'Pipeline processing failed';
    if (cleanMsg.includes('moov atom not found')) {
      cleanMsg = 'Video container error: "moov atom not found". The video file may be incomplete or corrupted.';
    } else if (cleanMsg.includes('FFmpeg frame extraction failed:')) {
      const lines = cleanMsg.split('\n').map(l => l.trim()).filter(l => l.length > 0);
      const errLine = lines.reverse().find(l => l.includes('Error') || l.includes('Invalid') || l.includes('failed')) || lines[0];
      cleanMsg = errLine.replace(/^Error:\s*/, '');
    }
    progressStage.textContent = `Error: ${cleanMsg}`;
    alert(`Pipeline Error: ${cleanMsg}`);
  }
}

function pollJobUntilReady(jobId, uploadData, decisions) {
  return new Promise((resolve) => {
    let seconds = 0;
    if (STATE.pollInterval) clearInterval(STATE.pollInterval);
    const interval = setInterval(async () => {
      seconds++;
      if (progressTimer) progressTimer.textContent = `${seconds}s`;

      try {
        const res = await fetch(`/api/spatial/job/status/${jobId}`);
        // Defensive check: guard against gateway timeout / 502 / 504 HTML error pages
        if (!res.ok) {
          console.warn(`[Splatris] Job status polling HTTP ${res.status}: ${res.statusText}`);
          return;
        }
        const contentType = res.headers.get('content-type') || '';
        if (!contentType.includes('application/json')) {
          console.warn(`[Splatris] Job status expected JSON but received ${contentType}`);
          return;
        }
        const data = await res.json();

        if (data.success) {
          const pct = Math.max(20, Math.min(100, data.progress || 20));
          progressBar.style.width = `${pct}%`;
          progressPct.textContent = `${pct}%`;
          progressStage.textContent = data.stage || 'Optimizing 3D Gaussians...';

          if (data.status === 'suspended' || data.status === 'gpu_offline') {
            jobPill.className = 'badge warning';
            jobPill.textContent = 'GPU Offline';
            progressStage.textContent = data.stage || 'GPU Instance Preempted/Offline';
            showGpuSuspendedBanner(data.error || 'Connection timed out to remote worker. Your progress is saved.');
            updateButtonsForActiveJob(true);
            return;
          }

          if (data.status === 'processing') {
            hideGpuSuspendedBanner();
            if (jobPill) {
              jobPill.className = 'badge processing';
              jobPill.textContent = 'Training';
            }
            updateButtonsForActiveJob(true);
          }

          if (data.status === 'aborted') {
            clearInterval(interval);
            STATE.pollInterval = null;
            localStorage.removeItem('splatris_active_job_id');
            localStorage.removeItem('splatris_active_scene_id');
            hideGpuSuspendedBanner();
            updateButtonsForActiveJob(false);
            resetPipelineUI({ notify: false });
            return;
          }

          if (data.status === 'failed') {
            clearInterval(interval);
            STATE.pollInterval = null;
            localStorage.removeItem('splatris_active_job_id');
            localStorage.removeItem('splatris_active_scene_id');
            hideGpuSuspendedBanner();
            updateButtonsForActiveJob(false);
            if (btnAbortJob) btnAbortJob.classList.add('hidden');
            jobPill.className = 'badge idle';
            jobPill.textContent = 'Failed';
            progressStage.textContent = `Training Error: ${data.error || '3DGS training failed'}`;
            showToast(`Training Failed: ${data.error || 'Unknown error'}`, 'error', 6000);
            return;
          }

          if (data.status === 'completed') {
            clearInterval(interval);
            STATE.pollInterval = null;
            localStorage.removeItem('splatris_active_job_id');
            localStorage.removeItem('splatris_active_scene_id');
            hideGpuSuspendedBanner();
            updateButtonsForActiveJob(false);
            if (btnAbortJob) btnAbortJob.classList.add('hidden');
            setPipelineStep(5);
            jobPill.className = 'badge completed';
            jobPill.textContent = 'Ready in WebXR';
            dropSubtitle.textContent = 'Reconstruction Complete • Active in WebXR';

            // --- Step 5: Dynamic 3D Scene Replacement ---
            const modelUrl = data.outputModelUrl || '/uploads/models/scene.splat';
            STATE.activeModelUrl = modelUrl;
            await loadGaussianSplatScene(modelUrl, true);
            await loadRecentScenes();

            // --- Step 6: Spatial Audio RAG Context Synchronization ---
            STATE.currentSceneContext = {
              videoId: uploadData.videoId,
              metadata: uploadData.metadata,
              nemotronDecisions: decisions,
              totalFrames: uploadData.totalFrames,
              modelUrl,
            };

            // Update Presets for the new space
            updateSpaceRagPresets(uploadData.videoId);

            // Post Guide Announcement
            const methodTitle = (data.trainingMethod || STATE.trainingMethod) === 'triangle'
              ? 'Triangle Splatting (Mesh-based)'
              : '3D Gaussians';
            appendChatMessage(
              'agent',
              `🎉 Space reconstruction for "${uploadData.originalName}" is complete! ${methodTitle} model loaded into viewport. You can now explore the space in WebXR or free roam.`
            );

            resolve();
          }
        }
      } catch (err) {
        console.warn('Polling status warning:', err);
      }
    }, 1000);
    STATE.pollInterval = interval;
  });
}

function updateSpaceRagPresets(videoId) {
  const presetsContainer = document.getElementById('rag-presets');
  if (!presetsContainer) return;

  presetsContainer.innerHTML = `
    <button class="chip-query" data-query="Describe the spatial dimensions and layout of this space">Space Layout</button>
    <button class="chip-query" data-query="Explain why Nemotron chose ${nemotronMatcherBadge.textContent} matching for this video">Matching Rationale</button>
    <button class="chip-query" data-query="Assess the 3D Gaussian density and lighting in the space center">Space Lighting & Center</button>
  `;

  presetsContainer.querySelectorAll('.chip-query').forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.getAttribute('data-query');
      if (query) {
        ragInput.value = query;
        submitRagQuery(query);
      }
    });
  });
}

// --- WebXR Button Handler ---
const btnWebXR = document.getElementById('btn-webxr');
if (btnWebXR) {
  btnWebXR.addEventListener('click', async () => {
    initSpatialAudio();
    if (!navigator.xr) {
      alert('WebXR is not supported in this browser. You can explore the 3D space using touch or mouse.');
      return;
    }

    try {
      // If already presenting, toggle off cleanly
      if (renderer.xr.isPresenting) {
        const currentSession = renderer.xr.getSession();
        if (currentSession) await currentSession.end();
        return;
      }

      const supported = await navigator.xr.isSessionSupported('immersive-vr');
      if (supported) {
        if (renderer.xr.setFramebufferScaleFactor) {
          renderer.xr.setFramebufferScaleFactor(0.7);
        }
        if (renderer.xr.setFoveation) {
          renderer.xr.setFoveation(1.0);
        }

        // 3. Verify and enforce local-floor reference space for 6DoF physical room-scale walking
        try {
          renderer.xr.setReferenceSpaceType('local-floor');
          console.log('[WebXR] Reference space type explicitly set to local-floor');
        } catch (e) {
          console.warn('[WebXR] Could not set local-floor reference space type:', e);
        }

        let session;
        try {
          session = await navigator.xr.requestSession('immersive-vr', {
            requiredFeatures: ['local-floor'],
            optionalFeatures: ['bounded-floor', 'hand-tracking'],
          });
        } catch (rfErr) {
          console.warn('[WebXR] Falling back to optional local-floor request:', rfErr);
          session = await navigator.xr.requestSession('immersive-vr', {
            optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking'],
          });
        }
        await renderer.xr.setSession(session);

        // 1. Disable Frustum Culling on main containers so room-scale walk never culls the environment
        if (STATE.splatPivotGroup) STATE.splatPivotGroup.frustumCulled = false;
        if (STATE.dropInViewer) {
          STATE.dropInViewer.frustumCulled = false;
          if (STATE.dropInViewer.splatMesh) STATE.dropInViewer.splatMesh.frustumCulled = false;
        }

        // 2. Increase Camera Far Plane to 1000m
        camera.near = 0.01;
        camera.far = 1000.0;
        camera.updateProjectionMatrix();

        const activeXrCam = renderer.xr.getCamera();
        if (activeXrCam) {
          activeXrCam.near = 0.01;
          activeXrCam.far = 1000.0;
          if (activeXrCam.cameras && activeXrCam.cameras.length > 0) {
            activeXrCam.cameras.forEach(c => {
              c.near = 0.01;
              c.far = 1000.0;
              // Note: Never call c.updateProjectionMatrix() on XR eye cameras as that destroys asymmetric stereoscopic matrices
            });
          }
        }

        // Reinforce Quest 3 Fixed Foveated Rendering and disable MSAA
        if (renderer.xr.setFoveation) {
          renderer.xr.setFoveation(1.0);
        }
        if (session.renderState && session.renderState.baseLayer) {
          if ('fixedFoveation' in session.renderState.baseLayer) {
            session.renderState.baseLayer.fixedFoveation = 1.0;
          }
        }
        const activeRt = renderer.getRenderTarget();
        if (activeRt && activeRt.samples !== undefined) {
          activeRt.samples = 0; // Aggressively bypass MSAA
        }

        controls.enabled = false;
        if (btnWebXR.querySelector('span')) {
          btnWebXR.querySelector('span').textContent = 'Exit WebXR';
        }

        // Position CameraRig in front of the tank at eye level
        if (STATE.isObjectScan) {
          cameraRig.position.set(0, 0, 1.6);
        } else {
          cameraRig.position.set(0, 0, 0.4);
        }
        cameraRig.rotation.set(0, 0, 0);

        // Display in-VR head-locked floating HUD card (attached to camera at top-left FOV)
        createXRInWorldHUD();
        if (xrHudMesh) {
          if (xrHudMesh.parent) {
            xrHudMesh.parent.remove(xrHudMesh);
          }
          camera.add(xrHudMesh);
          xrHudMesh.position.set(-0.5, 0.3, -1.5);
          xrHudMesh.rotation.set(-0.06, 0.22, 0);
          xrHudMesh.visible = true;
          console.log('[WebXR] Head-locked HUD attached to camera at (-0.5, 0.3, -1.5)');
        }

        // Attach THREE.AudioListener to the active XR camera so spatial audio doesn't drop to 0 when walking
        initSpatialAudio();
        if (threeAudioListener) {
          const xrCam = renderer.xr.getCamera(camera);
          if (threeAudioListener.parent) {
            threeAudioListener.parent.remove(threeAudioListener);
          }
          xrCam.add(threeAudioListener);
          console.log('[SpatialAudio] Attached THREE.AudioListener to active WebXR Camera.');
        }

        session.addEventListener('end', () => {
          window.__splatrisXrActive = false;
          controls.enabled = true;
          if (btnWebXR && btnWebXR.querySelector('span')) {
            btnWebXR.querySelector('span').textContent = 'Enter WebXR';
          }
          cameraRig.position.set(0, 0, 0);
          cameraRig.rotation.set(0, 0, 0);
          camera.position.set(0, 1.25, 0.4);
          camera.near = 0.01;
          camera.far = 1000.0;
          camera.updateProjectionMatrix();

          // Restore desktop 3DGS / triangle splat scale to 1.0
          if (STATE.dropInViewer && STATE.dropInViewer.splatMesh) {
            if (STATE.dropInViewer.splatMesh.setSplatScale) {
              STATE.dropInViewer.splatMesh.setSplatScale(1.0);
            } else if (STATE.dropInViewer.splatMesh.material && STATE.dropInViewer.splatMesh.material.uniforms && STATE.dropInViewer.splatMesh.material.uniforms.splatScale) {
              STATE.dropInViewer.splatMesh.material.uniforms.splatScale.value = 1.0;
            }
          }

          if (xrHudMesh) {
            xrHudMesh.visible = false;
            if (xrHudMesh.parent) {
              xrHudMesh.parent.remove(xrHudMesh);
            }
          }
          if (threeAudioListener) {
            if (threeAudioListener.parent) {
              threeAudioListener.parent.remove(threeAudioListener);
            }
            camera.add(threeAudioListener);
            console.log('[SpatialAudio] Re-attached THREE.AudioListener to default perspective camera.');
          }
          showToast('Exited WebXR. Returned to 2D browser view.', 'info', 3000);
        });
      } else {
        alert('Immersive VR session not supported on this device. Switching to full-screen 3D view.');
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen();
        } else {
          document.exitFullscreen();
        }
      }
    } catch (err) {
      console.warn('WebXR session request notice:', err);
      alert(`WebXR: ${err.message || 'Unable to launch VR session'}`);
    }
  });
}

// --- Frame Inspection Lightbox Controller ---
const frameLightboxModal = document.getElementById('frame-lightbox-modal');
const lightboxImg = document.getElementById('lightbox-img');
const lightboxTimestamp = document.getElementById('lightbox-timestamp');
const lightboxType = document.getElementById('lightbox-type');
const lightboxDetail = document.getElementById('lightbox-detail');
const btnCloseLightbox = document.getElementById('btn-close-lightbox');

function openFrameLightbox(thumbUrl, type, timestamp, detail) {
  if (!frameLightboxModal || !lightboxImg) return;
  lightboxImg.src = thumbUrl;
  if (lightboxTimestamp) lightboxTimestamp.textContent = timestamp || '00:00';
  if (lightboxType) lightboxType.textContent = type || 'Flagged Frame';
  if (lightboxDetail) lightboxDetail.textContent = detail || '';
  frameLightboxModal.classList.remove('hidden');
}

function closeFrameLightbox() {
  if (frameLightboxModal) frameLightboxModal.classList.add('hidden');
}

if (btnCloseLightbox) {
  btnCloseLightbox.addEventListener('click', closeFrameLightbox);
}
if (frameLightboxModal) {
  frameLightboxModal.addEventListener('click', (e) => {
    if (e.target === frameLightboxModal) closeFrameLightbox();
  });
}

// --- First-Person Free Roam (WASD) Controller ---
const btnFreeRoam = document.getElementById('btn-free-roam');
const instructionHint = document.getElementById('instruction-hint');
let moveForward = false, moveBackward = false, moveLeft = false, moveRight = false, moveUp = false, moveDown = false;
const freeRoamEuler = new THREE.Euler(0, 0, 0, 'YXZ');

function onMouseMovePointerLock(event) {
  if (!STATE.isFreeRoam) return;
  const movementX = event.movementX || event.mozMovementX || event.webkitMovementX || 0;
  const movementY = event.movementY || event.mozMovementY || event.webkitMovementY || 0;
  freeRoamEuler.setFromQuaternion(camera.quaternion);
  freeRoamEuler.y -= movementX * 0.0022;
  freeRoamEuler.x -= movementY * 0.0022;
  freeRoamEuler.x = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, freeRoamEuler.x));
  camera.quaternion.setFromEuler(freeRoamEuler);
}

function onKeyDownFreeRoam(event) {
  if (document.activeElement && (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA')) {
    return;
  }
  if (!STATE.isFreeRoam) return;
  switch (event.code) {
    case 'ArrowUp':
    case 'KeyW': moveForward = true; break;
    case 'ArrowLeft':
    case 'KeyA': moveLeft = true; break;
    case 'ArrowDown':
    case 'KeyS': moveBackward = true; break;
    case 'ArrowRight':
    case 'KeyD': moveRight = true; break;
    case 'Space':
    case 'KeyE': moveUp = true; break;
    case 'ShiftLeft':
    case 'ShiftRight':
    case 'KeyQ': moveDown = true; break;
  }
}

function onKeyUpFreeRoam(event) {
  switch (event.code) {
    case 'ArrowUp':
    case 'KeyW': moveForward = false; break;
    case 'ArrowLeft':
    case 'KeyA': moveLeft = false; break;
    case 'ArrowDown':
    case 'KeyS': moveBackward = false; break;
    case 'ArrowRight':
    case 'KeyD': moveRight = false; break;
    case 'Space':
    case 'KeyE': moveUp = false; break;
    case 'ShiftLeft':
    case 'ShiftRight':
    case 'KeyQ': moveDown = false; break;
  }
}

function enterFreeRoam() {
  STATE.autoOrbit = false;
  if (btnFlyCam) btnFlyCam.classList.remove('active');
  const targetEl = renderer.domElement;
  if (targetEl.requestPointerLock) {
    targetEl.requestPointerLock();
  } else if (document.body.requestPointerLock) {
    document.body.requestPointerLock();
  }
}

const btnOverlayFreeRoam = document.getElementById('btn-overlay-free-roam');
const btnOverlayFreeRoamText = document.getElementById('btn-overlay-free-roam-text');

function onPointerLockChange() {
  const isLocked = document.pointerLockElement === renderer.domElement || document.pointerLockElement === document.body;
  STATE.isFreeRoam = isLocked;
  if (btnFreeRoam) btnFreeRoam.classList.toggle('active', isLocked);
  if (btnOverlayFreeRoam) btnOverlayFreeRoam.classList.toggle('active', isLocked);
  if (btnOverlayFreeRoamText) {
    btnOverlayFreeRoamText.textContent = isLocked ? 'Exit Free Roam (ESC)' : 'Enter Free Roam';
  }
  setControlsEnabled(!isLocked);

  if (isLocked) {
    freeRoamEuler.setFromQuaternion(camera.quaternion);
    if (instructionHint) instructionHint.textContent = 'W / ↑: Fly in Gaze Direction • S / ↓: Reverse • A / D: Strafe • Space / Shift: Elevate • ESC: Exit';
    showToast('6DoF Free Roam Active: Move along Mouse Gaze Direction (W/↑: Forward, S/↓: Reverse)', 'info', 3500);
  } else {
    moveForward = moveBackward = moveLeft = moveRight = moveUp = moveDown = false;
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    updateCameraTargets(camera.position.clone().addScaledVector(dir, 2.0));
    if (instructionHint) instructionHint.textContent = 'Left Drag: Orbit/Rotate (3-Axis) • Right Drag: Pan • Scroll: Zoom • Meta Quest 3 Ready';
  }
}

document.addEventListener('pointerlockchange', onPointerLockChange);
document.addEventListener('mousemove', onMouseMovePointerLock);
window.addEventListener('keydown', onKeyDownFreeRoam);
window.addEventListener('keyup', onKeyUpFreeRoam);

function toggleFreeRoam() {
  if (STATE.isFreeRoam) {
    document.exitPointerLock();
  } else {
    enterFreeRoam();
  }
}

if (btnFreeRoam) btnFreeRoam.addEventListener('click', toggleFreeRoam);
if (btnOverlayFreeRoam) btnOverlayFreeRoam.addEventListener('click', toggleFreeRoam);

// --- Turntable and Vertical Axis Rotation Controls ---
const btnTurntableSpin = document.getElementById('btn-turntable-spin');
const btnTurntableText = document.getElementById('btn-turntable-text');
const btnRotateLeft = document.getElementById('btn-rotate-left');
const btnRotateRight = document.getElementById('btn-rotate-right');
const btnResetRotation = document.getElementById('btn-reset-rotation');

function toggleTurntable(forceState) {
  const next = typeof forceState === 'boolean' ? forceState : !STATE.turntableActive;
  STATE.turntableActive = next;
  STATE.autoOrbit = next;

  if (btnTurntableSpin) {
    btnTurntableSpin.classList.toggle('active', next);
    if (btnTurntableText) btnTurntableText.textContent = next ? 'Spinning...' : 'Turntable Spin';
  }
  if (btnFlyCam) {
    btnFlyCam.classList.toggle('active', next);
  }
}

if (btnTurntableSpin) {
  btnTurntableSpin.addEventListener('click', () => toggleTurntable());
}

if (btnRotateLeft) {
  btnRotateLeft.addEventListener('click', () => {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.rotation.y -= Math.PI / 12; // -15 deg
    }
  });
}

if (btnRotateRight) {
  btnRotateRight.addEventListener('click', () => {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.rotation.y += Math.PI / 12; // +15 deg
    }
  });
}

if (btnResetRotation) {
  btnResetRotation.addEventListener('click', () => {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.rotation.set(0, 0, 0);
    }
    const targetY = STATE.pivotTargetY || 1.0;
    camera.up.set(0, 1, 0);
    const viewDist = STATE.isObjectScan ? 2.3 : 1.8;
    camera.position.set(0, targetY + 0.35, viewDist);
    updateCameraTargets(0, targetY, 0);
    if (trackballControls) {
      trackballControls.target.set(0, targetY, 0);
      trackballControls.update();
    }
    controls.target.set(0, targetY, 0);
    controls.update();
    toggleTurntable(false);
  });
}

// --- Viewport Control Buttons ---
const btnFlyCam = document.getElementById('btn-fly-cam');
if (btnFlyCam) {
  btnFlyCam.addEventListener('click', () => {
    if (STATE.isFreeRoam) document.exitPointerLock();
    toggleTurntable();
  });
}

const btnResetCam = document.getElementById('btn-reset-cam');
if (btnResetCam) {
  btnResetCam.addEventListener('click', () => {
    if (STATE.isFreeRoam) document.exitPointerLock();
    toggleTurntable(false);
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.rotation.set(0, 0, 0);
    }
    const targetY = STATE.pivotTargetY || STATE.eyeHeight || 1.0;
    camera.up.set(0, 1, 0);
    const viewDist = STATE.isObjectScan ? 2.3 : 1.8;
    camera.position.set(0, targetY + 0.35, viewDist);
    updateCameraTargets(0, targetY, 0);
    if (trackballControls) {
      trackballControls.target.set(0, targetY, 0);
      trackballControls.update();
    }
    controls.target.set(0, targetY, 0);
    controls.update();
  });
}

// Alt + Drag or Shift + Drag to spin object directly around vertical axis with mouse
let isDirectSpinning = false;
let spinStartX = 0;

window.addEventListener('mousedown', (e) => {
  if (e.target.tagName === 'CANVAS' && (e.altKey || e.shiftKey)) {
    isDirectSpinning = true;
    spinStartX = e.clientX;
    setControlsEnabled(false);
  }
});

window.addEventListener('mousemove', (e) => {
  if (isDirectSpinning && STATE.splatPivotGroup) {
    const dx = e.clientX - spinStartX;
    spinStartX = e.clientX;
    STATE.splatPivotGroup.rotation.y += dx * 0.01;
  }
});

window.addEventListener('mouseup', () => {
  if (isDirectSpinning) {
    isDirectSpinning = false;
    setControlsEnabled(true);
  }
});

// Keyboard shortcuts for rotating object around vertical axis
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;

  if (e.key === 'q' || e.key === 'Q' || e.key === 'ArrowLeft') {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.rotation.y -= 0.08; // ~4.5 deg left
    }
  } else if (e.key === 'e' || e.key === 'E' || e.key === 'ArrowRight') {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.rotation.y += 0.08; // ~4.5 deg right
    }
  } else if (e.key === ' ' && !STATE.isFreeRoam) {
    // Spacebar toggles turntable spin
    e.preventDefault();
    toggleTurntable();
  } else if (e.key === 'r' || e.key === 'R') {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.rotation.set(0, 0, 0);
    }
  }
});

const btnTogglePoints = document.getElementById('btn-toggle-points');
if (btnTogglePoints) {
  btnTogglePoints.addEventListener('click', () => {
    if (STATE.dropInViewer) {
      STATE.dropInViewer.visible = !STATE.dropInViewer.visible;
      btnTogglePoints.classList.toggle('active', STATE.dropInViewer.visible);
    }
  });
}

// --- Spatial Audio RAG Panel Integration ---
const ragForm = document.getElementById('rag-form');
const ragInput = document.getElementById('rag-input');
const chatLog = document.getElementById('rag-chat-log');
const waveAnim = document.getElementById('audio-wave-anim');
const btnToggleAudio = document.getElementById('btn-toggle-audio');

if (btnToggleAudio) {
  btnToggleAudio.addEventListener('click', () => {
    STATE.audioEnabled = !STATE.audioEnabled;
    btnToggleAudio.style.opacity = STATE.audioEnabled ? '1' : '0.4';
    if (!STATE.audioEnabled) {
      stopSpatialAudio();
    }
  });
}

// Preset chips click
document.querySelectorAll('.chip-query').forEach((chip) => {
  chip.addEventListener('click', () => {
    const query = chip.getAttribute('data-query');
    if (query) {
      ragInput.value = query;
      submitRagQuery(query);
    }
  });
});

if (ragForm) {
  ragForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = ragInput.value.trim();
    if (!query) return;
    ragInput.value = '';
    submitRagQuery(query);
  });
}

async function submitRagQuery(query) {
  initSpatialAudio();
  appendChatMessage('user', query);

  const typingId = appendChatMessage('agent', 'Analyzing 3D spatial scene & querying grounding sources...');

  try {
    const coords = {
      x: camera.position.x,
      y: camera.position.y,
      z: camera.position.z,
    };

    const data = await safeFetchJson('/api/spatial/rag-query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        spatialContext: {
          activeModel: STATE.activeModelUrl,
          splatCount: STATE.currentSplatCount || 786568,
          cameraFov: camera.fov,
          sceneMetadata: STATE.currentSceneContext.metadata,
          nemotronDecisions: STATE.currentSceneContext.nemotronDecisions,
          identifiedObject: STATE.lastInspectedObject?.objectName,
          objectCategory: STATE.lastInspectedObject?.category,
          visualDescription: STATE.lastInspectedObject?.visualDescription,
          objectGrounding: STATE.lastInspectedObject?.tavilyData?.summary,
        },
        videoId: STATE.currentSceneContext.videoId,
        coordinates: coords,
      }),
    });
    if (!data.success) throw new Error(data.error || 'Failed to process RAG query');

    updateChatMessage(typingId, data.answer, data.sources, data.groundingUsed);

    if (data.audioSpatialPosition) {
      const pos = data.audioSpatialPosition;
      updateSpatialEmitterPosition(pos.x || 0, pos.y || 1.2, pos.z || -2.0);
    }

    if (STATE.audioEnabled) {
      speakSpatialAudio(data.answer, data.audioSpatialPosition);
    }
  } catch (err) {
    console.error('RAG query error:', err);
    updateChatMessage(typingId, `Spatial AI Guide temporarily unavailable: ${err.message}`, []);
  }
}

let activeSpatialAudioSource = null;
let activeUtterance = null; // CRITICAL: Retain global reference to prevent V8 garbage collection mid-playback
let ttsSentenceQueue = [];
let ttsCurrentSentenceIndex = 0;
let ttsPlaybackActive = false;
let ttsActiveFinishCallback = null;
let ttsHeartbeatTimer = null;

/**
 * Splits incoming text into an array of clean, digestible sentence chunks for TTS.
 * Bypasses browser character and ~15s timeout limits.
 */
function splitTextIntoSentences(text) {
  if (!text) return [];
  // Strip markdown formatting, URLs, and normalize whitespace
  const clean = text
    .replace(/[*_#`~[\]()]/g, ' ')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/\s+/g, ' ')
    .trim();
  if (!clean) return [];

  let rawSentences = [];
  // Match sentences ending in ., !, or ? (preserving punctuation)
  const matches = clean.match(/[^.!?]+[.!?]+(?:\s+|$)/g);
  if (matches && matches.length > 0) {
    rawSentences = matches.map((s) => s.trim()).filter(Boolean);
    const matchedLen = matches.join('').length;
    if (matchedLen < clean.length) {
      const remainder = clean.slice(matchedLen).trim();
      if (remainder) rawSentences.push(remainder);
    }
  } else {
    rawSentences = clean.split(/(?<=[.!?])\s+/).map((s) => s.trim()).filter(Boolean);
    if (rawSentences.length === 0) rawSentences = [clean];
  }

  // Ensure no single sentence chunk exceeds 180 chars to prevent browser engine speech stalls
  const finalChunks = [];
  for (const s of rawSentences) {
    if (s.length <= 180) {
      finalChunks.push(s);
    } else {
      const subMatches = s.match(/[^,;]+[,;]+(?:\s+|$)/g);
      if (subMatches && subMatches.length > 0) {
        subMatches.forEach((p) => {
          const t = p.trim();
          if (t) finalChunks.push(t);
        });
        const subMatchedLen = subMatches.join('').length;
        if (subMatchedLen < s.length) {
          const remainder = s.slice(subMatchedLen).trim();
          if (remainder) finalChunks.push(remainder);
        }
      } else {
        const words = s.split(' ');
        let cur = '';
        for (const w of words) {
          if ((cur + ' ' + w).length > 140 && cur) {
            finalChunks.push(cur.trim());
            cur = w;
          } else {
            cur = cur ? cur + ' ' + w : w;
          }
        }
        if (cur.trim()) finalChunks.push(cur.trim());
      }
    }
  }
  return finalChunks.filter((s) => s.length > 0);
}

/**
 * Stops ongoing spatial speech and resets sentence queue & visual emitter.
 */
function stopSpatialAudio() {
  ttsPlaybackActive = false;
  ttsSentenceQueue = [];
  ttsCurrentSentenceIndex = 0;
  if (ttsHeartbeatTimer) {
    clearInterval(ttsHeartbeatTimer);
    ttsHeartbeatTimer = null;
  }
  if (activeSpatialAudioSource) {
    try {
      activeSpatialAudioSource.stop();
      activeSpatialAudioSource.disconnect();
    } catch (e) {}
    activeSpatialAudioSource = null;
  }
  if ('speechSynthesis' in window) {
    try {
      window.speechSynthesis.cancel();
    } catch (e) {}
  }
  activeUtterance = null;
  if (waveAnim) waveAnim.classList.remove('active');
  if (typeof ttsActiveFinishCallback === 'function') {
    const cb = ttsActiveFinishCallback;
    ttsActiveFinishCallback = null;
    try {
      cb();
    } catch (e) {}
  }
}

/**
 * Sequentially triggers the next sentence in the queue.
 * Each sentence gets a new SpeechSynthesisUtterance, chained via onend.
 */
function playNextTtsSentenceInQueue() {
  if (!ttsPlaybackActive) {
    stopSpatialAudio();
    return;
  }

  if (ttsCurrentSentenceIndex >= ttsSentenceQueue.length) {
    console.log(`[SpatialTTS] Finished reading all ${ttsSentenceQueue.length} sentences successfully.`);
    const finish = ttsActiveFinishCallback;
    stopSpatialAudio();
    if (typeof finish === 'function') {
      finish();
    }
    return;
  }

  const sentence = ttsSentenceQueue[ttsCurrentSentenceIndex];
  console.log(`[SpatialTTS] Speaking sentence ${ttsCurrentSentenceIndex + 1}/${ttsSentenceQueue.length}: "${sentence}"`);

  try {
    const utterance = new SpeechSynthesisUtterance(sentence);
    utterance.rate = 1.03;
    utterance.pitch = 1.0;
    utterance.lang = 'en-US';

    // Retain global reference so V8 does not garbage collect it mid-playback
    activeUtterance = utterance;

    utterance.onstart = () => {
      if (waveAnim) waveAnim.classList.add('active');
    };

    utterance.onend = () => {
      if (!ttsPlaybackActive) return;
      // Instantly advance to the next sentence in the queue
      ttsCurrentSentenceIndex++;
      playNextTtsSentenceInQueue();
    };

    utterance.onerror = (event) => {
      console.warn(`[SpatialTTS] Sentence ${ttsCurrentSentenceIndex + 1} playback notice:`, event?.error);
      if (!ttsPlaybackActive) return;
      if (event?.error === 'interrupted' || event?.error === 'canceled') {
        return;
      }
      ttsCurrentSentenceIndex++;
      playNextTtsSentenceInQueue();
    };

    // Unpause if browser speech engine is suspended
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }

    window.speechSynthesis.speak(utterance);
  } catch (err) {
    console.warn('[SpatialTTS] Error synthesizing sentence:', err);
    ttsCurrentSentenceIndex++;
    playNextTtsSentenceInQueue();
  }
}

async function speakSpatialAudio(text, position, onComplete) {
  // 1. Clean stop of any previous speech/queue
  stopSpatialAudio();

  if (!text || typeof text !== 'string' || !text.trim()) {
    if (typeof onComplete === 'function') onComplete();
    return;
  }

  initSpatialAudio();

  // 2. Position spatial 3D audio panner & visual emitter at the target object coordinates
  if (position && typeof position.x === 'number') {
    updateSpatialEmitterPosition(position.x, position.y, position.z);
  }

  // 3. Play acoustic beacon feedback chime emanating directly from the object
  triggerSpatialChime(560, 0.3);

  // 4. Split incoming text into an array of smaller sentence chunks (Queue System)
  const sentences = splitTextIntoSentences(text);
  if (sentences.length === 0) {
    if (typeof onComplete === 'function') onComplete();
    return;
  }

  console.log(`[SpatialTTS] Queued ${sentences.length} sentences for sequential spatial playback.`);

  if (!('speechSynthesis' in window)) {
    console.warn('[SpatialTTS] SpeechSynthesis API unavailable in this browser.');
    if (typeof onComplete === 'function') onComplete();
    return;
  }

  ttsSentenceQueue = sentences;
  ttsCurrentSentenceIndex = 0;
  ttsPlaybackActive = true;
  ttsActiveFinishCallback = onComplete;

  // 5. Chrome watchdog timer: prevents background suspension during long multi-sentence playback
  if (ttsHeartbeatTimer) clearInterval(ttsHeartbeatTimer);
  ttsHeartbeatTimer = setInterval(() => {
    if (ttsPlaybackActive && 'speechSynthesis' in window && window.speechSynthesis.speaking) {
      window.speechSynthesis.pause();
      window.speechSynthesis.resume();
    }
  }, 10000);

  // 6. Begin sequential sentence queue playback
  playNextTtsSentenceInQueue();
}

function appendChatMessage(sender, text) {
  const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5);
  const div = document.createElement('div');
  div.id = id;
  div.className = `message ${sender}-msg`;

  if (sender === 'user') {
    div.textContent = text;
  } else {
    div.innerHTML = `
      <div class="msg-header">
        <span class="agent-avatar">AI</span>
        <span class="agent-name">Spatial Audio Guide</span>
      </div>
      <div class="msg-content">${text}</div>
    `;
  }

  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return id;
}

function updateChatMessage(id, text, sources = [], grounding = '') {
  const el = document.getElementById(id);
  if (!el) return;

  const contentEl = el.querySelector('.msg-content');
  if (contentEl) {
    contentEl.textContent = text;
  }

  if (sources && sources.length > 0) {
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'rag-sources';
    sourcesDiv.textContent = `Grounding: ${grounding || 'Web Search'} (${sources.length} sources)`;
    el.appendChild(sourcesDiv);
  }

  chatLog.scrollTop = chatLog.scrollHeight;
}

// --- WebGL Viewport Screenshot & Targeted Bounding Box Extraction ---
/**
 * Captures a targeted or full WebGL canvas viewport snapshot as a JPEG base64 data URL.
 * When targetPoint (Vector3) is supplied, uses a dedicated snapshot perspective camera
 * looking directly at the target object from the user's active viewpoint, guaranteeing
 * perfect centering and a generous context window (FOV ~52°) without ceiling bias or headset tilt distortion.
 * @param {object} options
 * @param {THREE.Vector3} [options.targetPoint] - 3D world coordinate to center snapshot upon
 * @param {boolean} [options.crop=false] - Whether to crop a bounding box (2D screen coordinate fallback)
 * @param {number} [options.screenX] - CSS X coordinate of click/target center
 * @param {number} [options.screenY] - CSS Y coordinate of click/target center
 * @param {number} [options.boxSize=540] - CSS bounding box dimension in pixels
 * @returns {string|null} Base64 data URL (image/jpeg) or null
 */
function takeScreenshot({ targetPoint = null, crop = false, screenX = null, screenY = null, boxSize = 540 } = {}) {
  try {
    const canvasEl = renderer?.domElement || document.getElementById('webgl-canvas');
    if (!canvasEl) return null;

    // Temporarily hide 3D reticle marker and XR HUD so they do not occlude the target object
    const wasReticleVisible = raycastTargetMesh ? raycastTargetMesh.visible : false;
    if (raycastTargetMesh) raycastTargetMesh.visible = false;
    const wasHudVisible = xrHudMesh ? xrHudMesh.visible : false;
    if (xrHudMesh) xrHudMesh.visible = false;

    const activeCam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;

    // PATH A: Targeted 3D Object Centered Snapshot (Guaranteed center + wide context window)
    if (targetPoint) {
      const renderSize = 768;
      const snapTarget = new THREE.WebGLRenderTarget(renderSize, renderSize, {
        format: THREE.RGBAFormat,
        type: THREE.UnsignedByteType,
      });

      // Compute viewer eye position in world space
      const eyePos = new THREE.Vector3();
      activeCam.getWorldPosition(eyePos);

      // Direction vector from targetPoint to user eye
      let viewDir = eyePos.clone().sub(targetPoint);
      let dist = viewDir.length();
      if (dist < 0.1) {
        viewDir.set(0, 0.4, 1.0);
        dist = 2.2;
      } else {
        viewDir.normalize();
      }

      // Optimal framing distance to capture the entire object silhouette with ample context padding
      const captureDist = Math.max(1.8, Math.min(3.2, dist));
      const snapCamPos = targetPoint.clone().addScaledVector(viewDir, captureDist);

      const snapCam = new THREE.PerspectiveCamera(52, 1.0, 0.05, 50.0);
      snapCam.position.copy(snapCamPos);
      snapCam.lookAt(targetPoint); // Mathematically locks targetPoint dead center (0,0) in viewport!
      snapCam.updateMatrixWorld(true);
      snapCam.updateProjectionMatrix();

      const prevRenderTarget = renderer.getRenderTarget();
      const prevXrEnabled = renderer.xr.enabled;

      // Temporarily disable XR projection override during off-screen render pass
      renderer.xr.enabled = false;
      renderer.setRenderTarget(snapTarget);
      renderer.clear();

      if (STATE.dropInViewer && STATE.dropInViewer.viewer) {
        try {
          STATE.dropInViewer.viewer.update(renderer, snapCam);
        } catch (e) {}
      }
      renderer.render(scene, snapCam);

      // Read rendered pixels
      const pixelBuffer = new Uint8Array(renderSize * renderSize * 4);
      renderer.readRenderTargetPixels(snapTarget, 0, 0, renderSize, renderSize, pixelBuffer);

      // Restore renderer states
      renderer.setRenderTarget(prevRenderTarget);
      renderer.xr.enabled = prevXrEnabled;
      snapTarget.dispose();

      // Restore reticle & HUD
      if (raycastTargetMesh) raycastTargetMesh.visible = wasReticleVisible;
      if (xrHudMesh) xrHudMesh.visible = wasHudVisible;

      // Transfer to 2D canvas and flip Y (WebGL bottom-up to canvas top-down)
      const snapCanvas = document.createElement('canvas');
      snapCanvas.width = renderSize;
      snapCanvas.height = renderSize;
      const ctx = snapCanvas.getContext('2d');
      const imgData = ctx.createImageData(renderSize, renderSize);

      for (let y = 0; y < renderSize; y++) {
        const srcRow = (renderSize - 1 - y) * renderSize * 4;
        const dstRow = y * renderSize * 4;
        imgData.data.set(pixelBuffer.subarray(srcRow, srcRow + renderSize * 4), dstRow);
      }
      ctx.putImageData(imgData, 0, 0);

      return snapCanvas.toDataURL('image/jpeg', 0.90);
    }

    // PATH B: Screen space or full viewport capture
    if (STATE.dropInViewer && STATE.dropInViewer.viewer) {
      try {
        STATE.dropInViewer.viewer.update(renderer, activeCam);
      } catch (e) {}
    }
    renderer.render(scene, activeCam);

    let dataUrl = null;
    if (!crop) {
      dataUrl = canvasEl.toDataURL('image/jpeg', 0.90);
    } else {
      const rect = canvasEl.getBoundingClientRect();
      const scaleX = canvasEl.width / (rect.width || 1);
      const scaleY = canvasEl.height / (rect.height || 1);

      const actualX = screenX != null ? screenX : (rect.width / 2);
      const actualY = screenY != null ? screenY : (rect.height / 2);

      const centerPixelX = Math.round(actualX * scaleX);
      const centerPixelY = Math.round(actualY * scaleY);

      const targetCropSize = Math.round(boxSize * Math.min(scaleX, scaleY));
      const cropSize = Math.max(320, Math.min(targetCropSize, canvasEl.width, canvasEl.height));

      const halfCrop = Math.floor(cropSize / 2);
      let srcX = Math.max(0, Math.min(canvasEl.width - cropSize, centerPixelX - halfCrop));
      let srcY = Math.max(0, Math.min(canvasEl.height - cropSize, centerPixelY - halfCrop));

      const cropCanvas = document.createElement('canvas');
      cropCanvas.width = cropSize;
      cropCanvas.height = cropSize;
      const ctx = cropCanvas.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#07090e';
        ctx.fillRect(0, 0, cropSize, cropSize);
        ctx.drawImage(canvasEl, srcX, srcY, cropSize, cropSize, 0, 0, cropSize, cropSize);
        dataUrl = cropCanvas.toDataURL('image/jpeg', 0.90);
      }
    }

    // Restore reticle visibility and re-render
    if (raycastTargetMesh) raycastTargetMesh.visible = wasReticleVisible;
    if (xrHudMesh) xrHudMesh.visible = wasHudVisible;

    return dataUrl;
  } catch (err) {
    console.warn('[SpatialRaycast] takeScreenshot error:', err);
    return null;
  }
}
window.takeScreenshot = takeScreenshot;

// --- 3D Spatial Object Selection (Raycasting) & Vision + Tavily Grounding Pipeline ---
function collectCandidateMeshes() {
  const candidateMeshes = [];
  if (Array.isArray(STATE.triangleMeshChunks) && STATE.triangleMeshChunks.length > 0) {
    for (const chunk of STATE.triangleMeshChunks) {
      if (chunk && chunk.isMesh && chunk.visible) candidateMeshes.push(chunk);
    }
  }
  if (STATE.splatPivotGroup) {
    STATE.splatPivotGroup.traverse((obj) => {
      if (obj.isMesh && obj.visible && !candidateMeshes.includes(obj)) {
        candidateMeshes.push(obj);
      }
    });
  }
  scene.traverse((obj) => {
    if (
      obj.isMesh &&
      obj.visible &&
      obj !== gridHelper &&
      obj !== raycastTargetMesh &&
      obj !== xrHudMesh &&
      !obj.name.includes('controller') &&
      !obj.name.includes('hud') &&
      !candidateMeshes.includes(obj)
    ) {
      candidateMeshes.push(obj);
    }
  });
  return candidateMeshes;
}

function performRaycastSelection(mouseNdc, screenX, screenY) {
  if (STATE.isRaycastQuerying) {
    showToast('⏳ Inspection in progress, please wait...', 'warning', 2000);
    return;
  }

  const GS = window.GaussianSplats3D || window['Gaussian Splats 3D'];
  let hitPoint = null;
  let minHitDist = Infinity;

  // 1. Explicit raycast against THREE.Mesh objects (Triangle Splatting chunks & scene meshes)
  const candidateMeshes = collectCandidateMeshes();
  if (candidateMeshes.length > 0) {
    try {
      const threeRaycaster = new THREE.Raycaster();
      threeRaycaster.setFromCamera(mouseNdc, camera);
      const origSides = new Map();
      for (const mesh of candidateMeshes) {
        if (mesh.material) {
          origSides.set(mesh, mesh.material.side);
          mesh.material.side = THREE.DoubleSide;
        }
      }

      const intersects = threeRaycaster.intersectObjects(candidateMeshes, false);

      for (const [mesh, side] of origSides.entries()) {
        mesh.material.side = side;
      }

      if (intersects.length > 0) {
        hitPoint = intersects[0].point.clone();
        minHitDist = camera.position.distanceTo(hitPoint);
      }
    } catch (e) {
      console.warn('[Splatris] Desktop mesh raycast note:', e.message);
    }
  }

  // 2. Raycast against Gaussian Splat Mesh if available
  if (GS && GS.Raycaster && STATE.dropInViewer && STATE.dropInViewer.splatMesh) {
    try {
      const gsRaycaster = new GS.Raycaster();
      const screenPos = new THREE.Vector2(screenX || window.innerWidth / 2, screenY || window.innerHeight / 2);
      const renderDim = new THREE.Vector2(window.innerWidth, window.innerHeight);
      gsRaycaster.setFromCameraAndScreenPosition(camera, screenPos, renderDim);
      const outHits = [];
      gsRaycaster.intersectSplatMesh(STATE.dropInViewer.splatMesh, outHits);
      if (outHits.length > 0 && outHits[0].origin) {
        const splatDist = camera.position.distanceTo(outHits[0].origin);
        if (splatDist < minHitDist) {
          hitPoint = outHits[0].origin.clone();
          minHitDist = splatDist;
        }
      }
    } catch (e) {
      console.warn('[Splatris] GS.Raycaster intersectSplatMesh note:', e.message);
    }
  }

  // 3. Fallback: Closest point on ray to target center
  if (!hitPoint) {
    const threeRaycaster = new THREE.Raycaster();
    threeRaycaster.setFromCamera(mouseNdc, camera);
    const targetCenter = controls.target || new THREE.Vector3(0, STATE.pivotTargetY || 1.2, 0);
    hitPoint = new THREE.Vector3();
    threeRaycaster.ray.closestPointToPoint(targetCenter, hitPoint);
  }

  if (hitPoint) {
    triggerSpatialObjectInspection(hitPoint, screenX, screenY);
  }
}

// --- In-World Floating WebXR HUD & Interactive Exit Button ---
let xrHudMesh = null;
let xrHudCanvas = null;
let xrHudTexture = null;

function createXRInWorldHUD() {
  if (xrHudMesh) return xrHudMesh;

  xrHudCanvas = document.createElement('canvas');
  xrHudCanvas.width = 768;
  xrHudCanvas.height = 360;

  xrHudTexture = new THREE.CanvasTexture(xrHudCanvas);
  xrHudTexture.minFilter = THREE.LinearFilter;
  xrHudTexture.magFilter = THREE.LinearFilter;
  xrHudTexture.needsUpdate = true;

  const hudGeo = new THREE.PlaneGeometry(0.88, 0.41);
  const hudMat = new THREE.MeshBasicMaterial({
    map: xrHudTexture,
    transparent: true,
    side: THREE.DoubleSide,
    depthTest: false, // Never occluded or clipped by scene geometry
    depthWrite: false,
  });
  xrHudMesh = new THREE.Mesh(hudGeo, hudMat);
  xrHudMesh.name = 'xrHudMesh';
  xrHudMesh.renderOrder = 9999;
  xrHudMesh.visible = false;
  // NOTE: Head-locked HUD: attached directly to camera on WebXR presentation (not to scene)

  updateXRHUDContent();
  return xrHudMesh;
}

function updateXRHUDContent({ title = 'SPLATRIS WEBXR', subtitle = '', status = '' } = {}) {
  if (!xrHudCanvas || !xrHudTexture || !xrHudMesh) return;

  const canvasWidth = 768;
  const paddingX = 36;
  const maxTextWidth = canvasWidth - (paddingX * 2);

  // Measure and wrap lines dynamically
  const tempCtx = xrHudCanvas.getContext('2d');
  tempCtx.font = '16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

  let lines = [];
  if (status) {
    const rawWords = status.replace(/[*_#`~[\]()]/g, ' ').replace(/\s+/g, ' ').trim().split(' ');
    let currentLine = '';
    for (const w of rawWords) {
      const testLine = currentLine ? `${currentLine} ${w}` : w;
      if (tempCtx.measureText(testLine).width <= maxTextWidth) {
        currentLine = testLine;
      } else {
        if (currentLine) lines.push(currentLine);
        currentLine = w;
      }
    }
    if (currentLine) lines.push(currentLine);
  } else {
    lines = [
      '🕹️ Left Stick: Move & Glide into Scene',
      '🔄 Right Stick: Turn & Fly Up / Down',
      '🎯 Controller Trigger: Point laser to inspect object details',
    ];
  }

  // Calculate dynamic canvas height based on wrapped lines (zero truncation)
  const lineHeight = 26;
  const topHeaderHeight = subtitle ? 96 : 72;
  const textContentHeight = Math.max(78, lines.length * lineHeight + 16);
  const exitButtonHeight = 56;
  const totalCanvasHeight = Math.max(300, Math.min(1000, topHeaderHeight + textContentHeight + exitButtonHeight + 36));

  xrHudCanvas.width = canvasWidth;
  xrHudCanvas.height = totalCanvasHeight;

  const ctx = xrHudCanvas.getContext('2d');
  ctx.clearRect(0, 0, canvasWidth, totalCanvasHeight);

  // Glassmorphic dark card
  ctx.fillStyle = 'rgba(11, 19, 38, 0.94)';
  ctx.beginPath();
  ctx.roundRect(10, 10, canvasWidth - 20, totalCanvasHeight - 20, 20);
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = '#06b6d4';
  ctx.stroke();

  // Header Title
  ctx.fillStyle = '#38bdf8';
  ctx.font = 'bold 22px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  const cleanTitle = (title || 'SPLATRIS WEBXR').slice(0, 36);
  ctx.fillText(cleanTitle, paddingX, 48);

  // Status green pulse dot
  ctx.fillStyle = '#10b981';
  ctx.beginPath();
  ctx.arc(canvasWidth - 44, 42, 8, 0, Math.PI * 2);
  ctx.fill();

  // Subtitle / category
  if (subtitle) {
    ctx.fillStyle = '#06b6d4';
    ctx.font = 'bold 15px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
    ctx.fillText(subtitle.slice(0, 60), paddingX, 78);
  }

  // Body text lines
  ctx.fillStyle = '#e2e8f0';
  ctx.font = '16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  let curY = topHeaderHeight + 14;
  for (const line of lines) {
    ctx.fillText(line, paddingX, curY);
    curY += lineHeight;
  }

  // Exit Button Badge at bottom
  const btnY = totalCanvasHeight - 64;
  ctx.fillStyle = 'rgba(239, 68, 68, 0.28)';
  ctx.beginPath();
  ctx.roundRect(paddingX, btnY, canvasWidth - (paddingX * 2), 44, 12);
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = '#ef4444';
  ctx.stroke();

  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 16px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('🚪 Click or Press [ B / Y ] to Exit WebXR', canvasWidth / 2, btnY + 28);
  ctx.textAlign = 'left';

  // Dynamic PlaneGeometry resizing to match calculated aspect ratio exactly
  const planeWidth = 0.88;
  const planeHeight = (totalCanvasHeight / canvasWidth) * planeWidth;
  if (xrHudMesh.geometry) {
    xrHudMesh.geometry.dispose();
  }
  xrHudMesh.geometry = new THREE.PlaneGeometry(planeWidth, planeHeight);
  xrHudTexture.needsUpdate = true;
}

function checkXRExitButtonHit(origin, direction) {
  if (!renderer.xr.isPresenting || !xrHudMesh || !xrHudMesh.visible) return false;
  xrHudMesh.updateMatrixWorld(true);
  const threeRaycaster = new THREE.Raycaster(origin, direction, 0.05, 10.0);
  const hits = threeRaycaster.intersectObject(xrHudMesh);
  if (hits.length > 0) {
    const uv = hits[0].uv;
    // In Three.js plane UVs, y goes from 0 at the bottom to 1 at the top.
    // The exit button is at the bottom of the card (uv.y < 0.28).
    if (uv && uv.y < 0.28) {
      console.log(`[Splatris] In-VR Exit HUD clicked (uv.y=${uv.y.toFixed(2)}): Ending WebXR session...`);
      const session = renderer.xr.getSession();
      if (session) {
        session.end();
        return true;
      }
    }
  }
  return false;
}

function performRaycastSelectionXR(origin, direction) {
  // Prevent WebXR multi-trigger spam & API pipeline crash
  if (STATE.isRaycastQuerying) {
    console.log('[Splatris] WebXR raycast ignored: inspection query already in progress');
    return;
  }

  const GS = window.GaussianSplats3D || window['Gaussian Splats 3D'];
  let hitPoint = null;
  let minHitDist = Infinity;

  // 1. Explicit Raycast against THREE.Mesh objects (Triangle Splatting chunks & scene meshes)
  const candidateMeshes = typeof collectCandidateMeshes === 'function' ? collectCandidateMeshes() : [];
  if (candidateMeshes.length > 0) {
    try {
      const threeRaycaster = new THREE.Raycaster(origin, direction, 0.05, 50.0);
      const origSides = new Map();
      for (const mesh of candidateMeshes) {
        if (mesh.material) {
          origSides.set(mesh, mesh.material.side);
          mesh.material.side = THREE.DoubleSide;
        }
      }

      const intersects = threeRaycaster.intersectObjects(candidateMeshes, false);

      for (const [mesh, side] of origSides.entries()) {
        mesh.material.side = side;
      }

      if (intersects.length > 0) {
        hitPoint = intersects[0].point.clone();
        minHitDist = intersects[0].distance;
        console.log(`[Splatris] WebXR raycast hit THREE.Mesh "${intersects[0].object.name}" at distance ${minHitDist.toFixed(2)}m`);
      }
    } catch (meshErr) {
      console.warn('[Splatris] WebXR THREE.Mesh raycaster warning:', meshErr);
    }
  }

  // 2. Precise raycast against Gaussian Splat Mesh in 3D
  if (GS && GS.Raycaster && STATE.dropInViewer && STATE.dropInViewer.splatMesh) {
    try {
      const gsRaycaster = new GS.Raycaster(origin, direction);
      const outHits = [];
      gsRaycaster.intersectSplatMesh(STATE.dropInViewer.splatMesh, outHits);
      if (outHits.length > 0 && outHits[0].origin) {
        const splatDist = origin.distanceTo(outHits[0].origin);
        if (splatDist < minHitDist) {
          hitPoint = outHits[0].origin.clone();
          minHitDist = splatDist;
          console.log(`[Splatris] WebXR raycast hit 3DGS SplatMesh at distance ${minHitDist.toFixed(2)}m`);
        }
      }
    } catch (e) {
      console.warn('[Splatris] XR GS.Raycaster note:', e.message);
    }
  }

  // 3. Fallback: 2.2 meters in front along ray
  if (!hitPoint) {
    hitPoint = origin.clone().addScaledVector(direction, 2.2);
  }

  triggerSpatialObjectInspection(hitPoint, null, null);
}

// State Lock Watchdog to prevent permanent lockup if network/TTS stalls
let raycastLockTimeout = null;
function lockRaycastQuery() {
  STATE.isRaycastQuerying = true;
  if (raycastLockTimeout) clearTimeout(raycastLockTimeout);
  raycastLockTimeout = setTimeout(() => {
    console.warn('[Splatris] Watchdog safety timeout: auto-unlocking raycast query lock');
    unlockRaycastQuery();
  }, 18000);
}

function unlockRaycastQuery() {
  if (raycastLockTimeout) {
    clearTimeout(raycastLockTimeout);
    raycastLockTimeout = null;
  }
  STATE.isRaycastQuerying = false;
  console.log('[Splatris] Raycast state lock released (ready for next inspection)');
}

async function triggerSpatialObjectInspection(hitPoint, screenX, screenY) {
  // Prevent simultaneous raycasts / duplicate API calls
  if (STATE.isRaycastQuerying) {
    console.log('[Splatris] Object inspection ignored: query already active');
    return;
  }
  lockRaycastQuery();

  initSpatialAudio();

  const activeCam = renderer.xr.isPresenting ? renderer.xr.getCamera() : camera;

  // Position 3D spatial audio emitter at the target object coordinates immediately
  updateSpatialEmitterPosition(hitPoint.x, hitPoint.y, hitPoint.z);
  triggerSpatialChime(660, 0.25);

  // Position 3D target reticle in world space
  if (raycastTargetMesh) {
    raycastTargetMesh.position.copy(hitPoint);
    raycastTargetMesh.lookAt(activeCam.position);
    raycastTargetMesh.visible = true;
  }

  // Show "Processing..." indicator in HUD & Desktop panel
  const emitterEl = document.getElementById('emitter-coords');
  if (emitterEl) {
    emitterEl.textContent = `Processing (${hitPoint.x.toFixed(2)}, ${hitPoint.y.toFixed(2)}, ${hitPoint.z.toFixed(2)})...`;
  }

  // 1. Capture dedicated target-centered snapshot (mathematically locks hitPoint at exact center)
  const imageBase64 = takeScreenshot({ targetPoint: hitPoint, crop: true, screenX, screenY, boxSize: 540 });

  showToast(`🎯 Object selected at (${hitPoint.x.toFixed(2)}, ${hitPoint.y.toFixed(2)}, ${hitPoint.z.toFixed(2)}). Querying Vision RAG...`, 'info', 3000);

  // In WebXR: Update floating in-world HUD to show active inspection
  updateXRHUDContent({
    title: 'PROCESSING...',
    subtitle: `Point: (${hitPoint.x.toFixed(2)}, ${hitPoint.y.toFixed(2)}, ${hitPoint.z.toFixed(2)})`,
    status: 'Identifying visual features via Nemotron Vision & Tavily...',
  });

  // Automatically open RAG drawer if collapsed (desktop)
  const rightPanel = document.getElementById('right-panel');
  if (rightPanel && rightPanel.classList.contains('collapsed')) {
    rightPanel.classList.remove('collapsed');
    const toggleBtn = document.getElementById('btn-toggle-right-panel');
    if (toggleBtn) toggleBtn.classList.add('active');
    if (typeof onWindowResize === 'function') onWindowResize();
  }

  // Post user query in chat
  appendChatMessage('user', `🎯 Spatial Raycast: Inspect object at [${hitPoint.x.toFixed(2)}, ${hitPoint.y.toFixed(2)}, ${hitPoint.z.toFixed(2)}]`);

  // Post loading card in chat
  const typingId = appendChatMessage('agent', '⏳ Processing: Capturing WebGL viewport crop, running Nemotron Vision identification & Tavily grounding...');

  try {
    const camPos = new THREE.Vector3();
    activeCam.getWorldPosition(camPos);

    const payload = {
      point: { x: hitPoint.x, y: hitPoint.y, z: hitPoint.z },
      activeModel: STATE.activeModelUrl,
      cameraPosition: { x: camPos.x, y: camPos.y, z: camPos.z },
      videoId: STATE.currentSceneContext.videoId,
      imageBase64,
    };

    const res = await safeFetchJson('/api/spatial/raycast-object', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.success) {
      throw new Error(res.error || 'Failed to inspect object');
    }

    const objectName = (res.objectName || '').trim();
    const isUnidentified = !objectName || objectName.toLowerCase() === 'unknown object' || (typeof res.confidence === 'number' && res.confidence < 0.35);

    if (isUnidentified) {
      const fallbackMsg = "I couldn't identify this object. Please try clicking closer or from another angle.";
      updateXRHUDContent({
        title: 'OBJECT NOT IDENTIFIED',
        subtitle: 'Low confidence detection',
        status: fallbackMsg,
      });
      updateChatMessage(typingId, `⚠️ ${fallbackMsg}`, []);
      speakSpatialAudio(fallbackMsg, hitPoint, () => {
        unlockRaycastQuery();
      });
      return;
    }

    // Persist inspected object into state for follow-up conversational RAG
    STATE.lastInspectedObject = {
      objectName: res.objectName,
      category: res.category,
      confidence: res.confidence,
      visualDescription: res.visualDescription,
      tavilyData: res.tavilyData,
      cropUrl: res.cropUrl,
      point: hitPoint,
    };

    // Update DOM written text in Desktop RAG panel
    renderRaycastResult(typingId, res);

    if (res.objectName) {
      updateQueryChipsForObject(res.objectName);
    }

    // In WebXR: Update floating HUD with object name and concise summary
    const summaryText = res.tavilyData?.summary || res.tavilyGrounding?.summary || res.visualDescription || '';
    updateXRHUDContent({
      title: (res.objectName || 'OBJECT IDENTIFIED').toUpperCase(),
      subtitle: `${res.category || 'Spatial Element'} • ${Math.round((res.confidence || 0.9) * 100)}% Match`,
      status: summaryText,
    });

    // Dynamic spatial speech narration positioned at the object
    // Automatically triggers in WebXR mode or when audio is enabled
    // Only unlock the trigger state AFTER audio finishes speaking
    if (renderer.xr.isPresenting || STATE.audioEnabled) {
      const speechText = `${res.objectName}. ${summaryText}`;
      speakSpatialAudio(speechText, hitPoint, () => {
        unlockRaycastQuery();
      });
    } else {
      unlockRaycastQuery();
    }
  } catch (err) {
    console.error('Raycast inspection error:', err);
    const fallbackMsg = "I couldn't identify this object. Please try clicking again.";
    updateChatMessage(typingId, `Target inspection error: ${err.message}`, []);
    updateXRHUDContent({
      title: 'INSPECTION ERROR',
      subtitle: 'Target inspection failed',
      status: fallbackMsg,
    });
    speakSpatialAudio(fallbackMsg, hitPoint, () => {
      unlockRaycastQuery();
    });
  }
}

function renderRaycastResult(id, result) {
  const el = document.getElementById(id);
  if (!el) return;

  const contentEl = el.querySelector('.msg-content');
  if (!contentEl) return;

  const objName = (result.objectName || 'Detected Spatial Object').trim();
  const desc = (result.visualDescription || result.description || 'Target object identified in 3D scene.').trim();
  const confidence = result.confidence ? Math.round(result.confidence * 100) : 92;
  const cropUrl = result.cropUrl || (result.originalFrame ? `/uploads/frames/${result.originalFrame}` : '');
  
  // Tavily payload mapping: support both tavilyData and tavilyGrounding
  const tavily = result.tavilyData || result.tavilyGrounding || {};
  const summary = (tavily.summary || result.summary || '').trim();
  const sources = tavily.results || tavily.sources || [];
  const displaySummary = summary || desc || `Identified ${objName} within the reconstructed 3D space.`;

  // Update Emitter Coords in the right panel DOM
  const emitterEl = document.getElementById('emitter-coords');
  if (emitterEl) {
    const p = result.point || (STATE.lastInspectedObject && STATE.lastInspectedObject.point);
    const coordsStr = p ? `(${p.x.toFixed(2)}, ${p.y.toFixed(2)}, ${p.z.toFixed(2)})` : '';
    emitterEl.textContent = `${objName} ${coordsStr}`.trim();
  }

  // Ensure right panel is visible
  const rightPanel = document.getElementById('right-panel');
  if (rightPanel && rightPanel.classList.contains('collapsed')) {
    rightPanel.classList.remove('collapsed');
    const toggleBtn = document.getElementById('btn-toggle-right-panel');
    if (toggleBtn) toggleBtn.classList.add('active');
    if (typeof onWindowResize === 'function') onWindowResize();
  }

  contentEl.innerHTML = `
    <div>Identified target via <strong>Nemotron Vision</strong>:</div>
    <div class="raycast-card">
      <div class="raycast-header">
        <div class="raycast-title">
          <span>🎯</span> ${objName}
        </div>
        <span class="raycast-badge">${confidence}% Confidence</span>
      </div>
      <div class="raycast-media-row">
        ${cropUrl ? `<img src="${cropUrl}" alt="${objName}" class="raycast-crop-thumb" title="Click to view full image" onclick="window.open('${cropUrl}', '_blank')"/>` : ''}
        <div class="raycast-body">
          <div class="raycast-desc">${desc}</div>
          <div class="raycast-meta">
            <span class="raycast-tag">Vision: Nemotron-MiniCPM-V</span>
            <span class="raycast-tag">Tavily Grounded</span>
          </div>
        </div>
      </div>
      ${displaySummary ? `
        <div class="raycast-grounding">
          <div style="font-weight: 600; color: #67e8f9; margin-bottom: 4px;">Real-World Grounding (Tavily AI):</div>
          <div class="raycast-summary-text" style="color: #e2e8f0; line-height: 1.45; font-size: 0.78rem;">${displaySummary}</div>
          ${sources.length > 0 ? `
            <div class="raycast-sources-list">
              ${sources.map(s => `
                <a href="${s.url}" target="_blank" rel="noopener noreferrer" class="raycast-source-chip">
                  <span>🔗</span> ${s.title || (s.url ? new URL(s.url).hostname : 'Source')}
                </a>
              `).join('')}
            </div>
          ` : ''}
        </div>
      ` : ''}
    </div>
  `;

  chatLog.scrollTop = chatLog.scrollHeight;
}

function updateQueryChipsForObject(objectName) {
  const chipsContainer = document.getElementById('rag-presets') || document.querySelector('.preset-chips') || document.querySelector('.rag-preset-chips');
  if (!chipsContainer || !objectName) return;

  const suggestions = [
    `Tell me about this ${objectName}`,
    `What are the key features of this ${objectName}?`,
    `Where was this ${objectName} manufactured?`,
    `Estimated value and specifications of this ${objectName}`
  ];

  chipsContainer.innerHTML = suggestions.map(q => `
    <button class="chip-query" data-query="${q}">${q}</button>
  `).join('');

  chipsContainer.querySelectorAll('.chip-query').forEach((chip) => {
    chip.addEventListener('click', () => {
      const q = chip.getAttribute('data-query');
      if (q) {
        if (ragInput) ragInput.value = q;
        if (typeof submitRagQuery === 'function') {
          submitRagQuery(q);
        }
      }
    });
  });
}

// Double-Click on 3D Viewport to inspect object
renderer.domElement.addEventListener('dblclick', (event) => {
  if (STATE.isRaycastQuerying) return;
  const rect = renderer.domElement.getBoundingClientRect();
  const mouseNdc = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
  performRaycastSelection(mouseNdc, event.clientX - rect.left, event.clientY - rect.top);
});

// Viewport Overlay: Toggle 3-Axis Free Rotation vs Grounded Orbit
const btnToggleTrackball = document.getElementById('btn-toggle-trackball');
if (btnToggleTrackball) {
  btnToggleTrackball.addEventListener('click', () => {
    const nextMode = STATE.cameraControlMode === 'trackball' ? 'orbit' : 'trackball';
    setCameraControlMode(nextMode);
    showToast(
      nextMode === 'trackball'
        ? '🌐 Switched to 3-Axis Free Rotation (Left-Click & Drag to rotate Pitch, Yaw & Roll)'
        : '🔄 Switched to Grounded Orbit (Turntable Rotation)',
      'info',
      3000
    );
  });
}

// Viewport Overlay: Inspect Object Button
const btnInspectRaycast = document.getElementById('btn-inspect-raycast');
if (btnInspectRaycast) {
  btnInspectRaycast.addEventListener('click', () => {
    if (STATE.isRaycastQuerying) {
      showToast('⏳ Inspection in progress, please wait...', 'warning', 2000);
      return;
    }
    const mouseNdc = new THREE.Vector2(0, 0);
    performRaycastSelection(mouseNdc, window.innerWidth / 2, window.innerHeight / 2);
    showToast('🎯 Inspecting centered object with Nemotron Vision & Tavily...', 'info', 2500);
  });
}

// --- Scene History Widget (Recent Spaces & Recent Objects) ---
const recentSpacesChips = document.getElementById('recent-spaces-chips');
const recentObjectsChips = document.getElementById('recent-objects-chips');
const recentSpacesCount = document.getElementById('recent-spaces-count');
const recentObjectsCount = document.getElementById('recent-objects-count');

function initRecentHistoryAccordions() {
  const spacesHeader = document.getElementById('header-recent-spaces');
  const spacesWidget = document.getElementById('section-recent-spaces');
  if (spacesHeader && spacesWidget && !spacesHeader.dataset.bound) {
    spacesHeader.dataset.bound = 'true';
    spacesHeader.addEventListener('click', () => {
      spacesWidget.classList.toggle('collapsed');
    });
  }

  const objectsHeader = document.getElementById('header-recent-objects');
  const objectsWidget = document.getElementById('section-recent-objects');
  if (objectsHeader && objectsWidget && !objectsHeader.dataset.bound) {
    objectsHeader.dataset.bound = 'true';
    objectsHeader.addEventListener('click', () => {
      objectsWidget.classList.toggle('collapsed');
    });
  }
}

function formatRelativeTime(isoString) {
  if (!isoString) return 'Recent';
  try {
    const date = new Date(isoString);
    const diffSec = Math.floor((Date.now() - date.getTime()) / 1000);
    if (diffSec < 60) return 'Just now';
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffHr = Math.floor(diffMin / 60);
    if (diffHr < 24) return `${diffHr}h ago`;
    return `${Math.floor(diffHr / 24)}d ago`;
  } catch (e) {
    return 'Recent';
  }
}

function formatDateTimeShort(isoString) {
  if (!isoString) return 'Recent';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return 'Recent';
    const month = d.toLocaleString('en-US', { month: 'short' });
    const day = d.getDate();
    const hours = String(d.getHours()).padStart(2, '0');
    const minutes = String(d.getMinutes()).padStart(2, '0');
    return `${month} ${day}, ${hours}:${minutes}`;
  } catch (e) {
    return 'Recent';
  }
}

/**
 * Clears the active GaussianSplats3D scene from Three.js and resets HUD metrics.
 */
async function clearActiveViewport() {
  if (STATE.dropInViewer) {
    if (STATE.splatPivotGroup) {
      STATE.splatPivotGroup.remove(STATE.dropInViewer);
    } else {
      scene.remove(STATE.dropInViewer);
    }
    try {
      await STATE.dropInViewer.dispose();
      console.log('[Splatris] Disposed active GaussianSplats3D instance');
    } catch (e) {
      console.warn('[Splatris] Instance disposal notice:', e);
    }
    STATE.dropInViewer = null;
    STATE.splatMesh = null;
  }
  disposeTriangleMeshChunks();
  STATE.currentSplatCount = 0;
  STATE.activeModelUrl = null;
  STATE.activeScene = null;

  // Reset HUD stats pills
  const countPill = document.getElementById('stat-splat-count');
  const unitPill = document.getElementById('stat-splat-unit');
  const modelNamePill = document.getElementById('stat-model-name');
  if (countPill) countPill.textContent = '0';
  if (unitPill) unitPill.textContent = 'Splats';
  if (modelNamePill) modelNamePill.textContent = 'No Model';

  // Hide the Fast Pipeline Switch banner
  if (reprocessBanner) {
    reprocessBanner.classList.add('hidden');
    reprocessBanner.style.setProperty('display', 'none', 'important');
  }
}

/**
 * Resets the entire UI to its clean, default upload state when an active scene is deleted.
 */
async function resetUIToDefaultState() {
  // 1. Clear 3D Viewport & HUD indicators
  await clearActiveViewport();

  // 2. Hide fast pipeline switch banner & GPU suspended banner
  if (reprocessBanner) {
    reprocessBanner.classList.add('hidden');
    reprocessBanner.style.setProperty('display', 'none', 'important');
  }
  hideGpuSuspendedBanner();

  // 3. Reset pipeline step indicators to Step 1
  setPipelineStep(1);

  // 4. Re-enable all training method radios and set default to Standard 3DGS
  const btn3dgs = document.getElementById('btn-method-3dgs');
  const btnTriangle = document.getElementById('btn-method-triangle');
  const btnImage = document.getElementById('btn-method-image2splat');
  [btn3dgs, btnTriangle, btnImage].forEach((btn) => {
    if (btn) {
      btn.classList.remove('method-disabled');
      btn.removeAttribute('disabled');
      btn.removeAttribute('title');
    }
  });
  setTrainingMethod('3dgs', false);

  // 5. Reset dropzone text & status pills
  if (dropTitle) dropTitle.textContent = 'Drop a walkthrough video or single object photo';
  if (dropSubtitle) dropSubtitle.textContent = 'Supports MP4/MOV walkthroughs (max 1GB) or JPG/PNG photos for Instant 3D scan';
  if (jobPill) {
    jobPill.className = 'badge idle';
    jobPill.textContent = 'Idle';
  }
  if (progressCard) progressCard.classList.add('hidden');
  if (nemotronCard) nemotronCard.classList.add('hidden');

  // 6. Reset state flags and clear active job/scene from localStorage
  STATE.activeModelUrl = null;
  STATE.activeScene = null;
  STATE.userWantsNewUpload = true;
  localStorage.removeItem('splatris_active_job_id');
  localStorage.removeItem('splatris_active_scene_id');

  // 7. Remove active class from all chips
  document.querySelectorAll('.scene-chip').forEach((c) => {
    c.classList.remove('active');
    const dot = c.querySelector('.status-dot');
    if (dot) dot.classList.remove('online');
  });

  // 8. Update device branding / HUD pills
  updateDeviceBrandingTag();
}

/**
 * Handles scene deletion request to the API, updates frontend arrays, and resets UI if needed.
 */
async function handleDeleteScene(m, chip) {
  const targetSceneId = m.jobId || m.sceneId || m.name;
  if (!targetSceneId) return;

  try {
    const res = await safeFetchJson(`/api/spatial/job/${encodeURIComponent(targetSceneId)}`, {
      method: 'DELETE',
    });

    if (!res || !res.success) {
      showToast(res?.error || 'Failed to delete scene.', 'error', 3500);
      if (chip) {
        const btn = chip.querySelector('.btn-chip-delete');
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="3 6 5 6 21 6"></polyline>
              <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
              <line x1="10" y1="11" x2="10" y2="17"></line>
              <line x1="14" y1="11" x2="14" y2="17"></line>
            </svg>
          `;
        }
      }
      return;
    }

    showToast('Scene deleted successfully.', 'success', 2500);

    // 1. Check if the scene being deleted is currently active/loaded in the 3D viewport
    const isActiveScene = Boolean(
      (STATE.activeScene && (STATE.activeScene.name === m.name || STATE.activeScene.jobId === m.jobId)) ||
      (STATE.activeModelUrl && (STATE.activeModelUrl === m.url || STATE.activeModelUrl.endsWith(m.filename))) ||
      chip.classList.contains('active')
    );

    // 2. Animate and remove chip from DOM
    chip.style.transition = 'all 0.25s ease';
    chip.style.opacity = '0';
    chip.style.transform = 'scale(0.92) translateX(-8px)';
    setTimeout(() => {
      chip.remove();

      // Check if sections are now empty
      if (recentSpacesChips && recentSpacesChips.querySelectorAll('.scene-chip').length === 0) {
        recentSpacesChips.innerHTML = '<span class="scenes-empty-text">No recent video spaces</span>';
      }
      if (recentObjectsChips && recentObjectsChips.querySelectorAll('.scene-chip').length === 0) {
        recentObjectsChips.innerHTML = '<span class="scenes-empty-text">No recent photo objects</span>';
      }
    }, 250);

    // 3. Dynamically update frontend arrays without reloading page
    STATE.recentVideoModels = (STATE.recentVideoModels || []).filter(
      (item) => item.name !== m.name && item.jobId !== m.jobId && item.url !== m.url
    );
    STATE.recentImageModels = (STATE.recentImageModels || []).filter(
      (item) => item.name !== m.name && item.jobId !== m.jobId && item.url !== m.url
    );
    STATE.recentModels = (STATE.recentModels || []).filter(
      (item) => item.name !== m.name && item.jobId !== m.jobId && item.url !== m.url
    );

    // Update count badges
    if (recentSpacesCount) recentSpacesCount.textContent = `${STATE.recentVideoModels.length}`;
    if (recentObjectsCount) recentObjectsCount.textContent = `${STATE.recentImageModels.length}`;

    // 4. If the active scene was deleted, clear viewport & reset UI to default state
    if (isActiveScene) {
      await resetUIToDefaultState();
    }
  } catch (err) {
    console.error('[Splatris] Delete error:', err);
    showToast(`Error deleting scene: ${err.message}`, 'error', 3500);
    if (chip) {
      const btn = chip.querySelector('.btn-chip-delete');
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="3 6 5 6 21 6"></polyline>
            <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            <line x1="10" y1="11" x2="10" y2="17"></line>
            <line x1="14" y1="11" x2="14" y2="17"></line>
          </svg>
        `;
      }
    }
  }
}

/**
 * Creates a DOM chip element for a recent space or object model with hover delete capability.
 */
function createRecentSceneChip(m, isCurrentActive) {
  const chip = document.createElement('div');
  chip.className = `scene-chip ${isCurrentActive ? 'active' : ''}`;
  chip.dataset.url = m.url;

  let displayTitle = '';
  if (m.originalFilename && m.timestamp) {
    const dateTimeStr = formatDateTimeShort(m.timestamp);
    displayTitle = `${dateTimeStr} - ${m.originalFilename}`;
  } else if (m.title) {
    displayTitle = m.title;
  } else if (m.isDefault) {
    displayTitle = 'Default Space';
  } else {
    displayTitle = `Scan ${m.name.slice(0, 8)}`;
  }

  let displaySubtitle = m.subtitle;
  if (!displaySubtitle) {
    const devName = m.deviceName || (m.isImage ? 'Instant Object' : 'Camera');
    const splatCountStr = typeof m.splatCount === 'number' ? m.splatCount.toLocaleString() : (m.splatCount || '0');
    displaySubtitle = `${devName} • ${splatCountStr} splats`;
  }

  const method = m.trainingMethod || (m.format === 'ply' ? 'triangle' : (m.isImage ? 'image2splat' : '3dgs'));
  let methodBadgeClass = 'badge-3dgs';
  let methodBadgeText = 'Standard 3DGS';

  if (method === 'triangle') {
    methodBadgeClass = 'badge-triangle';
    methodBadgeText = 'Triangle Mesh';
  } else if (method === 'image2splat' || m.isImage) {
    methodBadgeClass = 'badge-image';
    methodBadgeText = 'Instant Object';
  }

  // Delete button
  const deleteBtnHtml = `
    <button type="button" class="btn-chip-delete" title="Delete scene and free storage" aria-label="Delete scene">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="3 6 5 6 21 6"></polyline>
        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        <line x1="10" y1="11" x2="10" y2="17"></line>
        <line x1="14" y1="11" x2="14" y2="17"></line>
      </svg>
    </button>
  `;

  chip.innerHTML = `
    <div class="scene-chip-info">
      <div class="scene-chip-title" title="${escapeHtml(displayTitle)}">
        <span class="status-dot ${isCurrentActive ? 'online' : ''}" style="width:6px;height:6px;flex-shrink:0;"></span>
        <span class="scene-chip-name">${escapeHtml(displayTitle)}</span>
      </div>
      <div class="scene-chip-meta" title="${escapeHtml(displaySubtitle)}">${escapeHtml(displaySubtitle)}</div>
    </div>
    <div class="scene-chip-tags">
      <div class="scene-chip-top-row">
        <span class="badge-pipeline-tag ${methodBadgeClass}">${methodBadgeText}</span>
        ${deleteBtnHtml}
      </div>
      <span class="scene-chip-badge">${m.sizeFormatted || ''}</span>
    </div>
  `;

  // Attach delete button click listener
  const deleteBtn = chip.querySelector('.btn-chip-delete');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', async (e) => {
      e.stopPropagation();
      e.preventDefault();

      const confirmed = window.confirm(`Are you sure you want to delete this scene?\n\n"${displayTitle}"\n\nThis will permanently delete the 3D model, original upload, and cached training files from storage.`);
      if (!confirmed) return;

      deleteBtn.disabled = true;
      deleteBtn.style.opacity = '1';
      deleteBtn.innerHTML = `
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin-icon">
          <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
        </svg>
      `;

      await handleDeleteScene(m, chip);
    });
  }

  chip.addEventListener('click', async () => {
    if (STATE.activeModelUrl === m.url && STATE.dropInViewer && STATE.currentSplatCount > 0) {
      showToast('This model is already loaded and active.', 'info', 2500);
      return;
    }
    // Remove active state from all chips across both lists
    document.querySelectorAll('.scene-chip').forEach((c) => c.classList.remove('active'));
    chip.classList.add('active');
    STATE.activeModelUrl = m.url;
    STATE.activeScene = m;
    STATE.userWantsNewUpload = false;

    // Synchronize radio selection to match this model's pipeline
    const activeMethod = m.trainingMethod || (m.format === 'ply' ? 'triangle' : (m.isImage ? 'image2splat' : '3dgs'));
    setTrainingMethod(activeMethod, false);
    updatePipelineUIForActiveScene();

    // Check if there is an active job running for this scene
    const reconnected = await checkAndReconnectActiveJob(m.jobId || m.sceneId || m.name);
    if (!reconnected) {
      await loadGaussianSplatScene(m.url, true);
    }
  });

  return chip;
}

async function loadRecentScenes() {
  initRecentHistoryAccordions();

  try {
    const data = await safeFetchJson('/api/models/recent');
    if (!data.success) {
      if (recentSpacesChips) recentSpacesChips.innerHTML = '<span class="scenes-empty-text">Spaces unavailable</span>';
      if (recentObjectsChips) recentObjectsChips.innerHTML = '<span class="scenes-empty-text">Objects unavailable</span>';
      return;
    }

    const videoModels = (data.videoModels || []).slice(0, 3);
    const imageModels = (data.imageModels || []).slice(0, 3);
    const combinedModels = data.models || [...videoModels, ...imageModels];

    STATE.recentModels = combinedModels;
    STATE.recentVideoModels = videoModels;
    STATE.recentImageModels = imageModels;

    if (recentSpacesCount) recentSpacesCount.textContent = `${videoModels.length}`;
    if (recentObjectsCount) recentObjectsCount.textContent = `${imageModels.length}`;

    if (recentSpacesChips) recentSpacesChips.innerHTML = '';
    if (recentObjectsChips) recentObjectsChips.innerHTML = '';

    let matchedActive = false;
    let activeChipElement = null;

    // 1. Populate Recent Spaces (Video)
    if (videoModels.length === 0) {
      if (recentSpacesChips) {
        recentSpacesChips.innerHTML = '<span class="scenes-empty-text">No recent video spaces</span>';
      }
    } else {
      videoModels.forEach((m, idx) => {
        const isCurrentActive = STATE.activeModelUrl && (
          STATE.activeModelUrl === m.url ||
          STATE.activeModelUrl.endsWith(m.filename) ||
          (idx === 0 && !matchedActive && STATE.activeModelUrl.includes('scene.splat')) ||
          (m.isDefault && STATE.activeModelUrl.includes('scene.splat'))
        );

        if (isCurrentActive && !matchedActive) {
          matchedActive = true;
          STATE.activeScene = m;
          const method = m.trainingMethod || (m.format === 'ply' ? 'triangle' : (m.isImage ? 'image2splat' : '3dgs'));
          setTrainingMethod(method, false);
          updatePipelineUIForActiveScene();
        }

        const chip = createRecentSceneChip(m, isCurrentActive);
        if (isCurrentActive) activeChipElement = chip;
        if (recentSpacesChips) recentSpacesChips.appendChild(chip);
      });
    }

    // 2. Populate Recent Objects (Photo)
    if (imageModels.length === 0) {
      if (recentObjectsChips) {
        recentObjectsChips.innerHTML = '<span class="scenes-empty-text">No recent photo objects</span>';
      }
    } else {
      imageModels.forEach((m) => {
        const isCurrentActive = STATE.activeModelUrl && (
          STATE.activeModelUrl === m.url ||
          STATE.activeModelUrl.endsWith(m.filename)
        );

        if (isCurrentActive && !matchedActive) {
          matchedActive = true;
          STATE.activeScene = m;
          const method = m.trainingMethod || (m.format === 'ply' ? 'triangle' : (m.isImage ? 'image2splat' : '3dgs'));
          setTrainingMethod(method, false);
          updatePipelineUIForActiveScene();
        }

        const chip = createRecentSceneChip(m, isCurrentActive);
        if (isCurrentActive) activeChipElement = chip;
        if (recentObjectsChips) recentObjectsChips.appendChild(chip);
      });
    }

    // 3. Fallback selection if no model was matched active yet
    if (!matchedActive) {
      const fallbackModel = videoModels[0] || imageModels[0];
      if (fallbackModel) {
        STATE.activeScene = fallbackModel;
        const activeMethod = fallbackModel.trainingMethod || (fallbackModel.format === 'ply' ? 'triangle' : (fallbackModel.isImage ? 'image2splat' : '3dgs'));
        setTrainingMethod(activeMethod, false);
        updatePipelineUIForActiveScene();

        // Highlight fallback chip in whichever container it belongs to
        const allChips = document.querySelectorAll('.scene-chip');
        if (allChips.length > 0) {
          allChips[0].classList.add('active');
          const dot = allChips[0].querySelector('.status-dot');
          if (dot) dot.classList.add('online');
        }
      }
    }
  } catch (err) {
    console.warn('[Splatris] Could not load recent scenes:', err);
    if (recentSpacesChips) recentSpacesChips.innerHTML = '<span class="scenes-empty-text">Spaces unavailable</span>';
    if (recentObjectsChips) recentObjectsChips.innerHTML = '<span class="scenes-empty-text">Objects unavailable</span>';
  }
}

/**
 * Checks for any active or suspended training jobs for reconnection.
 * Re-attaches progress bar, timer, and polling automatically on load or scene select.
 */
async function checkAndReconnectActiveJob(targetSceneId = null) {
  const storedJobId = localStorage.getItem('splatris_active_job_id');
  const storedSceneId = targetSceneId || localStorage.getItem('splatris_active_scene_id') || STATE.activeScene?.jobId;
  const lookupId = storedJobId || storedSceneId;

  if (!lookupId) return false;

  try {
    const res = await safeFetchJson(`/api/spatial/job/status/${lookupId}`);
    if (!res || !res.success) return false;

    if (res.status === 'processing' || res.status === 'suspended') {
      console.log(`[Splatris] Reconnecting to active job ${res.jobId} (${res.status}, ${res.progress}%)...`);
      STATE.activeJobId = res.jobId;
      localStorage.setItem('splatris_active_job_id', res.jobId);
      if (res.sceneId) localStorage.setItem('splatris_active_scene_id', res.sceneId);

      // Re-attach UI
      setPipelineStep(4, { bypassColmap: true });
      const step4Text = document.querySelector('#step-4 span:last-child');
      if (step4Text) step4Text.textContent = res.trainingMethod === 'triangle' ? '4. Mesh Train' : '4. 3DGS Train';

      const progressCard = document.getElementById('job-progress-card');
      const progressStage = document.getElementById('progress-stage-text');
      const progressPct = document.getElementById('progress-pct');
      const progressBar = document.getElementById('progress-bar');
      const progressJobId = document.getElementById('progress-job-id');

      if (progressCard) progressCard.classList.remove('hidden');
      if (progressJobId) progressJobId.textContent = `job: ${res.jobId.slice(0, 8)}...`;
      const currentPct = Math.max(20, res.progress || 20);
      if (progressBar) progressBar.style.width = `${currentPct}%`;
      if (progressPct) progressPct.textContent = `${currentPct}%`;
      if (progressStage) progressStage.textContent = res.stage || 'Optimizing 3D scene on GPU...';

      if (btnAbortJob) {
        btnAbortJob.classList.remove('hidden');
        btnAbortJob.disabled = false;
      }

      updateButtonsForActiveJob(true);

      const uploadData = {
        videoId: res.sceneId || res.jobId,
        originalName: STATE.activeScene?.originalFilename || 'Walkthrough.mp4',
        metadata: { device: STATE.activeScene?.deviceName || 'Mobile Camera' },
        totalFrames: 10000,
      };

      if (res.status === 'suspended') {
        if (jobPill) {
          jobPill.className = 'badge warning';
          jobPill.textContent = 'GPU Offline';
        }
        showGpuSuspendedBanner(res.error || 'Nebius GPU worker is currently offline. Progress is saved.');
      } else {
        if (jobPill) {
          jobPill.className = 'badge processing';
          jobPill.textContent = 'Training';
        }
        hideGpuSuspendedBanner();
        pollJobUntilReady(res.jobId, uploadData, { colmapMatcher: 'vocab_tree' });
      }

      showToast(`Re-attached to active training (${currentPct}%)`, 'info', 3000);
      return true;
    } else {
      localStorage.removeItem('splatris_active_job_id');
      localStorage.removeItem('splatris_active_scene_id');
      updateButtonsForActiveJob(false);
    }
  } catch (err) {
    console.warn('[Splatris] Could not check active job:', err.message);
  }
  return false;
}

// --- Dynamic Hardware / Device Detection for Footer Branding ---
function updateDeviceBrandingTag() {
  const tag = document.getElementById('branding-tag');
  if (!tag) return;

  const ua = navigator.userAgent || '';
  const platform = navigator.platform || '';
  const isMac = platform.toUpperCase().includes('MAC') || ua.includes('Macintosh');

  let gpuRenderer = '';
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (gl) {
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        gpuRenderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL) || '';
      }
    }
  } catch (e) {}

  if (gpuRenderer.includes('Apple M') || gpuRenderer.includes('Apple Silicon')) {
    tag.textContent = 'Apple Silicon Native • Splatris Engine';
  } else if (isMac && (gpuRenderer.includes('Intel') || ua.includes('Intel'))) {
    tag.textContent = 'MacBook Pro (Intel) • Splatris Engine';
  } else if (isMac) {
    tag.textContent = 'macOS Native • Splatris Engine';
  } else if (ua.includes('Oculus') || ua.includes('Quest')) {
    tag.textContent = 'Meta Quest 3 Native • Splatris WebXR';
  } else if (/Android/i.test(ua)) {
    tag.textContent = 'Mobile WebXR • Splatris Engine';
  } else {
    tag.textContent = 'Hardware Accelerated • Splatris Engine';
  }
}

// --- Initial Scene Load & Boot ---
updateDeviceBrandingTag();
loadRecentScenes().then(async () => {
  checkAndReconnectActiveJob();
  const targetUrl = STATE.activeScene?.url || STATE.activeModelUrl;
  await loadGaussianSplatScene(targetUrl);
});
animate();

