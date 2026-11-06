"""Compare C++ rule masks/metrics against the retained Python reference on Jetson."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
reference = ROOT/'tools/python_reference'
if not reference.exists():
    reference = ROOT
sys.path.insert(0, str(reference))
from quality import analyze_one, make_predictor
from quality_core import read_image

cv2.setNumThreads(1)
cfg = yaml.safe_load((ROOT/'config/quality.yaml').read_text())
cfg['method'] = 'baseline'
out = ROOT/'outputs/cpp_validation'
out.mkdir(parents=True, exist_ok=True)
images = sorted((ROOT/'data/images').glob('sample*.jpg'))
assert len(images) == 4, 'Expected original sample01-04 images'
binary = ROOT/'build/grinding_quality'
comparisons = []
for side in (640, 1280):
    config = copy.deepcopy(cfg)
    config['max_side'] = side
    config_path = out/f'config_{side}.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True))
    cpp_out = out/f'cpp_{side}'
    subprocess.run([str(binary), 'analyze', '--config', str(config_path), '--input', str(ROOT/'data/images'),
                    '--output', str(cpp_out)], check=True)
    for path in images:
        py = analyze_one(read_image(path), path.name, config, make_predictor(config))
        cpp = json.loads((cpp_out/f'{path.stem}.json').read_text())
        mask = cv2.imread(str(cpp_out/f'{path.stem}_mask.png'), cv2.IMREAD_UNCHANGED)
        mismatch = float(np.mean(mask != py[2]))
        assert mismatch <= 0.001, (path.name, side, mismatch)
        assert cpp['status'] == py[0]['status']
        assert cpp['working_size_wh'] == py[0]['working_size_wh']
        assert len(cpp['cells']) == len(py[0]['cells'])
        keys = ['evaluable_fraction', 'unknown_fraction', 'residual_fraction_roi_lower_bound',
                'residual_fraction_evaluable']
        errors = {k: abs(cpp[k]-py[0][k]) for k in keys if cpp[k] is not None and py[0][k] is not None}
        assert max(errors.values(), default=0) <= 0.001
        for a, b in zip(cpp['cells'], py[0]['cells']):
            assert a['box_xyxy_working'] == b['box_xyxy_working']
            assert a['status'] == b['status']
        cv2.imwrite(str(out/f'{path.stem}_python_{side}_mask.png'), py[2])
        comparisons.append({'image': path.name, 'max_side': side, 'mask_mismatch_fraction': mismatch,
                            'metric_abs_errors': errors, 'cpp_analysis_ms': cpp['analysis_ms'],
                            'python_analysis_ms': py[0]['analysis_ms'], 'status': cpp['status']})

# Persistent C++ worker: live ROI override, repeated jobs, recoverable image error.
ipc = out/'worker_ipc'
ipc.mkdir(exist_ok=True)
(ipc/'quality.ready.json').unlink(missing_ok=True)
settings = {'config': str(ROOT/'config/quality.yaml'), 'method': 'baseline', 'max_side': 640,
            'roi': [0, 0, 1, 1], 'ipc': str(ipc)}
settings_path = out/'worker_settings.json'
settings_path.write_text(json.dumps(settings))
worker = subprocess.Popen([str(binary), 'worker', '--worker-config', str(settings_path)], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
worker_timings = []
try:
    deadline = time.monotonic()+10
    while not (ipc/'quality.ready.json').exists():
        assert worker.poll() is None and time.monotonic() < deadline
        time.sleep(0.01)
    for i, path in enumerate(images * 2):
        job = {'id': f'{i:032x}', 'image': str(path), 'output': str(out/f'worker_{i}'),
               'capture': {'frame_id': i, 'timestamp_unix_s': time.time()}, 'clock_source': 'tracking_process'}
        worker.stdin.write(json.dumps(job)+'\n')
        worker.stdin.flush()
        done = json.loads(worker.stdout.readline())
        assert done['ok'] and done['backend'] == 'cpp_opencv'
        config = copy.deepcopy(cfg)
        config['max_side'] = 640
        config['roi'] = {'default': [0,0,1,1], 'overrides': {}, 'exclude_rects': {}}
        original = read_image(path)
        reference_result = analyze_one(original, 'camera.jpg', config, make_predictor(config))
        cpp_mask = cv2.imread(str(out/f'worker_{i}/camera_mask.png'), cv2.IMREAD_UNCHANGED)
        assert float(np.mean(cpp_mask != reference_result[2])) <= 0.001
        worker_timings.append({'cpp_analysis_ms': done['analysis_ms'], 'cpp_total_ms': done['total_ms'],
                               'python_analysis_ms': reference_result[0]['analysis_ms']})
    job.update(id='f'*32, image=str(out/'missing.jpg'))
    worker.stdin.write(json.dumps(job)+'\n')
    worker.stdin.flush()
    assert json.loads(worker.stdout.readline())['ok'] is False
    worker.stdin.close()
    assert worker.wait(timeout=10) == 0
finally:
    if worker.poll() is None:
        worker.kill()
        worker.wait()

# Empty effective ROI, dark/overexposed frames, null denominators and millimetres.
edge_data = out/'edge_data'
edge_data.mkdir(exist_ok=True)
for name, value in [('black',0), ('white',255), ('gray',128)]:
    cv2.imwrite(str(edge_data/f'{name}.png'), np.full((64,96,3), value, np.uint8))
edge_cfg = copy.deepcopy(cfg)
edge_cfg['roi'] = {'default': [0,0,1,1], 'overrides': {}, 'exclude_rects': {'gray.png': [[0,0,1,1]]}}
edge_cfg['assessment']['mm_per_pixel_original'] = 0.1
edge_path = out/'edge_config.yaml'
edge_path.write_text(yaml.safe_dump(edge_cfg))
subprocess.run([str(binary), 'analyze', '--config', str(edge_path), '--input', str(edge_data),
                '--output', str(out/'edge_results')], check=True)
for name in ('black','white','gray'):
    path = edge_data/f'{name}.png'
    py = analyze_one(read_image(path), path.name, edge_cfg, make_predictor(edge_cfg))
    cpp = json.loads((out/f'edge_results/{name}.json').read_text())
    assert cpp['status'] == py[0]['status']
    assert cpp['unknown_fraction'] == py[0]['unknown_fraction']
    assert cpp['residual_fraction_evaluable'] == py[0]['residual_fraction_evaluable']
    assert np.array_equal(cv2.imread(str(out/f'edge_results/{name}_mask.png'),0),py[2])

report = {'comparisons': comparisons, 'worker_timings': worker_timings,
          'max_allowed_mask_mismatch_fraction': 0.001,
          'worker_bad_image_handling': 'passed', 'edge_cases': 'passed',
          'python_opencv': cv2.__version__,
          'warmed_cpp_analysis_ms_median': float(np.median([x['cpp_analysis_ms'] for x in worker_timings[4:]])),
          'warmed_python_analysis_ms_median': float(np.median([x['python_analysis_ms'] for x in worker_timings[4:]]))}
(out/'comparison.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
