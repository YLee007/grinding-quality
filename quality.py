"""CLI for offline inspection, camera preview, annotation, training and export."""
import argparse
import csv
import html
import json
import time
from pathlib import Path
import cv2
import numpy as np
from quality_core import (load_config, resolve, image_paths, read_image, write_image,
                          resize_image, roi_mask, reject_unobservable, measure, visualize,
                          baseline)


class BaselinePredictor:
    """Rule-only predictor that keeps the Jetson baseline free of ML dependencies."""

    def __init__(self, cfg):
        self.cfg = cfg

    def predict(self, image):
        return baseline(image, self.cfg)


def make_predictor(cfg):
    if cfg['method'] == 'baseline':
        return BaselinePredictor(cfg)
    # Import model dependencies only when an ML backend is explicitly selected.
    from quality_models import Predictor
    return Predictor(cfg)


def analyze_one(original, name, cfg, predictor):
    start = time.perf_counter()
    image = resize_image(original, cfg['max_side'])
    roi = roi_mask(image, name, cfg)
    mask, evidence = predictor.predict(image)
    mask = reject_unobservable(mask, image, roi, cfg)
    stats = measure(mask, original.shape, cfg)
    stats.update(image=name, method=cfg['method'], analysis_ms=1000*(time.perf_counter()-start),
                 decision_is_provisional=True, acceptance_thresholds_are_examples=True,
                 roi_purpose='exploratory plate inspection; confirm production footprint before use',
                 confidence_kind='heuristic evidence, NOT calibrated probability' if cfg['method']=='baseline'
                 else 'model output score, NOT calibrated probability')
    overlay, heat = visualize(image, mask, stats)
    return stats, image, mask, evidence, overlay, heat


def save_result(out, name, result):
    stats, image, mask, evidence, overlay, heat = result
    stats['image'] = name
    prefix = Path(name).stem
    write_image(out / f'{prefix}_original.jpg', image)
    write_image(out / f'{prefix}_overlay.jpg', overlay)
    write_image(out / f'{prefix}_heatmap.jpg', heat)
    write_image(out / f'{prefix}_mask.png', mask)
    write_image(out / f'{prefix}_evidence.png', np.uint8(np.clip(evidence, 0, 1)*255))
    (out / f'{prefix}.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats


def make_report(out, results, cfg):
    def percent(value):
        return '-' if value is None else f'{100*value:.1f}%'
    cards = []
    status_names = {'rework_candidate': '存在疑似待补磨区域', 'review': '需要人工复核', 'clean_candidate': '外观清洁候选'}
    for r in results:
        prefix = html.escape(Path(r['image']).stem)
        figures = ''.join(f'<figure><a href="{prefix}_{key}.jpg"><img src="{prefix}_{key}.jpg"></a><figcaption>{label}</figcaption></figure>'
                          for key, label in [('original','原图（缩略）'), ('overlay','区域分割'), ('heatmap','局部质量热力图')])
        cards.append(f'''<article><h2>{html.escape(r['image'])}<span>{status_names[r['status']]}</span></h2>
        <div class="metrics"><b>疑似残留 / ROI：{percent(r['residual_fraction_roi_lower_bound'])}</b>
        <b>可评价：{percent(r['evaluable_fraction'])}</b><b>待复核：{percent(r['unknown_fraction'])}</b>
        <b>最大残留连通域：{r['largest_residual_area_px_original']:.0f} px²（原图尺度）</b></div>
        <div class="images">{figures}</div><p><a href="{prefix}.json">查看逐网格指标及原图像素坐标 JSON</a> ·
        <a href="{prefix}_mask.png">类别掩膜</a> · 耗时 {r['analysis_ms']:.0f} ms（本机，含特征与统计，不含保存）</p></article>''')
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>打磨质量评估 · 初步验证</title>
    <style>body{margin:0;background:#f0f3f7;color:#203047;font:16px/1.7 'Microsoft YaHei',sans-serif}
    main{max-width:1500px;margin:40px auto;padding:0 24px}h1{font-size:30px;margin-bottom:8px}h2{font-size:20px}
    .note,article{background:white;padding:24px;border-radius:12px;margin:24px 0;box-shadow:0 2px 14px #2030470b}
    h2 span{font-size:14px;margin-left:24px;color:#9d5514}.metrics{display:flex;gap:24px;flex-wrap:wrap;font-size:14px}
    .images{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:20px}figure{margin:0}img{width:100%;max-height:680px;object-fit:contain;background:#151d28}
    figcaption{text-align:center;color:#607080}a{color:#2264a6}.green{color:#369632}.red{color:#d14235}.yellow{color:#a77912}
    @media(max-width:850px){.images{grid-template-columns:1fr}}</style><main>
    <h1>打磨质量评估 · 初步验证</h1><div class="note">'''
    body += f'<b>方法：{html.escape(cfg["method"])}</b>。本报告是算法候选判断，尚未经过人工标注精度验证，不是验收结论。'
    body += '''检测范围为示例钢板内部区域，尚未确认实际工艺要求的打磨范围。未校准几何尺寸，坐标均为原图像素，不能直接发送给机器人回位。<br>
    <span class="green">■ 裸露金属候选</span>　<span class="red">■ 残锈 / 深色残留候选</span>　
    <span class="yellow">■ 不确定 / 待复核</span>　灰暗：检测范围外。<br>
    阈值为示例；阴影、表面方向和反光仍可导致错误。可评价面积不足时，不能仅凭残留率低判为清洁。
    下方“残留 / ROI”是已识别残留占整个检测范围的比例；不确定区域未当作干净计入。
    掩膜编码：0 金属、1 残留、2 不确定、255 范围外。所有原始类别掩膜均为分析分辨率。</div>'''
    body += ''.join(cards)+'</main></html>'
    (out/'report.html').write_text(body, encoding='utf-8')
    fields = ['image', 'method', 'status', 'evaluable_fraction', 'unknown_fraction',
              'residual_fraction_evaluable', 'residual_fraction_roi_lower_bound',
              'largest_residual_area_px_original', 'analysis_ms']
    with (out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    (out/'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')


def analyze(cfg):
    paths = image_paths(cfg['input'])
    if not paths:
        raise ValueError('No input images found')
    if len({p.stem.lower() for p in paths}) != len(paths):
        raise ValueError('Input images must have unique stems to avoid overwriting report files')
    predictor = make_predictor(cfg)
    out = resolve(cfg['output'])
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        result = analyze_one(read_image(path), path.name, cfg, predictor)
        results.append(save_result(out, path.name, result))
        r = result[0]
        print(f'{path.name}: {r["status"]}, residual/ROI={r["residual_fraction_roi_lower_bound"]:.1%}, '
              f'unknown={r["unknown_fraction"]:.1%}, {r["analysis_ms"]:.0f} ms', flush=True)
    make_report(out, results, cfg)
    print(f'Report: {out / "report.html"}')


def camera(cfg):
    c = cfg['camera']
    if c['analyze_every_n_frames'] < 1 or c['save_every_n_results'] < 1:
        raise ValueError('camera sampling intervals must be >= 1')
    cap = cv2.VideoCapture(c['id'])
    if not cap.isOpened():
        raise RuntimeError('Camera could not be opened')
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c['height'])
    predictor = make_predictor(cfg)
    out = resolve(cfg['output']) / time.strftime('camera_%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    print('Camera preview: configure fixed exposure/lighting using your camera controls. Ctrl+C to stop.')
    print('Pixel coordinates only. No robot commands are sent.')
    frame_id, count = 0, 0
    saved = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Camera frame acquisition failed')
            frame_id += 1
            if frame_id % c['analyze_every_n_frames']:
                continue
            captured_ns = time.time_ns()
            result = analyze_one(frame, 'camera', cfg, predictor)
            count += 1
            result[0]['capture_host_time_ns'] = captured_ns
            print(json.dumps({k: result[0][k] for k in ['status','evaluable_fraction',
                             'residual_fraction_roi_lower_bound','analysis_ms','capture_host_time_ns']}), flush=True)
            if (count-1) % c['save_every_n_results'] == 0:
                saved.append(save_result(out, f'frame_{frame_id:07d}.jpg', result))
            if cfg.get('show_window'):
                cv2.imshow('Grinding quality (provisional)', result[4])
                if cv2.waitKey(1) & 255 in (27, ord('q')):
                    break
            if c['max_results'] > 0 and count >= c['max_results']:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        make_report(out, saved, cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['analyze','camera','annotate','train-rf','train-unet','export','self-test'])
    parser.add_argument('--config', default='config/quality.yaml')
    parser.add_argument('--input', help='Override file/directory input')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.input:
        cfg['input'] = args.input
    if args.task == 'analyze':
        analyze(cfg)
    elif args.task == 'camera':
        camera(cfg)
    elif args.task == 'annotate':
        from quality_annotate import annotate
        annotate(cfg)
    elif args.task == 'train-rf':
        from quality_models import train_rf
        train_rf(cfg)
    elif args.task == 'train-unet':
        from quality_models import train_unet
        train_unet(cfg)
    elif args.task == 'export':
        from quality_models import export_onnx
        export_onnx(cfg)
    else:
        from quality_selftest import run_tests
        run_tests(cfg)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        raise SystemExit(f'ERROR: {exc}')
