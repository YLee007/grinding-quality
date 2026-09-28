"""Image-space quality analysis. No robot commands and no automatic acceptance."""
from pathlib import Path
import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
METAL, RESIDUE, UNKNOWN, OUTSIDE = 0, 1, 2, 255
COLORS = np.array([[80, 185, 60], [55, 65, 235], [30, 200, 245]], np.uint8)
FEATURE_VERSION = 'lab-hsv-texture-v1'


def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def load_config(path):
    config = yaml.safe_load(resolve(path).read_text(encoding='utf-8'))
    if config['unet'].get('encoder') != 'mobilenet_v2':
        raise ValueError('This project currently supports unet.encoder: mobilenet_v2 only')
    if int(config['max_side']) < 64:
        raise ValueError('max_side must be >= 64')
    a = config['assessment']
    if int(a['grid_px']) < 1:
        raise ValueError('grid_px must be positive')
    for key in ('max_residual_fraction', 'min_evaluable_fraction'):
        if not 0 <= a[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    return config


def read_image(path):
    im = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f'Cannot read image: {path}')
    return im


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


def image_paths(value):
    p = resolve(value)
    paths = [p] if p.is_file() else sorted(p.glob('*'))
    return [x for x in paths if x.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}]


def resize_image(image, max_side):
    scale = min(1.0, float(max_side) / max(image.shape[:2]))
    if scale == 1:
        return image.copy()
    return cv2.resize(image, (round(image.shape[1] * scale), round(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)


def rectangle_pixels(rect, width, height):
    if len(rect) != 4 or not all(0 <= v <= 1 for v in rect):
        raise ValueError(f'ROI must be normalized [x0,y0,x1,y1]: {rect}')
    x0, y0, x1, y1 = rect
    if x0 >= x1 or y0 >= y1:
        raise ValueError('ROI has no area')
    return (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))


def roi_mask(image, name, cfg):
    h, w = image.shape[:2]
    r = cfg['roi']
    rect = r.get('overrides', {}).get(name, r['default'])
    x0, y0, x1, y1 = rectangle_pixels(rect, w, h)
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 1
    for excluded in r.get('exclude_rects', {}).get(name, []):
        x0, y0, x1, y1 = rectangle_pixels(excluded, w, h)
        mask[y0:y1, x0:x1] = 0
    return mask.astype(bool)


def features(image):
    """Color + multiscale texture; identically used for RF training and inference."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    mean = cv2.GaussianBlur(gray, (0, 0), 2)
    std = np.sqrt(np.maximum(cv2.GaussianBlur(gray * gray, (0, 0), 2) - mean * mean, 0))
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    xx, yy, xy = [cv2.GaussianBlur(v, (0, 0), 3) for v in (dx*dx, dy*dy, dx*dy)]
    coherence = np.sqrt((xx-yy)**2 + 4*xy*xy) / (xx+yy+1e-6)
    chroma = np.linalg.norm(lab[:, :, 1:3] - 128, axis=2)
    angles = hsv[:, :, 0] * (2*np.pi/180)
    x = np.dstack((lab/255, hsv[:, :, 1:]/255, np.sin(angles), np.cos(angles),
                   mean, std, dx, dy, coherence, chroma/128)).astype(np.float32)
    return x, lab, hsv, coherence


def ramp(x, low, high):
    return np.clip((x-low)/(high-low), 0, 1)


def baseline(image, cfg):
    _, lab, hsv, coherence = features(image)
    b = cfg['baseline']
    light = lab[:, :, 0]
    chroma = np.linalg.norm(lab[:, :, 1:] - 128, axis=2)
    bright = ramp(light, b['metal_l_min']-35, b['metal_l_min']+35)
    neutral = 1-ramp(chroma, b['metal_chroma_max']-7, b['metal_chroma_max']+10)
    wt = b['texture_weight']
    metal = bright * neutral * ((1-wt)+wt*coherence)
    warm = ((hsv[:, :, 0] >= b['rust_h_min']) & (hsv[:, :, 0] <= b['rust_h_max']))
    rust = warm * ramp(hsv[:, :, 1], b['rust_s_min']-15, b['rust_s_min']+35)
    dark = 1-ramp(light, b['dark_l_max']-30, b['dark_l_max']+30)
    residue = np.maximum(rust, 0.85*dark)
    evidence = np.maximum(metal, residue)
    mask = np.where(metal >= residue, METAL, RESIDUE).astype(np.uint8)
    mask[(evidence < b['min_evidence']) | (np.abs(metal-residue) < b['min_margin'])] = UNKNOWN
    return mask, evidence.astype(np.float32)


def reject_unobservable(mask, image, roi, cfg):
    result = mask.copy()
    g = cfg['image_gate']
    clipped = (image.min(axis=2) >= g['overexposed_min']) | (image.max(axis=2) <= g['underexposed_max'])
    result[clipped] = UNKNOWN
    result[~roi] = OUTSIDE
    return result


def summary(mask, cfg):
    roi_count = int(np.count_nonzero(mask != OUTSIDE))
    valid = int(np.count_nonzero((mask == METAL) | (mask == RESIDUE)))
    residual = int(np.count_nonzero(mask == RESIDUE))
    unknown = int(np.count_nonzero(mask == UNKNOWN))
    coverage = valid/roi_count if roi_count else 0.0
    ratio = residual/valid if valid else None
    a = cfg['assessment']
    # Known residue remains a rework candidate even if other areas are unobservable.
    lower_bound = residual/roi_count if roi_count else 0.0
    if roi_count and lower_bound > a['max_residual_fraction']:
        status = 'rework_candidate'
    elif not valid or coverage < a['min_evaluable_fraction']:
        status = 'review'
    elif ratio > a['max_residual_fraction']:
        status = 'rework_candidate'
    else:
        status = 'clean_candidate'
    return dict(roi_pixels=roi_count, evaluable_pixels=valid, residual_pixels=residual,
                unknown_pixels=unknown, evaluable_fraction=coverage,
                residual_fraction_evaluable=ratio, residual_fraction_roi_lower_bound=lower_bound,
                unknown_fraction=unknown/roi_count if roi_count else None, status=status)


def measure(mask, original_shape, cfg):
    result = summary(mask, cfg)
    h, w = mask.shape
    sy, sx = original_shape[0]/h, original_shape[1]/w
    grid = int(cfg['assessment']['grid_px'])
    cells = []
    for y in range(0, h, grid):
        for x in range(0, w, grid):
            cell = summary(mask[y:y+grid, x:x+grid], cfg)
            if cell['roi_pixels']:
                cell['box_xyxy_working'] = [x, y, min(x+grid, w), min(y+grid, h)]
                cells.append(cell)
    n, labels, stats, centers = cv2.connectedComponentsWithStats((mask == RESIDUE).astype(np.uint8), 8)
    defects = []
    mm = cfg['assessment'].get('mm_per_pixel_original')
    if mm is not None and mm <= 0:
        raise ValueError('mm_per_pixel_original must be positive or null')
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if area < cfg['assessment']['min_component_area_px']:
            continue  # reporting only; tiny residuals are still counted in metrics
        item = dict(area_px_original=float(area*sx*sy),
                    box_xyxy_original=[int(round(x*sx)), int(round(y*sy)),
                                       int(round((x+cw)*sx)), int(round((y+ch)*sy))],
                    centroid_xy_original=[float(centers[i, 0]*sx), float(centers[i, 1]*sy)])
        if mm is not None:
            item['area_mm2'] = item['area_px_original']*mm*mm
        defects.append(item)
    defects.sort(key=lambda d: d['area_px_original'], reverse=True)
    result.update(cells=cells, residual_components=defects,
                  largest_residual_area_px_original=defects[0]['area_px_original'] if defects else 0,
                  coordinate_frame='original image pixels; NOT robot/world coordinates',
                  original_size_wh=[original_shape[1], original_shape[0]], working_size_wh=[w, h])
    return result


def visualize(image, mask, stats):
    color = np.zeros_like(image)
    for cls in (METAL, RESIDUE, UNKNOWN):
        color[mask == cls] = COLORS[cls]
    overlay = image.copy()
    inside = mask != OUTSIDE
    overlay[inside] = (image[inside]*0.55+color[inside]*0.45).astype(np.uint8)
    overlay[~inside] = (image[~inside]*0.35).astype(np.uint8)
    heat = image.copy()
    states = {'clean_candidate': COLORS[METAL], 'rework_candidate': COLORS[RESIDUE], 'review': COLORS[UNKNOWN]}
    for cell in stats['cells']:
        x0, y0, x1, y1 = cell['box_xyxy_working']
        roi = mask[y0:y1, x0:x1] != OUTSIDE
        block = heat[y0:y1, x0:x1]
        block[roi] = (block[roi]*0.5+states[cell['status']]*0.5).astype(np.uint8)
        cv2.rectangle(heat, (x0, y0), (x1-1, y1-1), (80, 80, 80), 1)
    return overlay, heat
