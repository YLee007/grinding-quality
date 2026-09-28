"""Trainable RF and MobileNetV2 U-Net backends; no implicit network downloads."""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path
import cv2
import numpy as np
from quality_core import ROOT, FEATURE_VERSION, features, resolve, read_image, resize_image, roi_mask


def build_unet(encoder_path=None):
    import torch
    source = ROOT / 'third_party/segmentation_models.pytorch'
    if source.is_dir() and str(source) not in sys.path:
        sys.path.insert(0, str(source))
    import segmentation_models_pytorch as smp
    net = smp.Unet(encoder_name='mobilenet_v2', encoder_weights=None,
                   decoder_channels=(128, 64, 32, 16, 16), in_channels=3, classes=2)
    if encoder_path is not None:
        weights = torch.load(resolve(encoder_path), map_location='cpu', weights_only=True)
        net.encoder.load_state_dict(weights, strict=True)
    return net


def normalize(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)/255
    return ((rgb-np.array([0.485, 0.456, 0.406], np.float32)) /
            np.array([0.229, 0.224, 0.225], np.float32)).transpose(2, 0, 1).copy()


def select_device(cfg):
    import torch
    value = cfg['unet'].get('device', 'auto')
    return ('cuda' if torch.cuda.is_available() else 'cpu') if value == 'auto' else value


def tile_probabilities(image, size, forward):
    """Overlapping tiles preserve local features; same routine for Torch and ONNX."""
    if size < 64 or size % 32:
        raise ValueError('unet.input_size must be >=64 and divisible by 32')
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(image, 0, max(0, size-h), 0, max(0, size-w), cv2.BORDER_REFLECT_101)
    ph, pw = padded.shape[:2]
    step = size*3//4
    ys = sorted(set(list(range(0, ph-size+1, step))+[ph-size]))
    xs = sorted(set(list(range(0, pw-size+1, step))+[pw-size]))
    total = np.zeros((ph, pw, 2), np.float32)
    counts = np.zeros((ph, pw, 1), np.float32)
    for y in ys:
        for x in xs:
            logits = forward(normalize(padded[y:y+size, x:x+size])[None])[0]
            logits = logits - logits.max(axis=0, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=0, keepdims=True)
            total[y:y+size, x:x+size] += prob.transpose(1, 2, 0)
            counts[y:y+size, x:x+size] += 1
    return (total/counts)[:h, :w]


class Predictor:
    def __init__(self, cfg):
        self.cfg = cfg
        self.method = cfg['method']
        if self.method == 'baseline':
            return
        if self.method == 'rf':
            import joblib
            payload = joblib.load(resolve(cfg['rf']['checkpoint']))
            if payload.get('feature_version') != FEATURE_VERSION or not payload.get('trained'):
                raise ValueError('Not a compatible trained RF checkpoint')
            if payload.get('max_side') != cfg['max_side']:
                raise ValueError('RF texture scale differs: use the max_side used during training')
            self.model = payload['model']
        elif self.method == 'unet':
            import torch
            self.device = select_device(cfg)
            payload = torch.load(resolve(cfg['unet']['checkpoint']), map_location='cpu', weights_only=True)
            if not payload.get('trained') or payload.get('classes') != ['metal', 'residue']:
                raise ValueError('A grinding-quality trained checkpoint is required, not ImageNet weights')
            if payload.get('max_side') != cfg['max_side']:
                raise ValueError('Use the same max_side as the trained U-Net checkpoint')
            self.model = build_unet().to(self.device).eval()
            self.model.load_state_dict(payload['state_dict'])
        elif self.method == 'onnx':
            import onnxruntime as ort
            onnx_path = resolve(cfg['unet']['onnx_path'])
            metadata = json.loads(onnx_path.with_suffix('.json').read_text(encoding='utf-8'))
            if metadata.get('trained') is not True or metadata.get('input_size') != cfg['unet']['input_size']:
                raise ValueError('ONNX trained metadata / input_size mismatch')
            if metadata.get('max_side') != cfg['max_side']:
                raise ValueError('ONNX max_side mismatch')
            self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        else:
            raise ValueError(f'Unknown method: {self.method}')

    def predict(self, image):
        if self.method == 'baseline':
            from quality_core import baseline
            return baseline(image, self.cfg)
        if self.method == 'rf':
            x = features(image)[0]
            flat = x.reshape(-1, x.shape[-1])
            prob = np.concatenate([self.model.predict_proba(flat[i:i+65536])
                                   for i in range(0, len(flat), 65536)], axis=0).reshape(*image.shape[:2], 2)
            threshold = self.cfg['rf']['min_confidence']
        else:
            if self.method == 'unet':
                import torch
                def forward(x):
                    with torch.inference_mode():
                        return self.model(torch.from_numpy(x).to(self.device)).cpu().numpy()
            else:
                def forward(x):
                    return self.session.run(None, {self.session.get_inputs()[0].name: x})[0]
            prob = tile_probabilities(image, self.cfg['unet']['input_size'], forward)
            threshold = self.cfg['unet']['min_confidence']
        confidence = prob.max(axis=2)
        mask = prob.argmax(axis=2).astype(np.uint8)
        mask[confidence < threshold] = 2
        return mask, confidence


def training_records(cfg):
    path = resolve(cfg['training']['manifest'])
    if not path.exists():
        raise ValueError('Missing data/split.csv. Annotate images and set train/val workpiece groups first; see README.')
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    groups, hashes = {}, {}
    for row in rows:
        if row['split'] not in ('train', 'val') or not row['group'].strip():
            raise ValueError('Every row requires split=train/val and a nonempty workpiece group')
        for mapping, key in ((groups, row['group']),
                             (hashes, hashlib.sha256(resolve(row['image']).read_bytes()).hexdigest())):
            if key in mapping and mapping[key] != row['split']:
                raise ValueError('Train/val leakage: same workpiece group or identical image in both splits')
            mapping[key] = row['split']
        if not resolve(row['mask']).is_file():
            raise ValueError(f'Missing human annotation: {row["mask"]}')
    if {r['split'] for r in rows} != {'train', 'val'}:
        raise ValueError('Need independent train and val images. Do not split adjacent crops of one plate across sets.')
    return rows


def load_record(row, cfg):
    original = read_image(resolve(row['image']))
    raw = cv2.imdecode(np.fromfile(str(resolve(row['mask'])), np.uint8), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape != original.shape[:2]:
        raise ValueError('Mask must match original dimensions and contain 0=metal, 1=residue, 255=ignore')
    if not set(np.unique(raw)).issubset({0, 1, 255}):
        raise ValueError('Unexpected mask label; use 0, 1, 255 only')
    im = resize_image(original, cfg['max_side'])
    mask = cv2.resize(raw, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask[~roi_mask(im, Path(row['image']).name, cfg)] = 255
    if not np.any(mask != 255):
        raise ValueError(f'No labeled pixels inside ROI: {row["image"]}')
    return im, mask


def confusion_scores(cm):
    denom = cm.sum(axis=0)+cm.sum(axis=1)-cm.diagonal()
    iou = np.divide(cm.diagonal(), denom, out=np.zeros(2, float), where=denom > 0)
    return {'confusion_true_rows_pred_columns': cm.tolist(),
            'iou_metal': float(iou[0]), 'iou_residue': float(iou[1]),
            'residue_recall': float(cm[1, 1]/max(1, cm[1].sum())),
            'note': 'Metrics on human-labeled validation pixels only, not full-image certification'}


def train_rf(cfg):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    rows = training_records(cfg)
    rng = np.random.default_rng(cfg['training']['seed'])
    xx, yy = [], []
    for row in rows:
        if row['split'] != 'train':
            continue
        im, mask = load_record(row, cfg)
        x = features(im)[0].reshape(-1, 13)
        for cls in (0, 1):
            idx = np.flatnonzero(mask.ravel() == cls)
            idx = rng.choice(idx, min(len(idx), cfg['training']['samples_per_class_per_image']), replace=False)
            xx.append(x[idx])
            yy.append(np.full(len(idx), cls))
    x, y = np.concatenate(xx), np.concatenate(yy)
    if len(np.unique(y)) != 2:
        raise ValueError('Training requires metal AND residue human labels')
    model = RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=4,
                                    class_weight='balanced', n_jobs=-1, random_state=cfg['training']['seed'])
    model.fit(x, y)
    cm = np.zeros((2, 2), np.int64)
    for row in rows:
        if row['split'] != 'val':
            continue
        im, mask = load_record(row, cfg)
        valid = mask != 255
        pred = model.predict(features(im)[0][valid])
        cm += np.bincount(2*mask[valid]+pred, minlength=4).reshape(2, 2)
    target = resolve(cfg['rf']['checkpoint'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f'Choose a new checkpoint path to keep the old model: {target}')
    metrics = confusion_scores(cm)
    joblib.dump(dict(model=model, trained=True, feature_version=FEATURE_VERSION,
                     max_side=cfg['max_side'], records=rows, metrics=metrics), target)
    target.with_suffix('.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


def train_unet(cfg):
    import torch
    from torch.utils.data import Dataset, DataLoader
    rows = training_records(cfg)
    seed = cfg['training']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    size = cfg['unet']['input_size']
    if cfg['training']['epochs'] < 1 or cfg['training']['batch_size'] < 1:
        raise ValueError('epochs and batch_size must be >= 1')
    if size < 64 or size % 32:
        raise ValueError('input_size must be a multiple of 32, >=64')
    target = resolve(cfg['unet']['checkpoint'])
    if target.exists():
        raise FileExistsError(f'Choose a new checkpoint path: {target}')
    cached = [load_record(r, cfg) for r in rows if r['split'] == 'train']
    counts = np.array([sum(np.count_nonzero(m == c) for _, m in cached) for c in (0, 1)])
    if np.any(counts == 0):
        raise ValueError('Both metal and residue annotations are required')

    class Crops(Dataset):
        def __len__(self):
            return len(cached)*12

        def __getitem__(self, index):
            im, mask = cached[index % len(cached)]
            h, w = mask.shape
            im = cv2.copyMakeBorder(im, 0, max(0, size-h), 0, max(0, size-w), cv2.BORDER_REFLECT_101)
            mask = cv2.copyMakeBorder(mask, 0, max(0, size-h), 0, max(0, size-w), cv2.BORDER_CONSTANT, value=255)
            yy, xx = np.where(mask != 255)
            p = np.random.randint(len(xx))
            x = int(np.clip(xx[p]-size//2, 0, im.shape[1]-size))
            y = int(np.clip(yy[p]-size//2, 0, im.shape[0]-size))
            im, mask = im[y:y+size, x:x+size].copy(), mask[y:y+size, x:x+size].copy()
            if random.random() < 0.5:
                im, mask = im[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                im, mask = im[::-1], mask[::-1]
            im = np.clip(im*random.uniform(0.90, 1.10), 0, 255).astype(np.uint8)
            return torch.from_numpy(normalize(im)), torch.from_numpy(mask.astype(np.int64).copy())

    device = select_device(cfg)
    model = build_unet(cfg['unet']['encoder_weights']).to(device)
    loader = DataLoader(Crops(), batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
    weights = (counts.sum()/counts).astype(np.float32)
    weights /= weights.mean()
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=255, weight=torch.tensor(weights, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'])
    val = [load_record(r, cfg) for r in rows if r['split'] == 'val']
    best, history = -1.0, []
    target.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg['training']['epochs']):
        model.train()
        losses = []
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        cm = np.zeros((2, 2), np.int64)
        def forward(x):
            with torch.inference_mode():
                return model(torch.from_numpy(x).to(device)).cpu().numpy()
        for im, mask in val:
            prediction = tile_probabilities(im, size, forward).argmax(axis=2)
            valid = mask != 255
            cm += np.bincount(2*mask[valid]+prediction[valid], minlength=4).reshape(2, 2)
        scores = confusion_scores(cm)
        scores.update(epoch=epoch+1, train_loss=float(np.mean(losses)))
        history.append(scores)
        print(json.dumps(scores), flush=True)
        mean_iou = (scores['iou_metal']+scores['iou_residue'])/2
        if mean_iou > best:
            best = mean_iou
            torch.save(dict(state_dict={k: v.cpu() for k, v in model.state_dict().items()}, trained=True,
                            classes=['metal', 'residue'], input_size=size, max_side=cfg['max_side'],
                            metrics=scores, records=rows), target)
        target.with_suffix('.history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')


def export_onnx(cfg):
    import torch
    import onnx
    import onnxruntime as ort
    c = dict(cfg, method='unet')
    c['unet'] = dict(cfg['unet'], device='cpu')
    predictor = Predictor(c)
    size = c['unet']['input_size']
    target = resolve(c['unet']['onnx_path'])
    if target.exists():
        raise FileExistsError(f'Choose a new ONNX path: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    x = torch.randn(1, 3, size, size)
    torch.onnx.export(predictor.model, x, str(target), input_names=['image'],
                      output_names=['logits'], opset_version=17, do_constant_folding=True)
    onnx.checker.check_model(str(target))
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    with torch.inference_mode():
        reference = predictor.model(x).numpy()
    result = session.run(None, {'image': x.numpy()})[0]
    np.testing.assert_allclose(result, reference, rtol=1e-3, atol=1e-4)
    metadata = dict(trained=True, input_size=size, max_side=cfg['max_side'],
                    classes=['metal', 'residue'], input='float32 NCHW RGB ImageNet normalized',
                    output='float32 NCHW logits; softmax then confidence rejection',
                    max_abs_error=float(np.max(np.abs(result-reference))))
    target.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata, indent=2))
