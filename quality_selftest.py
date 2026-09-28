"""Synthetic plumbing checks; NOT quality accuracy measurements on user photos."""
import copy
import json
import tempfile
from pathlib import Path
import cv2
import numpy as np
from quality_core import summary, measure, roi_mask, reject_unobservable, write_image, resolve, ROOT, features


def run_tests(cfg):
    checks = []
    empty = np.full((24, 32), 255, np.uint8)
    assert summary(empty, cfg)['status'] == 'review'
    unknown = np.full_like(empty, 2)
    assert summary(unknown, cfg)['residual_fraction_evaluable'] is None
    assert summary(unknown, cfg)['status'] == 'review'
    checks.append('empty ROI and entirely unknown image cannot pass')
    mask = np.zeros((24, 32), np.uint8)
    mask[0, 0] = 1
    assert measure(mask, (48, 64, 3), cfg)['residual_pixels'] == 1
    mask[2:10, 3:15] = 1
    m = measure(mask, (48, 64, 3), cfg)
    assert m['residual_components'][0]['area_px_original'] == 8*12*4
    assert m['residual_components'][0]['box_xyxy_original'] == [6, 4, 30, 20]
    checks.append('small residuals counted; area and boxes map to original pixel scale')
    im = np.full((24, 32, 3), 255, np.uint8)
    roi = np.ones((24, 32), bool)
    roi[:4] = False
    gated = reject_unobservable(np.zeros((24, 32), np.uint8), im, roi, cfg)
    assert np.all(gated[:4] == 255) and np.all(gated[4:] == 2)
    assert features(im)[0].shape == (24, 32, 13)
    checks.append('clipping excluded from evaluation; feature shape stable')

    from quality_models import (training_records, train_rf, train_unet, Predictor, build_unet,
                                normalize, export_onnx, tile_probabilities)
    import torch
    with tempfile.TemporaryDirectory(prefix='quality_check_') as temporary:
        d = Path(temporary)
        c = copy.deepcopy(cfg)
        c['max_side'] = 128
        c['roi'] = {'default':[0,0,1,1], 'overrides':{}, 'exclude_rects':{}}
        c['assessment']['grid_px'] = 32
        for i in (0, 1):
            # Distinct synthetic scenes for testing plumbing only, no accuracy claim.
            x = np.full((96, 128, 3), 195+i, np.uint8)
            x[:, 64:] = (30+i, 70+i, 110+i)
            y = np.zeros((96, 128), np.uint8)
            y[:, 64:] = 1
            write_image(d/f'im{i}.png', x)
            write_image(d/f'mask{i}.png', y)
        manifest = d/'split.csv'
        manifest.write_text('image,mask,split,group\n'+
            f'{(d/"im0.png").as_posix()},{(d/"mask0.png").as_posix()},train,A\n'+
            f'{(d/"im1.png").as_posix()},{(d/"mask1.png").as_posix()},val,A\n', encoding='utf-8')
        c['training']['manifest'] = str(manifest)
        try:
            training_records(c)
        except ValueError as exc:
            assert 'leakage' in str(exc)
        else:
            raise AssertionError('Group leakage was not rejected')
        manifest.write_text(manifest.read_text().replace(',val,A', ',val,B'), encoding='utf-8')
        c['rf']['checkpoint'] = str(d/'rf.joblib')
        train_rf(c)
        c['method'] = 'rf'
        prediction, score = Predictor(c).predict(cv2.imread(str(d/'im1.png')))
        assert set(np.unique(prediction)) == {0, 1}
        assert prediction.shape == (96, 128) and np.isfinite(score).all()
        checks.append('RF train / checkpoint reload / inference and workpiece split guard')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(42)
        model = build_unet(cfg['unet']['encoder_weights']).to(device)
        model.train()
        x = torch.randn(2, 3, 64, 64, device=device)
        y = torch.randint(0, 2, (2, 64, 64), device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        assert torch.isfinite(loss) and any(p.grad is not None for p in model.parameters())
        optimizer.step()
        checks.append(f'official encoder loaded; U-Net forward / backward / update on {device}')
        # Exercise the actual training entry with disposable synthetic labels.
        c['unet'].update(checkpoint=str(d/'test_only.pt'), onnx_path=str(d/'test_only.onnx'),
                         input_size=64, device=device)
        c['training'].update(epochs=1, batch_size=2)
        train_unet(c)
        checks.append('actual U-Net training entry / validation / checkpoint on synthetic labels')
        c['unet']['device'] = 'cpu'
        export_onnx(c)
        c['method'] = 'unet'
        pt = Predictor(c)
        c_onnx = copy.deepcopy(c)
        c_onnx['method'] = 'onnx'
        onnx = Predictor(c_onnx)
        # Include an odd-sized image to exercise overlap and edge padding.
        small = np.full((73, 101, 3), 180, np.uint8)
        pt_mask, pt_score = pt.predict(small)
        onnx_mask, onnx_score = onnx.predict(small)
        assert pt_mask.shape == (73, 101)
        np.testing.assert_allclose(pt_score, onnx_score, rtol=1e-3, atol=1e-4)
        np.testing.assert_array_equal(pt_mask, onnx_mask)
        checks.append('ONNX checker / CPU numerical parity / odd-sized tiled inference')
    report = {'passed':checks, 'meaning':'Software plumbing only, no measured real-image segmentation accuracy',
              'production_quality_model_trained':False}
    output = ROOT/'outputs/self_test.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
