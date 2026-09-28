"""Scribble/paint labels: 1=metal, 2=residue, 0=erase, S=save, N/P=image."""
import shutil
import time
import cv2
import numpy as np
from quality_core import (COLORS, resolve, image_paths, read_image, write_image,
                          resize_image, roi_mask)


def annotate(cfg):
    paths = image_paths(cfg['input'])
    if not paths:
        raise ValueError('No images to annotate')
    target = resolve(cfg['training']['annotations'])
    target.mkdir(parents=True, exist_ok=True)
    print(__doc__)
    print('Paint only known regions. Unpainted=255 ignore, NOT clean. [ / ] changes brush size. Q exits.')
    print('Ctrl+Z undo last stroke. S saves; N/P also saves before switching images.')
    index, cls, radius = 0, 0, 10
    while 0 <= index < len(paths):
        path = paths[index]
        original = read_image(path)
        display = resize_image(original, 1100)
        mask_path = target / f'{path.stem}.png'
        shape = original.shape[:2]
        if mask_path.exists():
            full_mask = cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_GRAYSCALE)
            if full_mask is None or full_mask.shape != shape:
                raise ValueError(f'Annotation dimensions differ: {mask_path}')
        else:
            full_mask = np.full(shape, 255, np.uint8)
        roi_full = roi_mask(original, path.name, cfg)
        dh, dw = display.shape[:2]
        sy, sx = shape[0]/dh, shape[1]/dw
        state = dict(drawing=False, changed=False, previous=None)
        history = []
        title = 'Quality labels - 1 metal  2 residue  0 erase  [ ] brush  S save  N/P  Q'

        def paint(event, x, y, flags, param):
            nonlocal full_mask
            if event == cv2.EVENT_LBUTTONDOWN:
                history.append(full_mask.copy())
                if len(history) > 10:
                    history.pop(0)
                state['drawing'] = True
                state['previous'] = None
            if state['drawing'] and event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
                point = (int(np.clip(x*sx, 0, shape[1]-1)), int(np.clip((y-65)*sy, 0, shape[0]-1)))
                if y >= 65:
                    if state['previous'] is not None:
                        cv2.line(full_mask, state['previous'], point, cls, max(1, int(2*radius*sx)))
                    cv2.circle(full_mask, point, max(1, int(radius*sx)), cls, -1)
                    full_mask[~roi_full] = 255
                    state['changed'] = True
                    state['previous'] = point
            if event == cv2.EVENT_LBUTTONUP:
                state['drawing'] = False
                state['previous'] = None

        def save():
            if not state['changed']:
                return
            if mask_path.exists():
                backup = mask_path.with_name(mask_path.stem+f'.{time.time_ns()}.bak.png')
                shutil.copy2(mask_path, backup)
            write_image(mask_path, full_mask)
            state['changed'] = False
            print(f'Saved: {mask_path}', flush=True)

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, paint)
        while True:
            shown_mask = cv2.resize(full_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
            canvas = display.copy()
            for label in (0, 1):
                region = shown_mask == label
                canvas[region] = (0.5*canvas[region]+0.5*COLORS[label]).astype(np.uint8)
            inside = cv2.resize(roi_full.astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0
            canvas[~inside] //= 3
            canvas = cv2.copyMakeBorder(canvas, 65, 0, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
            label_name = {0:'METAL', 1:'RESIDUE', 255:'ERASE / IGNORE'}[cls]
            cv2.putText(canvas, f'{path.name}  {index+1}/{len(paths)}  BRUSH={label_name} r={radius}',
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (240,240,240), 1)
            cv2.putText(canvas, 'S save | N next | P previous | Ctrl+Z undo | Q save & quit',
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,230,230), 1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(30) & 255
            if key in (ord('1'), ord('2'), ord('0')):
                cls = {ord('1'):0, ord('2'):1, ord('0'):255}[key]
            elif key == ord('['):
                radius = max(1, radius-2)
            elif key == ord(']'):
                radius = min(150, radius+2)
            elif key == 26 and history:
                full_mask = history.pop()
                state['changed'] = True
            elif key == ord('s'):
                save()
            elif key in (ord('n'), ord('p'), ord('q'), 27) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                save()
                cv2.destroyAllWindows()
                if key in (ord('n'), ord('p')):
                    index += 1 if key == ord('n') else -1
                    index = max(0, index)
                    break
                return
