"""Fetch pinned third-party source and official encoder initialization weights."""
from pathlib import Path
import hashlib
import shutil
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parent
URL = 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'
SMP_URL = 'https://github.com/qubvel-org/segmentation_models.pytorch.git'
SMP_TAG = 'v0.5.0'
SMP_COMMIT = '420ce84d1f667f29a77a5de69bfc9d2f79185e71'


def main():
    source = ROOT / 'third_party/segmentation_models.pytorch'
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ['git', 'clone', '--depth', '1', '--branch', SMP_TAG, SMP_URL, str(source)],
            check=True,
        )
    commit = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True
    ).strip()
    if commit != SMP_COMMIT:
        raise RuntimeError(
            f'Unexpected segmentation_models.pytorch commit: {commit}; expected {SMP_COMMIT}'
        )

    weights = ROOT / 'weights'
    weights.mkdir(exist_ok=True)
    target = weights / 'mobilenet_v2-b0353104.pth'
    if not target.exists():
        temporary = target.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=120) as response, temporary.open('wb') as out:
            shutil.copyfileobj(response, out)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if not digest.startswith('b0353104'):
            raise RuntimeError(f'Official weight checksum mismatch: {digest}')
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if not digest.startswith('b0353104'):
        raise RuntimeError('Existing encoder weight checksum mismatch')
    print(f'segmentation_models.pytorch: {commit}')
    print(f'encoder: {target} ({digest})')
    print('Place private inspection images in data/images/. They are intentionally not downloaded.')


if __name__ == '__main__':
    main()
