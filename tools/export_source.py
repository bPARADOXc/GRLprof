#!/usr/bin/env python3
"""Export checked working-tree source without Git history or ignored artifacts."""
import argparse
import hashlib
from pathlib import Path
import stat
import zipfile

from check_release import audit


def export(repo, output):
    files, errors, _ = audit(repo, history=False)
    if errors:
        raise ValueError('Release checks failed:\n' + '\n'.join(errors))
    output = Path(output).resolve()
    # Avoid including a partially written archive; refuse overwrite as well.
    if output.exists() or output in [(Path(repo) / name).resolve() for name in files]:
        raise ValueError('Output already exists or overlaps source; choose a new output filename.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            path = Path(repo) / name
            info = zipfile.ZipInfo('GRLprof/' + name)
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if path.stat().st_mode & stat.S_IXUSR else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, path.read_bytes())
    print(f'Exported {len(files)} source files to {output}')
    print('SHA256:', hashlib.sha256(output.read_bytes()).hexdigest())
    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    export(args.repo.resolve(), args.output)
