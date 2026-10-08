#!/usr/bin/env python3
"""Read-only checks for source publication. Never print matched secret values."""
import argparse
from pathlib import Path
import re
import subprocess


RULES = {
    'private absolute path': re.compile(r'(?<![\w.$}])/(?:home|Users|data-sdb|mnt)/[\w.-]+|(?<![\w.$}])/data/[\w.-]+'),
    'access token': re.compile(r'\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b'),
    'private key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'literal credential': re.compile(r'''(?i)(?:api_key|access_token|password)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{16,}["']'''),
    'credential in URL': re.compile(r'https?://[^/\s:@]+:[^/\s@]+@'),
}
ARTIFACT_SUFFIXES = {'.pt', '.pth', '.bin', '.safetensors', '.ckpt', '.arrow', '.pkl', '.log', '.pem', '.key', '.p12', '.pfx'}


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def candidate_files(repo):
    """Include tracked files even if ignored, and non-ignored new source files."""
    output = git(repo, 'ls-files', '--cached', '--others', '--exclude-standard', '-z')
    return sorted(set(p for p in output.decode().split('\0') if p))


def content_findings(name, content):
    findings = []
    for number, line in enumerate(content.decode('utf-8', errors='replace').splitlines(), 1):
        for label, pattern in RULES.items():
            if pattern.search(line):
                findings.append(f'{name}:{number}: {label}')
    return findings


def audit(repo, history=True):
    repo = Path(repo).resolve()
    files = candidate_files(repo)
    errors, warnings = [], []
    for name in files:
        path = repo / name
        if any(part.is_symlink() for part in [path, *path.parents] if part != repo and repo in part.parents):
            errors.append(f'{name}: symlink is not publishable')
            continue
        if not path.is_file():
            errors.append(f'{name}: tracked file missing from working tree')
            continue
        if path.suffix.lower() in ARTIFACT_SUFFIXES or (
            path.name.startswith('.env') and path.name != '.env.example'
        ) or '.git' in path.parts or (
            path.parts[len(repo.parts)] in {'outputs', 'data', 'checkpoints', 'llama', '.cache', 'wandb'}
            and name not in {'data/README.md', 'checkpoints/README.md'}
        ):
            errors.append(f'{name}: runtime artifact or local configuration')
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f'{name}: source file exceeds 10 MiB')
            continue
        content = path.read_bytes()
        if b'\0' in content:
            errors.append(f'{name}: unexpected binary source file')
        errors.extend(content_findings(name, content))
    if history:
        # Check each unique reachable historical blob, including deleted files.
        seen = set()
        for revision in git(repo, 'rev-list', '--all').decode().splitlines():
            entries = git(repo, 'ls-tree', '-r', '-z', revision).split(b'\0')
            for entry in entries:
                if not entry:
                    continue
                meta, filename = entry.split(b'\t', 1)
                _, kind, oid = meta.split()
                if kind != b'blob' or oid in seen:
                    continue
                seen.add(oid)
                name = filename.decode()
                if int(git(repo, 'cat-file', '-s', oid.decode())) > 10 * 1024 * 1024:
                    errors.append(f'history/{revision[:8]}/{name}: large historical blob')
                    continue
                errors.extend(content_findings(f'history/{revision[:8]}/{name}', git(repo, 'cat-file', 'blob', oid.decode())))
        emails = git(repo, 'log', '--all', '--format=%ae%n%ce').decode().splitlines()
        private_emails = {e for e in emails if e and not e.endswith('@users.noreply.github.com')}
        if private_emails:
            warnings.append('Git history contains non-noreply author/committer emails; use a history-free source snapshot or explicitly review identities.')
    return files, errors, warnings


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--no-history', action='store_true', help='Check working source only, as used for a history-free snapshot.')
    args = parser.parse_args()
    files, errors, warnings = audit(args.repo, history=not args.no_history)
    for error in errors:
        print('ERROR:', error)
    for warning in warnings:
        print('WARNING:', warning)
    print(f'Checked {len(files)} source candidates; {len(errors)} errors; {len(warnings)} warnings.')
    print('Pattern checks are not proof that all sensitive information has been removed.')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
