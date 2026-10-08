"""Release tests use disposable repositories; no GPU or external assets."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
from check_release import audit
from export_source import export


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.name', 'Test Contributor')
        self.git('config', 'user.email', 'test@users.noreply.github.com')
        (self.repo / '.gitignore').write_text('outputs/\n.env\n')
        (self.repo / 'README.md').write_text('# Example\n')
        self.git('add', '.')
        self.git('commit', '-qm', 'initial')

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args])

    def test_clean_and_export(self):
        (self.repo / '.env').write_text('PRIVATE=excluded')
        files, errors, warnings = audit(self.repo)
        self.assertFalse(errors)
        self.assertFalse(warnings)
        archive = Path(self.tmp.name) / 'source.zip'
        export(self.repo, archive)
        with zipfile.ZipFile(archive) as z:
            self.assertEqual(set(z.namelist()), {'GRLprof/' + f for f in files})
            self.assertNotIn('GRLprof/.env', z.namelist())
        with self.assertRaises(ValueError):
            export(self.repo, archive)

    def test_secret_and_personal_path(self):
        token = 'hf_' + 'x' * 30
        (self.repo / 'bad.py').write_text('TOKEN="' + token + '"\npath="/' + 'home/private/model"\n')
        _, errors, _ = audit(self.repo, history=False)
        self.assertTrue(any('access token' in x for x in errors))
        self.assertTrue(any('private absolute path' in x for x in errors))
        self.assertTrue(all(token not in x for x in errors))

    def test_historical_secret(self):
        path = self.repo / 'old.txt'
        path.write_text('sk-' + 'x' * 30)
        self.git('add', '.')
        self.git('commit', '-qm', 'old secret')
        path.unlink()
        self.git('add', '-u')
        self.git('commit', '-qm', 'remove')
        self.assertFalse(audit(self.repo, history=False)[1])
        self.assertTrue(any('history/' in e for e in audit(self.repo)[1]))

    def test_tracked_ignored_artifact(self):
        path = self.repo / 'outputs' / 'cache.bin'
        path.parent.mkdir()
        path.write_bytes(b'weights')
        self.git('add', '-f', 'outputs/cache.bin')
        self.assertTrue(audit(self.repo, history=False)[1])

    def test_symlink(self):
        (self.repo / 'linked').symlink_to(self.repo / 'README.md')
        self.assertTrue(any('symlink' in e for e in audit(self.repo, history=False)[1]))


if __name__ == '__main__':
    unittest.main()
