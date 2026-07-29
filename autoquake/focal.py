import shutil
import subprocess
from pathlib import Path


class GAfocal:
    def __init__(self, dout_path: Path, result_path: Path):
        self.dout_path = Path(dout_path)
        self.main_dir = Path(__file__).parents[1] / 'GAfocal'
        self.result_path = result_path

    def run(self):
        target = self.main_dir / self.dout_path.name
        shutil.copy2(self.dout_path, target)
        subprocess.run(
            ['./gafocal'], input=target.name.encode() + b'\n', cwd=self.main_dir
        )
        shutil.copy2(self.main_dir / 'results.txt', self.result_path / 'gafocal_catalog.txt')
