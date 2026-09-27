"""Obtain official pinned LayerSkip outside the public repository."""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from april.config import LAYERSKIP_REVISION


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    from april.config import require_frozen_config
    require_frozen_config(args.config)
    dest=args.destination.resolve()
    root=Path(__file__).resolve().parents[1]
    if dest==root or root in dest.parents or dest.exists():
        parser.error('Use a new external directory, not inside this repository')
    subprocess.run(['git','-c','core.autocrlf=false','clone','https://github.com/facebookresearch/LayerSkip.git',str(dest)],check=True,timeout=600)
    subprocess.run(['git','-C',str(dest),'checkout','--detach',LAYERSKIP_REVISION],check=True,timeout=60)
    print('Installed official external LayerSkip at',LAYERSKIP_REVISION)


if __name__=='__main__':main()
