"""Exercise APRIL with the two original synthetic prompts and a 32-token cap."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from april.cli import main

if __name__ == '__main__':
    if '--input' in sys.argv or '--max-new-tokens' in sys.argv:
        raise SystemExit('Smoke inputs and the 32-token cap are fixed by this command')
    sys.argv.extend(['--input',str(Path(__file__).resolve().parents[1]/'examples/smoke_prompts.jsonl'),
                     '--max-new-tokens','32'])
    main('APRIL')
