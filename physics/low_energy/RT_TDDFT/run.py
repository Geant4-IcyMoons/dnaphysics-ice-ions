"""Run target ground state and unperturbed TD control in a fresh directory."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from physics.elastic.bca.structure import file_sha256


def run(case, output, executable='octopus'):
    binary = shutil.which(executable)
    if binary is None:
        raise FileNotFoundError(f'Octopus executable unavailable: {executable}')
    case, output = Path(case).resolve(), Path(output).resolve()
    manifest = json.loads((case / 'manifest.json').read_text())
    for name in ('gs.inp', 'control_td.inp'):
        if file_sha256(case / name) != manifest['input_sha256'][name]:
            raise ValueError(f'Input hash mismatch: {name}; regenerate the case')
    version = subprocess.run([binary, '--version'], capture_output=True, text=True, check=True)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'binary': binary, 'binary_sha256': file_sha256(Path(binary)),
               'version': version.stdout + version.stderr,
               'case_manifest_sha256': file_sha256(case / 'manifest.json'),
               'status': 'running', 'stages': [], 'physical_validation': False}
    try:
        for stage, name in [('gs','gs.inp'), ('control_td','control_td.inp')]:
            shutil.copyfile(case / name, output / 'inp')
            with (output / f'{stage}.log').open('x') as log:
                result = subprocess.run([binary], cwd=output, stdout=log, stderr=subprocess.STDOUT)
            receipt['stages'].append({'stage': stage, 'returncode': result.returncode})
            if result.returncode:
                raise RuntimeError(f'{stage} failed; inspect {output / (stage + ".log")}')
        receipt['status'] = 'executed_requires_numerical_review'
    except BaseException:
        receipt['status'] = 'failed_or_interrupted'
        raise
    finally:
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--executable', default='octopus')
    args = parser.parse_args()
    run(args.case, args.output, args.executable)


if __name__ == '__main__':
    main()
