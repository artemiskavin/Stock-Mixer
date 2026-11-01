"""Generate a PowerShell reproduction command."""
from pathlib import Path
import argparse,json,re

def main():
    root=Path(__file__).resolve().parents[1]
    configs=json.loads((root/'configs/configurations.json').read_text())
    runs=json.loads((root/'configs/run_registry.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configuration',choices=list(configs),required=True)
    parser.add_argument('--seed-index',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--run-tag',default='new_reproduction')
    a=parser.parse_args();c=configs[a.configuration]
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_tag):parser.error('Use letters, digits, underscore or hyphen for run tag.')
    if c['group']=='author' and a.seed_index!=0:parser.error('These author configurations retain their released seed pair only.')
    print('Set-Location "'+str(root/c['source']/'src')+'"')
    python=str(root/'.venv/Scripts/python.exe')
    if c['group']=='author':command='& "'+python+'" -u train.py'
    else:
        r=next(r for r in runs if r['configuration']==a.configuration and r['seed']==a.seed_index)
        metadata=json.loads((root/r['result_directory']/'run_metadata.json').read_text())
        args=metadata['reproduction_arguments']
        output=root/'artifacts/runs'/a.run_tag/a.configuration/f'seed_{a.seed_index}'
        if output.exists():parser.error('Choose a new output run tag; destination already exists.')
        command=(f'& "{python}" -B train.py --run --seed-index {a.seed_index} '
            f'--precision {r["precision"]} --compile-policy {args["compile_policy"]} '
            f'--data-mode {args["data_mode"]} --diagnostics {args["diagnostics"]} '
            f'--preflight "{root/c["preflight"]}" --output "{output}"')
    print(command)
    print('# Command generated. Use the working directory shown above.')

if __name__=='__main__':main()
