"""Verify and copy NASDAQ arrays into each implementation directory."""
from pathlib import Path
import argparse,hashlib,json,shutil

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    expected=json.loads((root/'data/nasdaq_hashes.json').read_text())
    configurations=json.loads((root/'configs/configurations.json').read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,value in expected.items():
        if digest(args.source/name)!=value:raise ValueError('Dataset mismatch: '+name)
    destinations=[root/c['source']/'dataset/NASDAQ' for c in configurations.values()]
    for dest in destinations:
        for name,value in expected.items():
            p=dest/name
            if p.exists() and digest(p)!=value:raise ValueError('Existing data differ: '+str(p))
    for dest in destinations:
        dest.mkdir(parents=True,exist_ok=True)
        for name,value in expected.items():
            p=dest/name
            if not p.exists():shutil.copyfile(args.source/name,p)
            assert digest(p)==value
        print(dest.relative_to(root))
    print('Dataset files verified.')

if __name__=='__main__':main()
