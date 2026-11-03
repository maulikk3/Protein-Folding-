"""Part 2 data prep: download PDB chains and save sequence + C-beta distance matrix.

Input : a text file with one chain per line, e.g. `1ubq_A` (PDB id + chain).
Output: data/contacts/<id>_<chain>.npz with `seq` (string) and `dist` (L x L, Angstrom, NaN = missing).

Important: pick your chain list from a sequence-clustered set (e.g. PISCES culled PDB list at
<=30% identity, resolution <= 2.5 A) so train and test proteins are not near-duplicates.

Example:
  python contact_prep.py chains.txt --max_len 300
"""
import argparse
import os

import numpy as np
import requests
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1


def fetch_pdb(pdb_id, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{pdb_id}.pdb")
    if not os.path.exists(path):
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb", timeout=30)
        r.raise_for_status()
        open(path, "w").write(r.text)
    return path


def chain_to_arrays(path, chain_id):
    model = PDBParser(QUIET=True).get_structure("x", path)[0]
    seq, coords = [], []
    for res in model[chain_id]:
        if res.id[0] != " ":  # skip waters / ligands
            continue
        atom = res["CB"] if "CB" in res else (res["CA"] if "CA" in res else None)  # Gly -> CA
        letter = seq1(res.get_resname())
        if atom is None or letter == "X":
            continue
        seq.append(letter)
        coords.append(atom.coord)
    return "".join(seq), np.array(coords, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chains_file")
    ap.add_argument("--out", default="data/contacts")
    ap.add_argument("--raw", default="data/pdb")
    ap.add_argument("--min_len", type=int, default=50)
    ap.add_argument("--max_len", type=int, default=300)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    kept = 0
    for line in open(args.chains_file):
        line = line.strip()
        if not line:
            continue
        pdb_id, chain = line.split("_")
        try:
            seq, xyz = chain_to_arrays(fetch_pdb(pdb_id, args.raw), chain)
        except Exception as e:
            print("skip", line, e)
            continue
        if not (args.min_len <= len(seq) <= args.max_len):
            continue
        dist = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
        np.savez_compressed(os.path.join(args.out, f"{line}.npz"), seq=seq, dist=dist)
        kept += 1
    print("saved", kept, "chains to", args.out)
    # Note: residues missing from the structure are dropped, so very gappy chains give
    # slightly wrong sequence context. Fine for a student project; mention it in your report.


if __name__ == "__main__":
    main()
