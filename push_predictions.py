#!/usr/bin/env python3
"""
Push your predictions to Hugging Face and write the pointer manifest Codabench wants.

This is the RECOMMENDED submission route. Instead of uploading ~1.9 GB to Codabench, you push
the .npz to a public Hugging Face dataset repo and submit a ~200-byte manifest.json naming a
pinned commit; the scorer pulls from Hugging Face's CDN.

Why it is worth the extra step, measured from the scoring machine:

    Codabench file storage   ~0.5 MB/s   ->  a 1.9 GB submission spends ~1 h in transfer
    Hugging Face CDN         ~50 MB/s    ->  the same predictions arrive in under a minute

The scoring worker runs one job at a time, so that hour is queue time everyone shares.

    uv run python push_predictions.py --repo your-username/fusion-eq-predictions
    uv run python push_predictions.py --repo your-username/fusion-eq-predictions --dry-run

You need to be logged in once:  uv run huggingface-cli login   (a write token)

The repo MUST be public: the scoring container authenticates as nobody and cannot read a private
or gated repo. That means other teams can see your predictions. Publishing your own work is fine
and carries no penalty -- but submitting predictions you did not produce is plagiarism and is
disqualifying for the whole team. See the competition Rules page.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

CONFIG_FILENAMES = [
    "diii_d_public_test.npz", "mast_public_test.npz",
    "diii_d_private_test.npz", "mast_private_test.npz",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="Hugging Face dataset repo, e.g. your-username/fusion-eq-predictions")
    ap.add_argument("--dir", type=Path, default=Path("submission"),
                    help="directory holding the .npz (default: submission)")
    ap.add_argument("--out", type=Path, default=Path("submission_pointer.zip"),
                    help="pointer zip to upload to Codabench (default: submission_pointer.zip)")
    ap.add_argument("--dry-run", action="store_true", help="check everything, upload nothing")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from huggingface_hub.errors import RepositoryNotFoundError

    npz = sorted(p for p in args.dir.glob("*.npz") if p.name in CONFIG_FILENAMES)
    if not npz:
        print(f"ERROR: no recognised .npz in {args.dir.resolve()}.\n"
              f"       Expected one or more of: {', '.join(CONFIG_FILENAMES)}\n"
              f"       Build them first: python submission_skeleton.py --out {args.dir} --max-shots 0",
              file=sys.stderr)
        return 1

    total = sum(p.stat().st_size for p in npz)
    print(f"Found {len(npz)} file(s) in {args.dir.resolve()}  ({total / 1e6:.0f} MB total):")
    for p in npz:
        print(f"    {p.name:28s} {p.stat().st_size / 1e6:8.0f} MB")
    if total > 3e9:
        print("\n  NOTE: that is larger than a float16 submission should be (~1.9 GB for both\n"
              "        machines). If you wrote psirz as float32, rebuild -- float32 roughly\n"
              "        doubles the size and costs ~0.1% of score.")

    api = HfApi()
    if args.dry_run:
        print(f"\n--dry-run: would create {args.repo} (public dataset) and upload the files above.")
        return 0

    who = api.whoami()
    print(f"\nLogged in as {who['name']}. Creating/reusing dataset repo {args.repo} (public)...")
    api.create_repo(args.repo, repo_type="dataset", private=False, exist_ok=True)

    # Refuse to continue against a repo that already exists and is private -- exist_ok=True will
    # NOT flip an existing private repo to public, and a private repo silently fails at scoring.
    info = api.repo_info(args.repo, repo_type="dataset")
    if getattr(info, "private", False):
        print(f"\nERROR: {args.repo} already exists and is PRIVATE. The scoring container cannot\n"
              f"       read it. Make it public in the repo settings, or pick a new --repo name.",
              file=sys.stderr)
        return 1

    print("Uploading (this is the slow part, but it only happens once per submission)...")
    commit = api.upload_folder(folder_path=str(args.dir), repo_id=args.repo,
                               repo_type="dataset", allow_patterns=["*.npz"])
    sha = commit.oid
    if not (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha.lower())):
        print(f"ERROR: expected a 40-character commit SHA, got {sha!r}", file=sys.stderr)
        return 1

    # Verify from the hub what the scorer will see, before you spend a submission slot on it.
    try:
        at_rev = api.repo_info(args.repo, revision=sha, repo_type="dataset", files_metadata=False)
    except RepositoryNotFoundError:
        print(f"ERROR: {args.repo} vanished between upload and verification.", file=sys.stderr)
        return 1
    root = {s.rfilename for s in at_rev.siblings if "/" not in s.rfilename}
    missing = [p.name for p in npz if p.name not in root]
    if missing:
        print(f"ERROR: uploaded, but these are not at the repo ROOT at {sha}: {missing}\n"
              f"       Files found at root: {sorted(root)}", file=sys.stderr)
        return 1

    manifest = {"repo_id": args.repo, "revision": sha}
    mpath = args.dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    subprocess.run(["zip", "-j", "-q", str(args.out), str(mpath)], check=True)

    print(f"\nPushed {len(npz)} file(s) at commit {sha}")
    print(f"Verified at the repo root: {', '.join(sorted(p.name for p in npz))}")
    print(f"\nWrote {args.out.resolve()} ({args.out.stat().st_size} bytes) containing:")
    print(json.dumps(manifest, indent=2))
    print(f"\nUpload {args.out.name} to Codabench. That is the whole submission -- the scorer\n"
          f"pulls the predictions from {args.repo} at the pinned commit.")
    print("\nYou can keep pushing new predictions to this same repo; re-run this script and each\n"
          "submission names the exact commit it was built from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
