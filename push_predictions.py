#!/usr/bin/env python3
"""
Push your predictions to a PRIVATE Hugging Face repo and write the pointer manifest for Codabench.

This is the RECOMMENDED submission route. Instead of uploading ~1.9 GB to Codabench, you push the
.npz to a private Hugging Face dataset repo and submit a small manifest.json naming a pinned
commit plus a read token scoped to that one repo. The scorer pulls from Hugging Face's CDN.

Why it is worth the extra step, measured from the scoring machine:

    Codabench file storage   ~0.5 MB/s   ->  a 1.9 GB submission spends ~1 h in transfer
    Hugging Face CDN         ~50 MB/s    ->  the same predictions arrive in under a minute

The scoring worker runs one job at a time, so that hour is queue time everyone shares.

The repo stays PRIVATE -- your predictions are never visible to other teams.

    uv run python push_predictions.py --repo your-username/fusion-eq-predictions --dry-run
    uv run python push_predictions.py --repo your-username/fusion-eq-predictions

You need TWO tokens, both from https://huggingface.co/settings/tokens :

  1. A WRITE token, to upload. Log in with it once:  uv run huggingface-cli login
     It never leaves your machine.
  2. A FINE-GRAINED READ token scoped to your predictions repo ONLY. This one goes into
     manifest.json and is submitted, so the scorer can read your private repo. Pass it with
     --read-token or set HF_READ_TOKEN.

     New token -> Fine-grained -> under "Repository permissions" pick your predictions repo
     -> tick "Read access to contents of selected repos" -> nothing else.

This script REFUSES a write token or a classic read token for #2, because both grant far more
than the scorer needs and both are stored with your submission. Revoke the read token after the
competition.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG_FILENAMES = [
    "diii_d_public_test.npz", "mast_public_test.npz",
    "diii_d_private_test.npz", "mast_private_test.npz",
]

TOKEN_HELP = (
    "Create one at https://huggingface.co/settings/tokens -> New token -> Fine-grained,\n"
    "       grant ONLY 'Read access to contents of selected repos' on your predictions repo."
)


def validate_read_token(api, token: str, repo: str) -> int:
    """Refuse anything broader than fine-grained read. Returns 0 if acceptable."""
    try:
        who = api.whoami(token=token)
    except Exception as exc:
        print(f"ERROR: the read token was rejected by Hugging Face ({type(exc).__name__}).\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1

    at = (who.get("auth") or {}).get("accessToken") or {}
    role = at.get("role")

    if role in ("write", "admin"):
        print(f"ERROR: that is a {role.upper()} token. It would be stored with your submission and\n"
              f"       could modify your repos. Use a read-only fine-grained token instead.\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1
    if role == "read":
        print("ERROR: that is a classic read token. It grants read access to ALL of your private\n"
              "       repos, not just your predictions, and it is stored with your submission.\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1
    if role == "fineGrained":
        scoped = (at.get("fineGrained") or {}).get("scoped") or []
        perms = [p for s in scoped for p in (s.get("permissions") or [])]
        bad = sorted({p for p in perms if "write" in p or "admin" in p})
        if bad:
            print(f"ERROR: that fine-grained token grants {bad}. Read access is all the scorer\n"
                  f"       needs, and the token is stored with your submission.\n"
                  f"       {TOKEN_HELP}", file=sys.stderr)
            return 1
        names = {(s.get("entity") or {}).get("name") for s in scoped}
        if repo not in names and repo.split("/")[0] not in names:
            print(f"WARNING: the token is scoped to {sorted(n for n in names if n)}, which does not\n"
                  f"         obviously include {repo}. Scoring will fail if it cannot read that repo.")
        elif len(scoped) > 1:
            print(f"NOTE: the token is scoped to {len(scoped)} entities; just the predictions repo "
                  "would be enough.")
        return 0

    print(f"NOTE: could not determine the token's scope (role={role!r}). Continuing -- but please\n"
          "      confirm it is a fine-grained, read-only token for your predictions repo.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True,
                    help="Hugging Face dataset repo, e.g. your-username/fusion-eq-predictions")
    ap.add_argument("--read-token", default=os.environ.get("HF_READ_TOKEN"),
                    help="fine-grained READ token scoped to --repo (or set HF_READ_TOKEN)")
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

    if not args.read_token:
        print("\nERROR: no read token. Pass --read-token or set HF_READ_TOKEN.\n"
              "       This is the token the scorer uses to read your PRIVATE predictions repo;\n"
              "       it is stored in manifest.json and submitted to Codabench.\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1

    api = HfApi()
    if validate_read_token(api, args.read_token, args.repo):
        return 1
    print("Read token accepted (fine-grained, read-only).")

    if args.dry_run:
        print(f"\n--dry-run: would create {args.repo} (PRIVATE dataset) and upload the files above.")
        return 0

    who = api.whoami()
    print(f"\nLogged in as {who['name']}. Creating/reusing PRIVATE dataset repo {args.repo}...")
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)

    # exist_ok=True does NOT flip an existing public repo to private. Say so loudly rather than
    # leaving a team to discover their predictions were readable all along.
    info = api.repo_info(args.repo, repo_type="dataset")
    if not getattr(info, "private", True):
        print(f"\nWARNING: {args.repo} already existed and is PUBLIC. Your predictions will be\n"
              f"         visible to other teams. Scoring still works, but consider making it\n"
              f"         private in the repo settings, or using a fresh --repo name.\n")

    print("Uploading (this is the slow part, but it only happens once per submission)...")
    commit = api.upload_folder(folder_path=str(args.dir), repo_id=args.repo,
                               repo_type="dataset", allow_patterns=["*.npz"])
    sha = commit.oid
    if not (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha.lower())):
        print(f"ERROR: expected a 40-character commit SHA, got {sha!r}", file=sys.stderr)
        return 1

    # Verify with the READ token exactly what the scorer will see, before a submission slot is
    # spent on it. This catches a token scoped to the wrong repo, which is the likely mistake.
    try:
        at_rev = api.repo_info(args.repo, revision=sha, repo_type="dataset",
                              token=args.read_token)
    except RepositoryNotFoundError:
        print(f"ERROR: the upload succeeded, but the READ token cannot see {args.repo}.\n"
              f"       Re-scope it to that repo -- otherwise scoring will fail.\n"
              f"       {TOKEN_HELP}", file=sys.stderr)
        return 1
    root = {s.rfilename for s in at_rev.siblings if "/" not in s.rfilename}
    missing = [p.name for p in npz if p.name not in root]
    if missing:
        print(f"ERROR: uploaded, but these are not at the repo ROOT at {sha}: {missing}\n"
              f"       Files found at root: {sorted(root)}", file=sys.stderr)
        return 1

    manifest = {"repo_id": args.repo, "revision": sha, "token": args.read_token}
    mpath = args.dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2))
    args.out.unlink(missing_ok=True)   # zip appends to an existing archive otherwise
    subprocess.run(["zip", "-j", "-q", str(args.out), str(mpath)], check=True)

    print(f"\nPushed {len(npz)} file(s) at commit {sha}")
    print(f"Verified with the read token, at the repo root: "
          f"{', '.join(sorted(p.name for p in npz))}")
    print(f"\nWrote {args.out.resolve()} ({args.out.stat().st_size} bytes) containing:")
    print(json.dumps({**manifest, "token": "hf_***redacted***"}, indent=2))
    print(f"\nUpload {args.out.name} to Codabench. That is the whole submission -- the scorer\n"
          f"pulls the predictions from {args.repo} at the pinned commit.")
    print(f"\nNote that {mpath} now contains your read token. It is meant to be submitted, but do\n"
          "not commit it to a public git repo, and revoke it after the competition.")
    print("\nYou can keep pushing new predictions to this same repo; re-run this script and each\n"
          "submission names the exact commit it was built from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
