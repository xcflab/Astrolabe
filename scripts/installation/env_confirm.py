#!/usr/bin/env python3
"""
Interactive helper to decide what to do with already-existing envs.

Outputs a JSON decisions file usable by `scripts/installation/env_setup.py --decisions <file>`,
mapping env name -> action:
  u = use existing (continue/resume)
  m = mark done (skip all steps)
  r = remove and recreate
  s = skip
"""

import argparse
import json
import os
import pathlib
import sys
import subprocess


def _bootstrap_with_omegaconf(root_prefix, bin_dir):
    """
    Ensure this process can import omegaconf by bootstrapping a tiny micromamba env and re-exec.
    """
    try:
        import omegaconf  # noqa: F401

        return
    except Exception:
        pass

    if os.environ.get("GH_ENV_CONFIRM_BOOTSTRAPPED") == "1":
        raise RuntimeError("Failed to import omegaconf even after bootstrapping.")

    # local imports to avoid adding dependencies
    from shutil import which
    import tarfile
    import urllib.request

    root_prefix = _expand_path(root_prefix)
    bin_dir = _expand_path(bin_dir)
    bin_dir.mkdir(parents=True, exist_ok=True)

    micromamba = which("micromamba")
    if not micromamba:
        url = "https://micro.mamba.pm/api/micromamba/linux-64/latest"
        tar_path = bin_dir / "micromamba.tar.bz2"
        with urllib.request.urlopen(url) as resp:
            tar_path.write_bytes(resp.read())
        with tarfile.open(str(tar_path), "r:bz2") as tf:
            member = None
            for m in tf.getmembers():
                if m.name.endswith("/micromamba"):
                    member = m
                    break
            if member is None:
                raise RuntimeError("micromamba tarball did not contain micromamba binary")
            member.name = pathlib.Path(member.name).name
            tf.extract(member, str(bin_dir))
        tar_path.unlink()
        micromamba = str(bin_dir / "micromamba")
        os.chmod(micromamba, 0o755)

    env = dict(os.environ)
    env["MAMBA_ROOT_PREFIX"] = str(root_prefix)
    bootstrap_env = "_gh_bootstrap"
    bootstrap_prefix = root_prefix / "envs" / bootstrap_env
    if not _env_exists(bootstrap_prefix):
        subprocess.run(
            [
                micromamba,
                "create",
                "-y",
                "-n",
                bootstrap_env,
                "--root-prefix",
                str(root_prefix),
                "-c",
                "conda-forge",
                "python=3.11",
                "pip",
            ],
            env=env,
            check=True,
        )
        subprocess.run(
            [
                micromamba,
                "run",
                "-n",
                bootstrap_env,
                "--root-prefix",
                str(root_prefix),
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
                "omegaconf",
            ],
            env=env,
            check=True,
        )

    cmd = [
        micromamba,
        "run",
        "-n",
        bootstrap_env,
        "--root-prefix",
        str(root_prefix),
        "python",
        str(pathlib.Path(__file__).resolve()),
    ] + sys.argv[1:]
    env["GH_ENV_CONFIRM_BOOTSTRAPPED"] = "1"
    subprocess.run(cmd, env=env, check=True)
    raise SystemExit(0)


def _expand_path(path_str):
    return pathlib.Path(os.path.expandvars(os.path.expanduser(path_str))).resolve()


def _require_omegaconf():
    try:
        from omegaconf import OmegaConf  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "This tool requires `omegaconf` to read the YAML spec. "
            "Run it from an env where Astrolabe is installed (or install omegaconf)."
        ) from exc


def _env_prefix_micromamba(root_prefix, env_name):
    return root_prefix / "envs" / env_name


def _env_exists(prefix):
    return prefix.exists() and any(prefix.iterdir())


def _detect_executable(name):
    from shutil import which

    return which(name)


def _conda_find_env_prefix(conda_exe, env_name, fallback_prefix=None):
    if fallback_prefix:
        p = _expand_path(fallback_prefix)
        if _env_exists(p):
            return p
    if not conda_exe:
        return None
    try:
        out = subprocess.check_output([conda_exe, "env", "list", "--json"], text=True)
        payload = json.loads(out)
        for p in payload.get("envs", []):
            pp = pathlib.Path(p)
            if pp.name == env_name and _env_exists(pp):
                return pp
    except Exception:
        return None
    return None


def _prompt(env_name, prefix):
    prompt = (
        "Env '%s' exists at %s. Choose: [u] use  [m] mark-done  [r] recreate  [s] skip (default: u): "
        % (env_name, prefix)
    )
    while True:
        try:
            resp = input(prompt).strip().lower()
        except EOFError:
            resp = ""
        if not resp:
            return "u"
        if resp in {"u", "m", "r", "s"}:
            return resp
        print("Invalid choice (u/m/r/s).")


def main():
    p = argparse.ArgumentParser(description="Interactive env decision helper.")
    p.add_argument(
        "--spec",
        default=str(pathlib.Path(__file__).resolve().parents[2] / "config" / "third_party_envs.yaml"),
        help="YAML spec file describing environments.",
    )
    p.add_argument(
        "--root-prefix",
        default=str(pathlib.Path.home() / ".gh_envs"),
        help="micromamba root prefix (envs under <root>/envs/<name>).",
    )
    p.add_argument(
        "--bin-dir",
        default=None,
        help="Directory to place/download micromamba if needed (default: <root>/.bin).",
    )
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated env names to include (default: all).",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output JSON path (env_name -> action).",
    )
    args = p.parse_args()

    root_prefix = _expand_path(args.root_prefix)
    bin_dir = _expand_path(args.bin_dir) if args.bin_dir else (root_prefix / ".bin")
    _bootstrap_with_omegaconf(root_prefix=str(root_prefix), bin_dir=str(bin_dir))
    _require_omegaconf()
    from omegaconf import OmegaConf

    spec_path = _expand_path(args.spec)
    out_path = _expand_path(args.out)
    conda_exe = _detect_executable("conda")

    only = None
    if args.only:
        only = set([x.strip() for x in args.only.split(",") if x.strip()])

    spec = OmegaConf.to_container(OmegaConf.load(str(spec_path)), resolve=True)
    envs = spec.get("envs") or []
    decisions = {}
    for env_spec in envs:
        name = env_spec.get("name")
        if not name:
            continue
        if only and name not in only:
            continue
        manager = (env_spec.get("manager") or "micromamba").strip().lower()
        if manager == "micromamba":
            prefix = _env_prefix_micromamba(root_prefix, name)
            exists = _env_exists(prefix)
        elif manager == "conda":
            prefix = _conda_find_env_prefix(conda_exe, name, env_spec.get("prefix"))
            exists = bool(prefix)
        elif manager == "external":
            prefix = None
            if env_spec.get("prefix"):
                prefix = _expand_path(env_spec["prefix"])
            elif env_spec.get("python_bin"):
                py_bin = _expand_path(env_spec["python_bin"])
                prefix = py_bin.parent.parent if py_bin.exists() else None
            exists = bool(prefix and _env_exists(prefix))
        elif manager in {"path", "asset"}:
            prefix = _expand_path(env_spec.get("path") or env_spec.get("prefix") or str(root_prefix / "repos" / name))
            exists = prefix.exists()
        else:
            raise ValueError("Unknown manager for env '%s': %s" % (name, manager))

        if exists:
            decisions[name] = _prompt(name, prefix)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(decisions, f, indent=2, sort_keys=True)
        f.write("\n")
    print("Wrote: %s" % out_path)


if __name__ == "__main__":
    main()
