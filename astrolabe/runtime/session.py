"""Conservative process isolation for the existing repeat lifecycle."""
import os
import pathlib
import subprocess
import sys
import tempfile
import socket
import time
import uuid
import importlib
from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from omegaconf import OmegaConf


def _child_env():
    env = os.environ.copy()
    roots = [str(pathlib.Path(p).resolve()) for p in sys.path if p]
    env["PYTHONPATH"] = os.pathsep.join(roots + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


class BrokerSession:
    def __init__(self, max_resident_models=1):
        if type(max_resident_models) is not int or max_resident_models < 1:
            raise ValueError('max_resident_models must be a positive integer')
        self.max_resident_models = max_resident_models

    def __enter__(self):
        self.directory = tempfile.TemporaryDirectory(prefix='astrolabe-session-')
        self.endpoint = '@astrolabe-' + uuid.uuid4().hex
        self.token = uuid.uuid4().hex + uuid.uuid4().hex
        read_fd, self.write_fd = os.pipe()
        token_read, token_write = os.pipe()
        self.env = _child_env()
        # A broker must never accidentally connect to a parent broker.
        for key in ('ASTROLABE_EXECUTION_ENDPOINT', 'ASTROLABE_EXECUTION_TOKEN'):
            self.env.pop(key, None)
        self.env['ASTROLABE_EXECUTION_ARTIFACT_ROOT'] = self.directory.name
        try:
            self.process = subprocess.Popen(
                [sys.executable, '-m', 'astrolabe.runtime.broker', self.endpoint,
                 str(token_read), str(read_fd), self.directory.name, str(self.max_resident_models)], env=self.env,
                pass_fds=(read_fd, token_read))
        except BaseException:
            os.close(read_fd)
            os.close(self.write_fd)
            os.close(token_read)
            os.close(token_write)
            self.directory.cleanup()
            raise
        os.close(read_fd)
        os.close(token_read)
        try:
            os.write(token_write, self.token.encode())
        except BaseException:
            os.close(token_write)
            self.__exit__(None, None, None)
            raise
        else:
            os.close(token_write)
        try:
            deadline = time.monotonic() + 15
            while True:
                if self.process.poll() is not None:
                    raise RuntimeError('execution broker exited during startup')
                try:
                    with socket.socket(socket.AF_UNIX) as probe:
                        probe.connect('\0' + self.endpoint[1:])
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError('execution broker startup timed out')
                    time.sleep(.02)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self.env.update(ASTROLABE_EXECUTION_ENDPOINT=self.endpoint,
                        ASTROLABE_EXECUTION_TOKEN=self.token,
                        ASTROLABE_EXECUTION_ARTIFACT_ROOT=self.directory.name)
        return self

    def __exit__(self, *_):
        os.close(self.write_fd)
        # Broker holds in-flight leases until workers terminate; never remove
        # its artifact root while it can still be using request files.
        self.process.wait()
        self.directory.cleanup()


def validate_parallel(cfg):
    """Reject any component that has not explicitly opted into this path."""
    if cfg.get('structure_start') is not None:
        raise ValueError('parallel runtime does not support structure_start')
    initializer = cfg.get('initializer') or {}
    if initializer.get('mode', 'random') not in ('random', 'all_a', 'aa_set', 'pssm', 'mask_mix', 'x_mix'):
        raise ValueError('parallel runtime requires a CPU-only initializer')
    if (cfg.get('predocker') or {}).get('predocker_terms'):
        raise ValueError('parallel runtime does not support predockers')
    if (cfg.get('selector') or {}).get('selector_terms'):
        raise ValueError('parallel runtime does not support custom selectors')
    if (cfg.get('per_residue_scores') or {}).get('enabled', False):
        raise ValueError('parallel runtime does not support per-residue plugins')
    if cfg.get('pre_eval_hooks') or cfg.get('post_eval_hooks'):
        raise ValueError('parallel runtime does not support evaluation hooks')
    if (cfg.get('evaluator') or {}).get('pre_eval_hooks'):
        raise ValueError('parallel runtime does not support evaluation hooks')
    stages = cfg.get('stages') or {}
    allowed_losses = {'sequence_residue_fraction', 'sequence_region_identity',
                      'sequence_edge_region_identity', 'plddt', 'ptm', 'iptm', 'pae', 'pde'}
    for alias, term in ((cfg.get('loss') or {}).get('loss_terms') or {}).items():
        if term and str(term.get('_type', '')).lower() not in allowed_losses:
            raise ValueError(f'parallel runtime does not support loss {alias}')
    evaluators = (cfg.get('evaluator') or {}).get('evaluator_terms') or {}
    mutators = (cfg.get('mutator') or {}).get('mutator_terms') or {}
    default_stage = stages.get('default') or {}
    for stage in stages.values():
        if not stage:
            continue
        for alias in stage.get('mutators') or default_stage.get('mutators') or []:
            term = mutators.get(alias) or {}
            if str(term.get('_type', '')).lower() != 'mcmc':
                raise ValueError(f'parallel runtime does not support mutator {alias}')
        aliases = [stage.get('evaluator') or default_stage.get('evaluator')]
        gate = stage.get('transition_consistency_gate') or {}
        if gate.get('evaluator'):
            aliases.append(gate['evaluator'])
        for alias in aliases:
            term = evaluators.get(alias) or {}
            typ = str(term.get('_type', '')).lower()
            if typ == 'dummy':
                continue
            if typ == 'alphafold3':
                mode = term.get('execution', (cfg.get('evaluator') or {}).get('alphafold3', {}).get('execution', 'legacy'))
                if mode == 'daemon':
                    try:
                        mod = importlib.import_module('astrolabe.evaluators.af3_resident')
                        if (getattr(mod, 'BROKER_SHARED_ROOT', False) is True and
                            getattr(mod, 'BROKER_ADMITTED_PROBE', False) is True):
                            continue
                    except ImportError:
                        pass
            raise ValueError(f'parallel runtime does not support evaluator {alias}')


def run_parallel(configs, max_active, stop_requested, on_completed, *, max_resident_models=1):
    """Conservative online admission, increasing on completions and reducing under CPU pressure."""
    remaining = iter(configs)
    exhausted = False
    limit = min(2, max_active)
    active = {}
    completed = 0
    with BrokerSession(max_resident_models) as session, ThreadPoolExecutor(max_workers=max_active) as pool:
        while not exhausted or active:
            while not exhausted and len(active) < limit and not stop_requested():
                try:
                    cfg = next(remaining)
                except StopIteration:
                    exhausted = True
                    break
                future = pool.submit(run_trajectory, cfg, env=session.env)
                active[future] = cfg
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                active.pop(future)
                future.result()
                completed += 1
                on_completed(completed)
            cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
            try:
                busy = os.getloadavg()[0] >= cpus
            except OSError:
                busy = True
            limit = max(1, limit - 1) if busy else min(max_active, limit + 1)
    return completed


def run_trajectory(cfg, *, env=None):
    with tempfile.TemporaryDirectory(prefix='astrolabe-trajectory-') as tmp:
        config_path = pathlib.Path(tmp) / 'config.yaml'
        OmegaConf.save(cfg, config_path, resolve=False)
        subprocess.run([sys.executable, '-m', 'astrolabe.runtime.trajectory', str(config_path)],
                       check=True, env=env or _child_env())
