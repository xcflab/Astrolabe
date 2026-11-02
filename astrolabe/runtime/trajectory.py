"""Execute a single existing trajectory in a fresh Python interpreter."""
import sys
from omegaconf import OmegaConf


def main():
    from astrolabe.app import _load_run_module
    cfg = OmegaConf.load(sys.argv[1])
    run = _load_run_module()
    run.global_config.reset()
    run._run_pipeline(cfg)


if __name__ == '__main__':
    main()
