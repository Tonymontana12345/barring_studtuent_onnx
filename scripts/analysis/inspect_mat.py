"""Print Paderborn recording metadata without plotting or modifying files."""

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from bearing_kd.preprocessing import extract_vibration


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    record = loadmat(args.path, squeeze_me=True, struct_as_record=False)[args.path.stem]
    for item in np.atleast_1d(record.Y):
        print({name: str(getattr(item, name, "")) for name in ("Name", "Unit", "Raster", "Device")})
    signal = extract_vibration(args.path)
    print({"samples": len(signal), "mean": float(signal.mean()), "std": float(signal.std()),
           "rms": float(np.sqrt(np.mean(signal**2)))})


if __name__ == "__main__":
    main()
