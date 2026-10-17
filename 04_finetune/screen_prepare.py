"""Build the data root for the regularisation screen: train on every n-th clip, never touch test.

From a prepared official root (official_ft_prepare.py: videos/, triplet/, labels_pkl/{train,val,test}) it writes
<dst>/labels_pkl with
  train: every n-th sample of each training video (n = --thin; unique ids renumbered). Clips stay intact: the
         official loader builds each clip from frame numbers in the file names, not from neighbouring samples.
  val:   unchanged.
  test:  a COPY OF VAL. The official script always ends with a "final test" of the best checkpoint; in the screen
         that pass scores validation again, so no test video is loaded while configurations are being chosen.
videos/ and triplet/ are links to the source root (no frames copied). Asserts that no test video is in any pickle.

    python 04_finetune/screen_prepare.py --src $WORK/official_ft/CholecT50 --dst $WORK/official_ft/screen/CholecT50 --thin 4
"""
import argparse
import json
import os
import pickle
import shutil
from pathlib import Path


def link_dir(src, dst):
    if dst.exists():
        return
    try:
        os.symlink(src, dst, target_is_directory=True)
    except OSError:  # Windows without symlink rights: directory junction
        import _winapi
        _winapi.CreateJunction(str(src), str(dst))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="prepared official root (official_ft_prepare.py)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--thin", type=int, default=4, help="keep every n-th training sample per video")
    a = ap.parse_args()
    src, dst = Path(a.src).resolve(), Path(a.dst)
    pk = lambda root, s: root / "labels_pkl" / s / f"1fps{s}.pickle"  # noqa: E731
    test_videos = set(pickle.load(open(pk(src, "test"), "rb")))

    for d in ("videos", "triplet"):
        dst.mkdir(parents=True, exist_ok=True)
        link_dir(src / d, dst / d)
    for s in ("train", "val", "test"):
        pk(dst, s).parent.mkdir(parents=True, exist_ok=True)

    train, u = pickle.load(open(pk(src, "train"), "rb")), 0
    n_full = sum(len(v) for v in train.values())
    for k in train:
        train[k] = train[k][::a.thin]
        for f in train[k]:
            f["unique_id"] = u
            u += 1
    pickle.dump(train, open(pk(dst, "train"), "wb"))
    shutil.copyfile(pk(src, "val"), pk(dst, "val"))
    shutil.copyfile(pk(src, "val"), pk(dst, "test"))

    for s in ("train", "val", "test"):
        leaked = set(pickle.load(open(pk(dst, s), "rb"))) & test_videos
        assert not leaked, f"test videos {sorted(leaked)} in the screen's {s} pickle"
    val = pickle.load(open(pk(dst, "val"), "rb"))
    meta = {"src": str(src), "thin": a.thin, "train_samples": u, "train_samples_full": n_full,
            "val_samples": sum(len(v) for v in val.values()), "val_videos": sorted(val),
            "test_pickle": "copy of val (screening never loads test videos)", "test_videos_excluded": sorted(test_videos)}
    (dst / "screen_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"screen root {dst}: train {u} of {n_full} samples (every {a.thin}th), val {meta['val_samples']} "
          f"(videos {meta['val_videos']}), test = val; test videos {meta['test_videos_excluded']} absent", flush=True)


if __name__ == "__main__":
    main()
