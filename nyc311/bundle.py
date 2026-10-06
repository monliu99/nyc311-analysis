import zipfile

MAX_FILE_MB = 50


def zip_outputs(cfg):
    """Zip the small outputs folder next to it, for downloading from the cluster."""
    out = cfg.out_dir
    dest = out.parent / f"{out.name}_bundle.zip"
    skipped = []
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob("*")):
            if not p.is_file():
                continue
            if p.stat().st_size > MAX_FILE_MB * 1e6:
                skipped.append(p.name)
                continue
            z.write(p, f"outputs/{p.relative_to(out).as_posix()}")
    print(f"Download this file: {dest.resolve()} ({dest.stat().st_size / 1e6:,.1f} MB)")
    if skipped:
        print(f"Skipped files over {MAX_FILE_MB} MB: {skipped}")
    return dest
