"""Lean image toolbox: HEIC -> PNG conversion, and PNG shrink-to-target-size.

Images are processed in parallel across CPU cores. Both operations are CPU-bound
(HEIF decode, PNG encode), so a process pool gives near-linear speedup on a batch.
"""
import io
import os
import json
import math
import time
import tempfile
import zipfile
from pathlib import Path
from multiprocessing import Pool

from flask import Flask, render_template, request, send_file, abort
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import pillow_heif

pillow_heif.register_heif_opener()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024 * 1024  # 4 GB total upload cap
app.config["TEMPLATES_AUTO_RELOAD"] = True  # pick up index.html edits without a restart

HEIC_EXTS = {".heic", ".heif"}
PNG_EXTS = {".png"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
PREVIEW_MAX = 1600  # longest side of the preview sent to the browser for box drawing

CONVERT_LEVEL = 1   # fast PNG compression for HEIC->PNG (ZIP stores it anyway)
SHRINK_LEVEL = 6    # balanced PNG compression while hunting for target size
MAX_WORKERS = 16    # cap worker processes regardless of core count


def log(msg):
    """Print immediately so progress shows in the terminal live."""
    print(msg, flush=True)


# ── PNG encoding / shrinking (module-level so worker processes can import them) ──

def _encode_png(img, scale, level):
    """Encode img (optionally downscaled by `scale`) as PNG bytes."""
    if scale >= 1.0:
        im = img
    else:
        w = max(1, round(img.width * scale))
        h = max(1, round(img.height * scale))
        im = img.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG", compress_level=level)
    return buf.getvalue()


def _shrink_to_target(img, target_bytes, level=SHRINK_LEVEL):
    """Return (png_bytes, note). Estimate the scale directly instead of blind
    binary search: PNG size ~ pixel area ~ scale^2, so scale *= sqrt(target/size).
    A couple of refinement steps land it under target in ~3 encodes."""
    data = _encode_png(img, 1.0, level)
    if len(data) <= target_bytes:
        return data, "already under target"

    scale = 1.0
    best_under = None
    for _ in range(6):
        # Newton-style step toward the target size.
        scale = max(0.01, min(1.0, scale * (target_bytes / len(data)) ** 0.5))
        data = _encode_png(img, scale, level)
        if len(data) <= target_bytes:
            if best_under is None or len(data) > len(best_under):
                best_under = data
            if len(data) >= target_bytes * 0.88:  # close enough; stop early
                break

    if best_under is not None:
        return best_under, "downscaled to fit"
    return data, "target too small — using smallest possible"


# ── Worker tasks (run in the process pool) ──────────────────────────────────────

def _convert_task(item):
    """(filename, raw_bytes) -> (filename, png_bytes | None, error | None)."""
    filename, raw = item
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:  # not a real/readable image
        return filename, None, str(e)
    return filename, _encode_png(img, 1.0, CONVERT_LEVEL), None


def _shrink_task(item):
    """(filename, raw_bytes, target_bytes, level) -> (filename, png_bytes | None, note)."""
    filename, raw, target_bytes, level = item
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as e:
        return filename, None, str(e)
    data, note = _shrink_to_target(img, target_bytes, level)
    return filename, data, note


def _worker_count(n_items):
    return max(1, min(os.cpu_count() or 1, MAX_WORKERS, n_items))


def _parallel_map(files, prep, worker, workers):
    """Yield worker(prep(f)) for each file, running across `workers` processes.
    Reads file bytes one chunk at a time so memory stays bounded on big batches."""
    if workers <= 1 or len(files) <= 1:
        for f in files:
            yield worker(prep(f))
        return

    with Pool(processes=workers) as pool:
        step = workers * 2  # only this many images held in memory at once
        for start in range(0, len(files), step):
            batch = [prep(f) for f in files[start:start + step]]
            for result in pool.imap(worker, batch):
                yield result


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/convert", methods=["POST"])
def convert():
    files = request.files.getlist("files")
    heic_files = [f for f in files if Path(f.filename).suffix.lower() in HEIC_EXTS]
    if not heic_files:
        abort(400, "No HEIC/HEIF images found in the dropped folder.")

    total = len(heic_files)
    workers = _worker_count(total)
    log(f"\n[convert] {total} HEIC file(s) across {workers} worker(s) — starting…")
    start = time.time()

    def results():
        prep = lambda f: (f.filename, f.read())
        for i, (filename, data, err) in enumerate(
            _parallel_map(heic_files, prep, _convert_task, workers), 1
        ):
            if data is None:
                log(f"[{i}/{total}] SKIP {filename} — not a readable image ({err})")
            else:
                log(f"[{i}/{total}] {filename}  ({len(data) / 1048576:.2f} MB)")
            yield filename, data
        log(f"[convert] done in {time.time() - start:.1f}s — sending ZIP")

    return _build_zip(results(), "converted_png.zip")


@app.route("/shrink", methods=["POST"])
def shrink():
    files = request.files.getlist("files")
    png_files = [f for f in files if Path(f.filename).suffix.lower() in PNG_EXTS]

    try:
        target_mb = float(request.form.get("target_mb", ""))
    except ValueError:
        abort(400, "Please enter a valid target size in MB.")
    if target_mb <= 0:
        abort(400, "Target size must be greater than 0 MB.")
    target_bytes = int(target_mb * 1024 * 1024)

    if not png_files:
        abort(400, "No PNG images found in the dropped folder.")

    total = len(png_files)
    workers = _worker_count(total)
    log(f"\n[shrink] {total} PNG file(s), target {target_mb} MB, across {workers} worker(s) — starting…")
    start = time.time()

    def results():
        prep = lambda f: (f.filename, f.read(), target_bytes, SHRINK_LEVEL)
        for i, (filename, data, note) in enumerate(
            _parallel_map(png_files, prep, _shrink_task, workers), 1
        ):
            if data is None:
                log(f"[{i}/{total}] SKIP {filename} — not a readable image ({note})")
            else:
                log(f"[{i}/{total}] {filename}  {len(data) / 1048576:.2f} MB  {note}")
            yield filename, data
        log(f"[shrink] done in {time.time() - start:.1f}s — sending result")

    if total == 1:
        # A single image comes back as a plain PNG, not a one-file ZIP.
        filename, data = list(results())[0]  # exhaust so the "done" log line runs
        if data is None:
            abort(400, "Could not process the dropped file.")
        return send_file(
            io.BytesIO(data),
            mimetype="image/png",
            as_attachment=True,
            download_name=_unique_name(filename, set()),
        )

    return _build_zip(results(), "shrunk_png.zip")


# ── Plate blurring ───────────────────────────────────────────────────────────────

def _open_photo(raw):
    """Decode any supported photo, apply EXIF rotation, return an RGB image.
    Rotation is baked in so browser preview and server output always agree."""
    img = Image.open(io.BytesIO(raw))
    img = ImageOps.exif_transpose(img)
    img.load()
    return img.convert("RGB")


def _redact_box(img, box, mode):
    """Redact one box in place. Box is {cx, cy, w, h, a}: centre and size as
    0..1 fractions of the image, `a` the tilt in degrees (clockwise, optional)."""
    W, H = img.size
    cx, cy = box["cx"] * W, box["cy"] * H
    bw, bh = box["w"] * W, box["h"] * H
    if bw < 2 or bh < 2:
        return
    theta = math.radians(float(box.get("a", 0)))
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    # Corners of the tilted rectangle in image pixels.
    corners = [
        (cx + dx * cos_t - dy * sin_t, cy + dx * sin_t + dy * cos_t)
        for dx, dy in ((-bw / 2, -bh / 2), (bw / 2, -bh / 2), (bw / 2, bh / 2), (-bw / 2, bh / 2))
    ]
    # Axis-aligned crop that contains the whole tilted box, clipped to the image.
    x0 = max(0, int(math.floor(min(c[0] for c in corners))))
    y0 = max(0, int(math.floor(min(c[1] for c in corners))))
    x1 = min(W, int(math.ceil(max(c[0] for c in corners))))
    y1 = min(H, int(math.ceil(max(c[1] for c in corners))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    region = img.crop((x0, y0, x1, y1))
    rw, rh = region.size

    if mode == "black":
        redacted = Image.new("RGB", (rw, rh), (0, 0, 0))
    elif mode == "blur":
        # Blur radius scales with box size so it is always strong enough to hide text.
        redacted = region.filter(ImageFilter.GaussianBlur(max(10, int(min(bw, bh)) // 4)))
    else:
        # Pixelate: shrink to a handful of cells across the box's short side, scale back.
        cell = max(1, int(min(bw, bh)) // 8)
        small = region.resize((max(1, rw // cell), max(1, rh // cell)), Image.BILINEAR)
        redacted = small.resize((rw, rh), Image.NEAREST)

    # Only the tilted rectangle itself is replaced; the rest of the crop stays.
    mask = Image.new("L", (rw, rh), 0)
    ImageDraw.Draw(mask).polygon([(x - x0, y - y0) for x, y in corners], fill=255)
    img.paste(redacted, (x0, y0), mask)


def _save_photo(img, original_name):
    """Encode as JPEG (photos) or PNG (if the source was PNG). EXIF is dropped,
    which also strips GPS location data from the listing photos."""
    suffix = Path(original_name).suffix.lower()
    buf = io.BytesIO()
    if suffix == ".png":
        img.save(buf, format="PNG", compress_level=CONVERT_LEVEL)
        return buf.getvalue(), ".png", "image/png"
    img.save(buf, format="JPEG", quality=92, subsampling=0)
    return buf.getvalue(), ".jpg", "image/jpeg"


@app.route("/preview", methods=["POST"])
def preview():
    """Return a downscaled JPEG of one uploaded photo for drawing boxes on."""
    f = request.files.get("file")
    if f is None or Path(f.filename).suffix.lower() not in PHOTO_EXTS:
        abort(400, "Unsupported image type.")
    try:
        img = _open_photo(f.read())
    except Exception as e:
        abort(400, f"Could not read {f.filename}: {e}")
    img.thumbnail((PREVIEW_MAX, PREVIEW_MAX))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return send_file(buf, mimetype="image/jpeg")


@app.route("/blur", methods=["POST"])
def blur():
    """Redact user-drawn boxes on each photo. One photo -> image; many -> ZIP."""
    files = request.files.getlist("files")
    photos = [f for f in files if Path(f.filename).suffix.lower() in PHOTO_EXTS]
    if not photos:
        abort(400, "No supported photos uploaded.")

    mode = request.form.get("mode", "pixelate")
    if mode not in {"pixelate", "blur", "black"}:
        abort(400, "Unknown redaction mode.")
    try:
        boxes_per_file = json.loads(request.form.get("boxes", "[]"))
        assert isinstance(boxes_per_file, list) and len(boxes_per_file) == len(photos)
    except (ValueError, AssertionError):
        abort(400, "Box data did not match the uploaded photos.")

    log(f"\n[blur] {len(photos)} photo(s), mode={mode}")
    start = time.time()
    outputs = []
    for i, (f, boxes) in enumerate(zip(photos, boxes_per_file), 1):
        try:
            img = _open_photo(f.read())
        except Exception as e:
            log(f"[{i}/{len(photos)}] SKIP {f.filename} — not a readable image ({e})")
            continue
        for box in boxes:
            _redact_box(img, box, mode)
        data, ext, mime = _save_photo(img, f.filename)
        out_name = Path(f.filename).stem + "_blurred" + ext
        log(f"[{i}/{len(photos)}] {f.filename} -> {out_name}  {len(boxes)} box(es)  {len(data) / 1048576:.2f} MB")
        outputs.append((out_name, data, mime))
    log(f"[blur] done in {time.time() - start:.1f}s")

    if not outputs:
        abort(400, "Could not process any of the photos.")
    if len(outputs) == 1:
        name, data, mime = outputs[0]
        return send_file(io.BytesIO(data), mimetype=mime, as_attachment=True, download_name=name)

    used = set()
    results = []
    for name, data, _ in outputs:
        # _unique_name forces .png; keep the real extension here instead.
        candidate, stem, suffix, n = name, Path(name).stem, Path(name).suffix, 1
        while candidate in used:
            candidate = f"{stem}_{n}{suffix}"; n += 1
        used.add(candidate)
        results.append((candidate, data))
    return _build_zip(results, "blurred_photos.zip", keep_names=True)


# ── ZIP assembly (shared) ────────────────────────────────────────────────────────

def _build_zip(results, download_name, keep_names=False):
    """Stream (filename, bytes|None) results into a temp ZIP on disk.
    Names are forced to .png and de-duplicated unless keep_names is set."""
    tmp = tempfile.NamedTemporaryFile(suffix=".zip", delete=False)
    tmp.close()
    used_names = set()
    written = 0
    try:
        with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_STORED) as zf:
            for filename, data in results:
                if data is None:
                    continue
                name = filename if keep_names else _unique_name(filename, used_names)
                zf.writestr(name, data)
                written += 1

        if written == 0:
            os.unlink(tmp.name)
            abort(400, "Could not process any of the dropped files.")

        resp = send_file(
            tmp.name,
            mimetype="application/zip",
            as_attachment=True,
            download_name=download_name,
        )
        resp.call_on_close(lambda: _quiet_unlink(tmp.name))
        return resp
    except Exception:
        _quiet_unlink(tmp.name)
        raise


def _unique_name(filename, used_names):
    """Original name with a .png extension; add _1, _2… on collision."""
    name = Path(filename).with_suffix(".png").name
    stem = Path(name).stem
    counter = 1
    while name in used_names:
        name = f"{stem}_{counter}.png"
        counter += 1
    used_names.add(name)
    return name


def _quiet_unlink(path):
    try:
        os.unlink(path)
    except OSError:
        pass


if __name__ == "__main__":
    log("Image toolbox running at http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
