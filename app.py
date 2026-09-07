"""Live demo: every request renders the scene through core.py and support.py.

Stdlib HTTP server, no framework. Run `python app.py`; honours PORT and THREADS.
Training stays offline: this only reads demo/checkpoint.pt and demo/initial.npz.
"""
import io
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
from PIL import Image

import core
from support import Scene, camera, render

torch.set_num_threads(int(os.environ.get('THREADS', 1)))
SIZES = (128, 256, 384)  # dense renderer allocates gaussians x pixels; 384 is ~70 MB
DEMO = os.environ.get('DEMO', 'demo')  # folder holding checkpoint.pt and initial.npz


def load_checkpoint(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)['model']
    return Scene((sd['means'], sd['log_scales'].exp(), sd['quats'],
                  sd['opacity_logits'].sigmoid(), sd['color_logits'].sigmoid()))


# Loaded once at startup. Requests render from a copy and never modify these.
with np.load(f'{DEMO}/initial.npz') as d:
    INITIAL = Scene(tuple(d[f'init_{i}'] for i in range(5)))
TRAINED = load_checkpoint(f'{DEMO}/checkpoint.pt')
N = len(TRAINED.means)
# Dense path is gaussians x pixels; past a few hundred use the per-splat bounded loop.
MODE = 'dense' if N <= 300 else 'bounded'
LOCK = threading.Lock()  # ponytail: one render at a time; fine for a demo box


def param(q, key, lo, hi, default, cast=float):
    """Read one query value, clamped into [lo, hi]; bad input falls back to default."""
    try:
        return min(hi, max(lo, cast(q.get(key, [default])[0])))
    except ValueError:
        return default


def render_png(q):
    scene = INITIAL if q.get('scene', [''])[0] == 'initial' else TRAINED
    angle = np.radians(param(q, 'angle', 0, 360, 20))
    elev = param(q, 'elev', -1, 1.5, .35)
    size = min(SIZES, key=lambda s: abs(s - param(q, 'size', 0, 999, 256, int)))
    order = 'reverse' if q.get('order', [''])[0] == 'reverse' else 'depth'
    gi = param(q, 'gi', 0, N - 1, 0, int)
    with torch.no_grad(), LOCK:
        means, scales, quats, opacity, colors = (p.clone() for p in scene.physical())
        if 'iso' in q:
            # Same volume per ellipsoid, forced spherical: shows why anisotropy matters.
            scales = scales.log().mean(-1, keepdim=True).exp().expand_as(scales)
        if 'ovr' in q:
            opacity[gi] = param(q, 'op', 0, 1, .9)
        W, eye = camera(angle, elevation=elev)
        im = render((means, scales, quats, opacity, colors), W, eye, size, core, order=order, mode=MODE)
    buf = io.BytesIO()
    Image.fromarray((im.clamp(0, 1).numpy() * 255).round().astype('uint8')).save(buf, 'PNG')
    return buf.getvalue()


PAGE = f"""<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width">
<title>Gaussian splat demo</title>
<style>
body{{font:15px system-ui;background:#fff;color:#111;margin:2rem auto;max-width:880px;padding:0 1rem;
     display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-start}}
form{{display:grid;gap:.7rem;min-width:280px;flex:1}}
label{{display:flex;justify-content:space-between;gap:1rem;align-items:center}}
img{{width:384px;max-width:100%;aspect-ratio:1;background:#090b10;border-radius:4px}}
small{{color:#666}}
</style>
<form id=f>
<b>{N} Gaussians, rendered live by the training-path renderer</b>
<label>Camera angle <input name=angle type=range min=0 max=360 value=20></label>
<label>Elevation <input name=elev type=range min=-1 max=1.5 step=.05 value=.35></label>
<label>Scene <select name=scene><option value=trained>trained</option>
  <option value=initial>initial (before training)</option></select></label>
<label>Depth order <select name=order><option value=depth>near to far (correct)</option>
  <option value=reverse>far to near (wrong)</option></select></label>
<label>Force spherical, same volume <input name=iso type=checkbox></label>
<label>Gaussian index <input name=gi type=number min=0 max={N - 1} value=0></label>
<label>Override its opacity <input name=ovr type=checkbox></label>
<label>Opacity value <input name=op type=range min=0 max=1 step=.01 value=.9></label>
<label>Image size <select name=size><option>128</option><option selected>256</option>
  <option>384</option></select></label>
<button>Render</button>
<small id=t>Each render runs the same dense PyTorch code used for training, on CPU.</small>
</form>
<img id=out alt="rendered scene">
<script>
const f=document.getElementById('f'),out=document.getElementById('out'),t=document.getElementById('t');
function go(e){{if(e)e.preventDefault();const t0=performance.now();
  out.onload=()=>t.textContent='rendered in '+((performance.now()-t0)/1000).toFixed(2)+' s';
  out.src='/render?'+new URLSearchParams(new FormData(f))}}
f.onsubmit=go;f.onchange=go;go();
</script>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        url = urlparse(self.path)
        if url.path == '/':
            body, ctype = PAGE.encode(), 'text/html; charset=utf-8'
        elif url.path == '/render':
            body, ctype = render_png(parse_qs(url.query)), 'image/png'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    print(f'Serving {N} Gaussians on http://localhost:{port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
