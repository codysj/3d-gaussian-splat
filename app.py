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
SIZES = (256, 384, 512)  # dense path allocates gaussians x pixels; 512 x 31 is ~130 MB
DEMO = os.environ.get('DEMO', 'demo')  # one sub-folder per scene: checkpoint.pt + initial.npz


def load_checkpoint(path):
    sd = torch.load(path, map_location='cpu', weights_only=True)['model']
    return Scene((sd['means'], sd['log_scales'].exp(), sd['quats'],
                  sd['opacity_logits'].sigmoid(), sd['color_logits'].sigmoid()))


# Loaded once at startup. Requests render from a copy and never modify these.
SCENES = {}
for name in sorted(os.listdir(DEMO)):
    folder = f'{DEMO}/{name}'
    if not os.path.isfile(f'{folder}/checkpoint.pt'):
        continue
    with np.load(f'{folder}/initial.npz') as d:
        initial = Scene(tuple(d[f'init_{i}'] for i in range(5)))
    trained = load_checkpoint(f'{folder}/checkpoint.pt')
    n = len(trained.means)
    # Dense path is gaussians x pixels; past a few hundred use the per-splat bounded loop.
    SCENES[name] = dict(initial=initial, trained=trained, n=n, mode='dense' if n <= 300 else 'bounded')
DEFAULT = 'gnome' if 'gnome' in SCENES else next(iter(SCENES))
LOCK = threading.Lock()  # ponytail: one render at a time; fine for a demo box


def param(q, key, lo, hi, default, cast=float):
    """Read one query value, clamped into [lo, hi]; bad input falls back to default."""
    try:
        return min(hi, max(lo, cast(q.get(key, [default])[0])))
    except ValueError:
        return default


def render_png(q):
    model = SCENES.get(q.get('model', [''])[0], SCENES[DEFAULT])
    scene = model['initial'] if q.get('scene', [''])[0] == 'initial' else model['trained']
    angle = np.radians(param(q, 'angle', 0, 360, 20))
    elev = param(q, 'elev', -1, 1.5, .35)
    size = min(SIZES, key=lambda s: abs(s - param(q, 'size', 0, 999, 384, int)))
    order = 'reverse' if q.get('order', [''])[0] == 'reverse' else 'depth'
    gi = param(q, 'gi', 0, model['n'] - 1, 0, int)
    with torch.no_grad(), LOCK:
        means, scales, quats, opacity, colors = (p.clone() for p in scene.physical())
        if 'iso' in q:
            # Same volume per ellipsoid, forced spherical: shows why anisotropy matters.
            scales = scales.log().mean(-1, keepdim=True).exp().expand_as(scales)
        if 'ovr' in q:
            opacity[gi] = param(q, 'op', 0, 1, .9)
        W, eye = camera(angle, elevation=elev)
        im = render((means, scales, quats, opacity, colors), W, eye, size, core, order=order, mode=model['mode'])
    buf = io.BytesIO()
    Image.fromarray((im.clamp(0, 1).numpy() * 255).round().astype('uint8')).save(buf, 'PNG')
    return buf.getvalue()


# page.html is static apart from the model dropdown, filled in once here.
OPTIONS = ''.join(f'<option value="{k}" data-n="{v["n"]}"{" selected" if k == DEFAULT else ""}>'
                  f'{k} ({v["n"]:,} Gaussians)</option>' for k, v in SCENES.items())
PAGE = open('page.html', encoding='utf-8').read().replace('<!--OPTIONS-->', OPTIONS)


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
    print(f'Serving {list(SCENES)} on http://localhost:{port}', flush=True)
    ThreadingHTTPServer(('0.0.0.0', port), Handler).serve_forever()
