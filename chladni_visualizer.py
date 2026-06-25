#!/usr/bin/env python3
"""
Chladni Plate Live Visualizer

Real-time 3D visualization of audio input from a function generator
driving physical Chladni plates. Displays a rolling waterfall spectrogram
and the predicted nodal pattern for the detected vibration mode.

Controls (keyboard while PyVista window is focused):
    S       Switch to square plate
    C       Switch to circular plate
    +/-     Adjust reference fundamental frequency ±5 Hz
    A       Auto-lock fundamental to current peak frequency
    Q       Quit

Usage:
    pip install pyvista sounddevice numpy scipy
    python chladni_visualizer.py
"""

import sys
import queue
import numpy as np
import sounddevice as sd
from scipy.special import jn, jn_zeros
import pyvista as pv

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
BLOCK_SIZE = 4096          # ~93 ms per block → ~10.7 Hz frequency resolution
NOISE_FLOOR = 0.005        # minimum spectral peak to trigger mode matching

# ── Display ──────────────────────────────────────────────────────────────────
MAX_FREQ_HZ = 5000         # Chladni modes typically < 5 kHz
WATERFALL_ROWS = 80        # number of FFT frames kept in the waterfall
PATTERN_RES = 300          # grid resolution for the Chladni surface
UPDATE_MS = 45             # callback interval → ~22 fps
MAX_MODE_ORDER = 8         # highest mode index to consider
PATTERN_HEIGHT = 0.18      # Z-scale of the pattern surface relative to plate

# ── Derived ──────────────────────────────────────────────────────────────────
FREQ_BIN_HZ = SAMPLE_RATE / BLOCK_SIZE
N_BINS = int(MAX_FREQ_HZ / FREQ_BIN_HZ)
FREQ_AXIS = np.fft.rfftfreq(BLOCK_SIZE, 1.0 / SAMPLE_RATE)[:N_BINS]


class ChladniVisualizer:

    def __init__(self):
        self.audio_q: queue.Queue = queue.Queue(maxsize=40)

        self.waterfall = np.full((WATERFALL_ROWS, N_BINS), -80.0)
        self.peak_freq = 0.0
        self.peak_amp = 0.0

        self.plate = "square"
        self.mode_n = 1
        self.mode_m = 1
        self.f_ref = 200.0

        self._pattern_cache: dict = {}
        self._build_mode_tables()

    # ── mode tables ──────────────────────────────────────────────────────────

    def _build_mode_tables(self):
        sq = []
        for n in range(MAX_MODE_ORDER + 1):
            for m in range(n, MAX_MODE_ORDER + 1):
                if n == 0 and m == 0:
                    continue
                sq.append((n, m, (n * n + m * m) / 2.0))
        self.sq_modes = sorted(sq, key=lambda t: t[2])

        k_ref = jn_zeros(0, 1)[0] ** 2
        ci = []
        for n in range(MAX_MODE_ORDER + 1):
            zeros = jn_zeros(n, MAX_MODE_ORDER)
            for mi, k in enumerate(zeros):
                ci.append((n, mi + 1, k * k / k_ref))
        self.circ_modes = sorted(ci, key=lambda t: t[2])

    # ── audio ────────────────────────────────────────────────────────────────

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            print(f"  audio: {status}", file=sys.stderr)
        try:
            self.audio_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _process_audio(self) -> bool:
        got_data = False
        while not self.audio_q.empty():
            try:
                block = self.audio_q.get_nowait()
            except queue.Empty:
                break

            window = np.hanning(len(block))
            spec = np.abs(np.fft.rfft(block * window))[:N_BINS]
            self.waterfall = np.roll(self.waterfall, -1, axis=0)
            self.waterfall[-1] = 20.0 * np.log10(spec + 1e-10)

            if spec.max() > NOISE_FLOOR:
                pk = int(np.argmax(spec[1:])) + 1
                if 1 < pk < len(spec) - 1:
                    a = np.log(spec[pk - 1] + 1e-10)
                    b = np.log(spec[pk] + 1e-10)
                    g = np.log(spec[pk + 1] + 1e-10)
                    denom = a - 2.0 * b + g
                    delta = 0.5 * (a - g) / denom if abs(denom) > 1e-12 else 0.0
                    self.peak_freq = (pk + delta) * FREQ_BIN_HZ
                else:
                    self.peak_freq = pk * FREQ_BIN_HZ
                self.peak_amp = spec[pk]
            got_data = True

        if got_data:
            self._match_mode()
        return got_data

    # ── mode matching ────────────────────────────────────────────────────────

    def _match_mode(self):
        if self.peak_freq < 20.0 or self.f_ref < 1.0 or self.peak_amp < NOISE_FLOOR:
            return
        ratio = self.peak_freq / self.f_ref
        modes = self.sq_modes if self.plate == "square" else self.circ_modes
        best = min(modes, key=lambda t: abs(ratio - t[2]))
        self.mode_n, self.mode_m = best[0], best[1]

    # ── Chladni patterns ─────────────────────────────────────────────────────

    def _chladni(self, n: int, m: int):
        key = (self.plate, n, m)
        if key in self._pattern_cache:
            return self._pattern_cache[key]

        res = PATTERN_RES
        if self.plate == "square":
            lin = np.linspace(0, 1, res)
            X, Y = np.meshgrid(lin, lin)
            Z = (np.cos(n * np.pi * X) * np.cos(m * np.pi * Y)
                 + np.cos(m * np.pi * X) * np.cos(n * np.pi * Y))
        else:
            r = np.linspace(0, 1, res)
            th = np.linspace(0, 2.0 * np.pi, res)
            R, Th = np.meshgrid(r, th)
            k_nm = jn_zeros(n, m)[-1]
            Z = jn(n, k_nm * R) * np.cos(n * Th)
            X = R * np.cos(Th)
            Y = R * np.sin(Th)

        zmax = np.max(np.abs(Z))
        if zmax > 0:
            Z /= zmax
        self._pattern_cache[key] = (X, Y, Z)
        return X, Y, Z

    # ── main loop ────────────────────────────────────────────────────────────

    def run(self):
        pv.global_theme.background = "#0d1117"
        pv.global_theme.font.color = "white"
        pv.global_theme.font.size = 13

        pl = pv.Plotter(
            shape=(1, 2),
            title="Chladni Plate Live Visualizer",
            window_size=(1920, 900),
            border=False,
        )

        # ────────────── left: waterfall spectrogram ──────────────────────────
        pl.subplot(0, 0)

        f_kHz = FREQ_AXIS / 1000.0
        t_sec = np.linspace(
            0, WATERFALL_ROWS * BLOCK_SIZE / SAMPLE_RATE, WATERFALL_ROWS
        )
        F, T = np.meshgrid(f_kHz, t_sec)
        wf_mesh = pv.StructuredGrid(F, T, np.zeros_like(F))
        wf_mesh.point_data["dB"] = self.waterfall.flatten()

        pl.add_mesh(
            wf_mesh,
            scalars="dB",
            cmap="inferno",
            clim=[-80, 0],
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "dB",
                "position_x": 0.87,
                "width": 0.06,
                "height": 0.4,
                "fmt": "%.0f",
            },
            smooth_shading=True,
            lighting=True,
        )
        pl.add_text(
            "Live Frequency Spectrum",
            position="upper_left",
            font_size=13,
            shadow=True,
        )
        pl.add_text(
            "Peak: — Hz",
            position="lower_left",
            font_size=11,
            name="freq_lbl",
        )
        pl.add_axes(
            xlabel="Freq (kHz)",
            ylabel="Time (s)",
            zlabel="Amp",
            line_width=2,
        )
        pl.camera_position = [
            (3.0, -2.0, 2.0),
            (np.median(f_kHz), np.median(t_sec), 0.15),
            (0, 0, 1),
        ]

        # ────────────── right: Chladni pattern ───────────────────────────────
        pl.subplot(0, 1)

        X0, Y0, Z0 = self._chladni(1, 1)
        pat_mesh = pv.StructuredGrid(X0, Y0, Z0 * PATTERN_HEIGHT)
        pat_mesh.point_data["disp"] = Z0.flatten()
        # "sand density": bright where |Z|≈0 (nodal lines)
        pat_mesh.point_data["sand"] = (1.0 - np.abs(Z0)).flatten()

        pl.add_mesh(
            pat_mesh,
            scalars="disp",
            cmap="coolwarm",
            clim=[-1, 1],
            show_scalar_bar=True,
            scalar_bar_args={
                "title": "Displacement",
                "position_x": 0.87,
                "width": 0.06,
                "height": 0.4,
                "fmt": "%.1f",
            },
            smooth_shading=True,
            lighting=True,
            name="chladni_surface",
        )
        pl.add_text(
            "Chladni Pattern",
            position="upper_left",
            font_size=13,
            shadow=True,
        )
        pl.add_text(
            "Mode (1, 1) | SQUARE",
            position="lower_left",
            font_size=11,
            name="mode_lbl",
        )
        pl.add_text(
            "Ref: 200 Hz  |  [S]quare  [C]ircular  [+/-] Ref  [A]uto",
            position="lower_right",
            font_size=9,
            name="ctrl_lbl",
        )
        pl.add_axes(xlabel="X", ylabel="Y", zlabel="Z", line_width=2)

        cx, cy = (0.5, 0.5) if self.plate == "square" else (0.0, 0.0)
        pl.camera_position = [
            (cx + 1.3, cy - 1.0, 0.9),
            (cx, cy, 0.0),
            (0, 0, 1),
        ]

        # ────────────── keyboard controls ────────────────────────────────────
        def _set_sq():
            self.plate = "square"

        def _set_ci():
            self.plate = "circular"

        def _ref_up():
            self.f_ref += 5.0

        def _ref_dn():
            self.f_ref = max(10.0, self.f_ref - 5.0)

        def _auto():
            if self.peak_freq > 20:
                self.f_ref = self.peak_freq

        pl.add_key_event("s", _set_sq)
        pl.add_key_event("c", _set_ci)
        pl.add_key_event("plus", _ref_up)
        pl.add_key_event("minus", _ref_dn)
        pl.add_key_event("a", _auto)

        # ────────────── render callback ──────────────────────────────────────
        prev_mode = [None]

        def tick(step):
            self._process_audio()

            # — waterfall update —
            pl.subplot(0, 0)
            z_height = np.clip((self.waterfall + 80.0) / 80.0, 0, 1) * 0.5
            pts = wf_mesh.points.copy()
            pts[:, 2] = z_height.flatten()
            wf_mesh.points = pts
            wf_mesh.point_data["dB"] = self.waterfall.flatten()

            pl.add_text(
                f"Peak: {self.peak_freq:.1f} Hz  ({self.peak_freq / 1000:.2f} kHz)",
                position="lower_left",
                font_size=11,
                name="freq_lbl",
            )

            # — pattern update (only on mode change) —
            mk = (self.plate, self.mode_n, self.mode_m)
            if mk != prev_mode[0]:
                prev_mode[0] = mk
                pl.subplot(0, 1)

                X, Y, Z = self._chladni(self.mode_n, self.mode_m)
                Zs = Z * PATTERN_HEIGHT

                pat_mesh.points = np.column_stack(
                    [X.flatten(), Y.flatten(), Zs.flatten()]
                )
                pat_mesh.point_data["disp"] = Z.flatten()
                pat_mesh.point_data["sand"] = (1.0 - np.abs(Z)).flatten()

                pl.add_text(
                    f"Mode ({self.mode_n}, {self.mode_m})  |  {self.plate.upper()}",
                    position="lower_left",
                    font_size=11,
                    name="mode_lbl",
                )
                pl.add_text(
                    f"Ref: {self.f_ref:.0f} Hz  |  "
                    f"[S]quare  [C]ircular  [+/-] Ref  [A]uto",
                    position="lower_right",
                    font_size=9,
                    name="ctrl_lbl",
                )

                # re-center camera for shape change
                if self.plate == "circular":
                    pl.camera_position = [
                        (1.3, -1.0, 0.9),
                        (0.0, 0.0, 0.0),
                        (0, 0, 1),
                    ]
                else:
                    pl.camera_position = [
                        (1.8, -0.5, 0.9),
                        (0.5, 0.5, 0.0),
                        (0, 0, 1),
                    ]

        pl.add_timer_event(
            max_steps=10_000_000, duration=UPDATE_MS, callback=tick
        )

        # ────────────── start ────────────────────────────────────────────────
        hdr = (
            "\n"
            "  ┌──────────────────────────────────────────┐\n"
            "  │   Chladni Plate Live Visualizer           │\n"
            "  ├──────────────────────────────────────────┤\n"
            "  │  S = Square plate    C = Circular plate   │\n"
            "  │  +/- = Adjust ref fundamental (±5 Hz)    │\n"
            "  │  A   = Auto-lock fundamental to peak     │\n"
            "  │  Q   = Quit                              │\n"
            "  └──────────────────────────────────────────┘\n"
        )
        print(hdr)

        try:
            dev = sd.query_devices(kind="input")
            print(f"  Input device: {dev['name']}")
            print(f"  Channels: {dev['max_input_channels']}, "
                  f"Sample rate: {SAMPLE_RATE} Hz, "
                  f"Block: {BLOCK_SIZE} samples\n")
        except Exception as e:
            print(f"  Could not query input device: {e}\n", file=sys.stderr)

        stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self._audio_cb,
        )

        with stream:
            pl.show()

        print("  Visualizer closed.")


if __name__ == "__main__":
    ChladniVisualizer().run()
