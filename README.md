# Chladni Plate Live Visualizer

Real-time 3D visualization of Chladni plate vibration patterns. Captures audio from a function generator, displays a rolling waterfall spectrogram and the predicted nodal pattern for the detected frequency.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

## Install

```bash
pip install pyvista sounddevice numpy scipy
python chladni_visualizer.py
```

## Connecting the Function Generator

### Windows

Most Windows laptops have a mic-in port that accepts input directly.

```
Function Generator ──[aux cable]──► Laptop mic-in
```

1. Connect a 3.5mm aux cable from the function generator output to the laptop's mic/line-in port
2. Go to **Settings → System → Sound → Input** and select the mic/line-in device
3. **Important:** Turn the function generator amplitude to minimum first, then slowly increase until you see a clean peak in the waterfall — mic inputs expect millivolt signals, and line-level output can clip
4. Disable any audio enhancements: right-click speaker icon → **Sound settings** → **Input device properties** → **Advanced** → uncheck **Enable audio enhancements**

### Mac

MacBook 3.5mm jacks are output-only (headphones), so you need a USB audio interface in between.

```
Function Generator ──[audio cable]──► USB Audio Interface ──[USB]──► Mac
```

A **Behringer UCA202** (~$25) works well:

1. Connect the function generator output to the interface's RCA input (use a 3.5mm-to-RCA or BNC-to-RCA cable)
2. Plug the interface into the Mac via USB
3. Go to **System Settings → Sound → Input** and select the USB audio device
4. Start with the function generator amplitude at minimum and bring it up gradually

## Controls

| Key | Action |
|-----|--------|
| `S` | Switch to square plate |
| `C` | Switch to circular plate |
| `+` / `-` | Adjust reference fundamental ±5 Hz |
| `A` | Auto-lock fundamental to current peak |
| `Q` | Quit |

## How It Works

- Audio is captured at 44.1 kHz via the system's default input device
- A windowed FFT extracts the frequency spectrum each frame
- The dominant frequency is mapped to the nearest Chladni eigenmode (n, m) relative to a reference fundamental
- **Left panel**: 3D waterfall spectrogram (frequency × time × amplitude)
- **Right panel**: predicted Chladni nodal pattern — square plates use cosine product modes, circular plates use Bessel function modes
