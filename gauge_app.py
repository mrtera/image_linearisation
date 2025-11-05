#%%
"""
Simple analog gauge GUI that reads a CSV and shows the latest value.

- Uses tkinter for GUI and embeds a matplotlib figure for the gauge.
- Reads a CSV with a header; you can choose column, min/max and refresh interval.

Run: python gauge_app.py
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import math
import os
import time

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Wedge, FancyArrow
import numpy as np


class GaugeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Analog Gauge")
        self.geometry("700x520")

        self.csv_path = None
        self.df = None
        self.selected_column = tk.StringVar()
        self.min_val = tk.DoubleVar(value=0)
        self.max_val = tk.DoubleVar(value=20)
        self.interval_ms = tk.IntVar(value=0.06305170239596469)
        self.running = False

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(control_frame, text="Open CSV", command=self.open_csv).grid(row=0, column=0, padx=6)
        ttk.Label(control_frame, text="Min:").grid(row=0, column=2)
        ttk.Entry(control_frame, textvariable=self.min_val, width=4).grid(row=0, column=3)
        ttk.Label(control_frame, text="Max:").grid(row=0, column=4)
        ttk.Entry(control_frame, textvariable=self.max_val, width=4).grid(row=0, column=5)
        ttk.Label(control_frame, text="Interval (ms):").grid(row=0, column=6)
        ttk.Entry(control_frame, textvariable=self.interval_ms, width=20).grid(row=0, column=7)
        ttk.Label(control_frame, text="Units:").grid(row=0, column=8)
        self.units = tk.StringVar(value="µm/s")
        ttk.Entry(control_frame, textvariable=self.units, width=8).grid(row=0, column=9)
        ttk.Button(control_frame, text="Start", command=self.start).grid(row=1, column=0, padx=6)
        ttk.Button(control_frame, text="Stop", command=self.stop).grid(row=1, column=2, columnspan=2, padx=6)
        # Minimal Prev/Next controls (manual stepping). These only pause playback and
        # step through the currently loaded `self.data` array.
        ttk.Button(control_frame, text="Prev", command=self.prev_point).grid(row=1, column=4, columnspan=2, padx=6)
        ttk.Button(control_frame, text="Next", command=self.next_point).grid(row=1, column=6, padx=6)

        # Matplotlib figure
        self.fig = Figure(figsize=(6.5,4.5), dpi=100)
        self.ax = self.fig.add_subplot(111, polar=False)
        self.ax.set_facecolor('white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # State for gauge drawing
        self.last_value = 0

        # status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self, textvariable=self.status_var)
        status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=4)

        # initial blank gauge
        self.draw_gauge(0)

    def open_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            # fallback to plain numeric load for simple files
            try:
                arr = np.loadtxt(path, delimiter=',')
                if arr.ndim == 2:
                    arr = arr[:, -1]
                self.data = arr.astype(float)
                self.csv_path = path
                self.status_var.set(f"Loaded numeric CSV: {os.path.basename(path)}")
                return
            except Exception:
                messagebox.showerror("Error", f"Failed to read CSV: {e}")
                return

        # prefer numeric columns
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            series = df[numeric_cols[0]].dropna()
            try:
                self.selected_column.set(numeric_cols[0])
            except Exception:
                pass
        else:
            # try to coerce last column
            series = pd.to_numeric(df.iloc[:, -1], errors='coerce').dropna()

        if series.empty:
            messagebox.showerror("Error", "No numeric data found in CSV")
            return

        self.data = series.values.astype(float)
        self.csv_path = path
        self.status_var.set(f"Loaded {os.path.basename(path)} ({len(self.data)} rows)")

    def start(self):
        if not self.csv_path:
            messagebox.showwarning("No CSV", "Please open a CSV first.")
            return
        # prepare for a single pass playback: start a 3s countdown then play once
        self._play_index = 0
        self.single_pass = True
        self.running = False
        # start 3-second countdown
        self._countdown(3)

    def _countdown(self, seconds_left: int):
        """Show a countdown in the status label, then begin playback."""
        if seconds_left > 0:
            try:
                self.status_var.set(f"Starting in {seconds_left}...")
            except Exception:
                pass
            # call again in 1 second
            self.after(1000, lambda: self._countdown(seconds_left - 1))
            return

        # start running and play from the beginning once
        self._play_index = 0
        self.running = True
        try:
            self.status_var.set("Running...")
        except Exception:
            pass
        self.poll_csv()

    def stop(self):
        self.running = False

    def prev_point(self):
        """Pause playback and move one step backward in the loaded data."""
        if getattr(self, 'data', None) is None:
            return
        # pause automatic playback
        self.running = False
        # ensure play index exists
        if not hasattr(self, '_play_index'):
            self._play_index = len(self.data) - 1
        else:
            self._play_index = max(0, int(self._play_index) - 1)

        try:
            value = float(self.data[int(self._play_index)])
        except Exception:
            return
        self.update_gauge(value)
        try:
            self.status_var.set(f"Manual {int(self._play_index)+1}/{len(self.data)}: {value}")
        except Exception:
            pass

    def next_point(self):
        """Pause playback and move one step forward in the loaded data."""
        if getattr(self, 'data', None) is None:
            return
        # pause automatic playback
        self.running = False
        # if no index yet, start at the beginning
        if not hasattr(self, '_play_index'):
            self._play_index = 0
        else:
            self._play_index = min(len(self.data) - 1, int(self._play_index) + 1)

        try:
            value = float(self.data[int(self._play_index)])
        except Exception:
            return
        self.update_gauge(value)
        try:
            self.status_var.set(f"Manual {int(self._play_index)+1}/{len(self.data)}: {value}")
        except Exception:
            pass

    def poll_csv(self):
        # non-blocking single-step advance through self.data
        if getattr(self, 'data', None) is None:
            return
        try:
            idx = int(getattr(self, '_play_index', 0))
            if idx >= len(self.data):
                # no more data
                if getattr(self, 'single_pass', False):
                    self.running = False
                    try:
                        self.status_var.set("Done")
                    except Exception:
                        pass
                    return
                idx = 0
            value = float(self.data[idx])
        except Exception:
            return

        self.update_gauge(value)
        try:
            self.status_var.set(f"Last read: {value} @ {time.strftime('%H:%M:%S')}")
        except Exception:
            pass

        # advance index
        self._play_index = idx + 1

        # if single_pass and we've consumed all data, stop
        if getattr(self, 'single_pass', False) and self._play_index >= len(self.data):
            self.running = False
            try:
                self.status_var.set("Done")
            except Exception:
                pass
            return

        if self.running:
            interval = max(50, int(self.interval_ms.get()))
            self.after(interval, self.poll_csv)

    def update_gauge(self, value):
        self.last_value = value
        self.draw_gauge(value)

    def draw_gauge(self, value):
        # simple semicircular gauge from -90 to +90 degrees
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.1, 1.3)
        ax.axis('off')

        # draw arcs for zones (green, yellow, red)
        minv = float(self.min_val.get())
        maxv = float(self.max_val.get())
        if maxv <= minv:
            maxv = minv + 1

        def value_to_angle(v):
            # map [minv,maxv] to angles across the TOP semicircle: left->top->right
            # We want smallest values on the left (180°), middle near north (90°),
            # and largest values on the right (0°). So angle = 180 - frac*180.
            frac = (v - minv) / (maxv - minv)
            frac = max(0.0, min(1.0, frac))
            angle_deg = 180.0 - frac * 180.0
            return math.radians(angle_deg)

        # draw colored wedges
        # green: 0-60%, yellow:60-85%, red:85-100%
        zones = [(0.0,0.6,'#4CAF50'), (0.6,0.85,'#FFC107'), (0.85,1.0,'#F44336')]
        for (a,b,color) in zones:
            # convert fractional zone [a,b] to angles along top semicircle
            start = 180.0 - a * 180.0
            end = 180.0 - b * 180.0
            wedge = Wedge((0,0), 1.0, start, end, width=0.3, facecolor=color, transform=None)
            ax.add_patch(wedge)

        # ticks and labels
        for frac in np.linspace(0,1,11):
            # tick angles along top semicircle (left=180 -> right=0)
            ang = math.radians(180.0 - frac * 180.0)
            x1 = 0.7 * math.cos(ang)
            y1 = 0.7 * math.sin(ang)
            x2 = 0.9 * math.cos(ang)
            y2 = 0.9 * math.sin(ang)
            ax.plot([x1,x2],[y1,y2], color='k', linewidth=7)
            label = f"{minv + frac*(maxv-minv):.0f}"
            lx = 1.05 * math.cos(ang)
            ly = 1.05 * math.sin(ang)
            ax.text(lx, ly, label, horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight='bold')

        # draw needle
        if value is not None:
            angle = value_to_angle(value)
            nx = 0.83 * math.cos(angle)
            ny = 0.83 * math.sin(angle)
            ax.plot([0, nx], [0, ny], color='black', linewidth=7)

            # center circle
            center = matplotlib.patches.Circle((0,0), 0.05, color='black')
            ax.add_patch(center)

            ax.text(0, -0.08, f"{value}", horizontalalignment='center', verticalalignment='top', fontsize=25, fontweight='bold')
            # show units next to the numeric value
            try:
                unit = self.units.get()
            except Exception:
                unit = ''
            ax.text(0, -0.18, f"{unit}", horizontalalignment='center', verticalalignment='top', fontsize=25, fontweight='bold')

        self.canvas.draw()


if __name__ == '__main__':
    app = GaugeApp()
    app.mainloop()


# %%
