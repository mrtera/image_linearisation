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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Wedge, FancyArrow
import numpy as np
import tempfile
import shutil
import subprocess
import glob


class GaugeApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CSV Analog Gauge")
        self.geometry("600x420")

        self.csv_path = None
        self.df = None
        self.selected_column = tk.StringVar()
        self.min_val = tk.DoubleVar(value=0)
        self.max_val = tk.DoubleVar(value=250)
        # interval in milliseconds between frames (default corresponds to 15.86 fps)
        self.interval_ms = tk.DoubleVar(value=63.05170239596469)
        # playback / export options
        self.loop_var = tk.BooleanVar(value=False)
        self.fps_var = tk.DoubleVar(value=15.86)
        self.running = False
        # track the id returned by `after` so we can cancel scheduled callbacks
        self._after_id = None
        # track any long-running subprocess (ffmpeg) if we add streaming in future
        self._ffmpeg_proc = None
        # handle window close to ensure background callbacks are cancelled
        try:
            self.protocol("WM_DELETE_WINDOW", self.on_close)
        except Exception:
            pass

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        ttk.Button(control_frame, text="Open CSV", command=self.open_csv).grid(row=0, column=0, padx=6)
        ttk.Label(control_frame, text="Min:").grid(row=0, column=1)
        ttk.Entry(control_frame, textvariable=self.min_val, width=4).grid(row=0, column=2)
        ttk.Label(control_frame, text="Max:").grid(row=0, column=3)
        ttk.Entry(control_frame, textvariable=self.max_val, width=4).grid(row=0, column=4)
        ttk.Label(control_frame, text="Interval (ms):").grid(row=0, column=5)
        ttk.Entry(control_frame, textvariable=self.interval_ms, width=5).grid(row=0, column=6)
        ttk.Label(control_frame, text="Units:").grid(row=0, column=7)
        self.units = tk.StringVar(value="µm/s")
        ttk.Entry(control_frame, textvariable=self.units, width=8).grid(row=0, column=8)

        # playback control row: loop checkbox, fps entry, export button
        ttk.Button(control_frame, text="Start", command=self.start).grid(row=1, column=0, padx=6)
        ttk.Button(control_frame, text="Stop", command=self.stop).grid(row=1, column=1, columnspan=2, padx=6)
        # Minimal Prev/Next controls (manual stepping)
        ttk.Button(control_frame, text="Prev", command=self.prev_point).grid(row=1, column=3, columnspan=2, padx=6)
        ttk.Button(control_frame, text="Next", command=self.next_point).grid(row=1, column=5, padx=6)

        ttk.Checkbutton(control_frame, text="Loop", variable=self.loop_var).grid(row=1, column=6, padx=6)
        ttk.Label(control_frame, text="FPS:").grid(row=1, column=7)
        ttk.Entry(control_frame, textvariable=self.fps_var, width=6).grid(row=1, column=8)
        ttk.Button(control_frame, text="Exp. Video", command=self.export_video).grid(row=1, column=5, padx=6)
        ttk.Button(control_frame, text="Exp. Frames", command=self.export_frames_png).grid(row=0, column=8, padx=6)
        ttk.Button(control_frame, text="Exp. Colormap", command=self.export_cmap).grid(row=1, column=9, padx=6)

        # Matplotlib figure
        self.fig = Figure(figsize=(5.5,4.5), dpi=100)
        # make overall figure background black
        self.fig.patch.set_facecolor('black')
        self.ax = self.fig.add_subplot(111, polar=False)
        self.ax.set_facecolor('black')
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
        # prepare playback. If Loop checkbox is set, run continuously; otherwise play once.
        self._play_index = 0
        self.single_pass = not self.loop_var.get()
        self.running = False
        # start 1-second countdown
        self._countdown(1)

    def export_frames_png(self):
        """Export each frame as a PNG with transparent background."""
        if getattr(self, 'data', None) is None:
            messagebox.showwarning("No data", "Open a CSV with data before exporting.")
            return

        out_dir = filedialog.askdirectory(title="Select folder to save PNG frames")
        if not out_dir:
            return

        try:
            self.status_var.set(f"Exporting {len(self.data)} frames as PNG...")
            
            # Store original facecolor to restore later
            original_fig_color = self.fig.get_facecolor()
            original_ax_color = self.ax.get_facecolor()
            
            # Set transparent background
            self.fig.patch.set_alpha(0)
            self.ax.set_facecolor('none')
            self.ax.patch.set_alpha(0)
            
            # Render frames
            for i, v in enumerate(self.data):
                try:
                    # draw into the Figure and save
                    self.draw_gauge(float(v))
                    fname = os.path.join(out_dir, f'frame_{i:06d}.png')
                    # save with transparent background
                    self.fig.savefig(fname, transparent=True, dpi=self.fig.dpi)
                except Exception as e:
                    # continue but note error
                    print('Frame render error', i, e)
            
            # Restore original background colors
            self.fig.patch.set_facecolor(original_fig_color)
            self.fig.patch.set_alpha(1)
            self.ax.set_facecolor(original_ax_color)
            self.ax.patch.set_alpha(1)
            
            # Redraw the current gauge to show restored background
            if hasattr(self, '_play_index'):
                self.draw_gauge(float(self.data[self._play_index]))
                self.canvas.draw()
            
            self.status_var.set(f'Exported {len(self.data)} frames to: {out_dir}')
            messagebox.showinfo('Export complete', f'{len(self.data)} PNG frames saved to:\n{out_dir}')
        except Exception as e:
            messagebox.showerror('Export failed', f'Error during export: {str(e)}')

    def export_video(self):
        """Render frames for all data points, encode with ffmpeg and save an MP4.

        This method saves per-frame PNGs into a temp dir, then calls ffmpeg.
        """
        if getattr(self, 'data', None) is None:
            messagebox.showwarning("No data", "Open a CSV with data before exporting.")
            return

        out_path = filedialog.asksaveasfilename(defaultextension='.mp4', filetypes=[('MP4 video','*.mp4')])
        if not out_path:
            return

        fps = float(self.fps_var.get() or 15.86)
        tmpdir = tempfile.mkdtemp(prefix='gauge_frames_')
        try:
            self.status_var.set(f"Rendering {len(self.data)} frames...")
            # Render frames
            for i, v in enumerate(self.data):
                try:
                    # draw into the Figure and save
                    self.draw_gauge(float(v))
                    fname = os.path.join(tmpdir, f'frame_{i:06d}.png')
                    # save with black background
                    self.fig.savefig(fname, facecolor=self.fig.get_facecolor(), dpi=self.fig.dpi)
                except Exception as e:
                    # continue but note error
                    print('Frame render error', i, e)

            # ensure ffmpeg exists
            ffmpeg = shutil.which('ffmpeg')
            if ffmpeg is None:
                messagebox.showerror('ffmpeg not found', 'ffmpeg is required to encode video. Install ffmpeg and ensure it is on PATH.')
                return

            self.status_var.set('Encoding video (ffmpeg)...')
            # build ffmpeg command
            inp = os.path.join(tmpdir, 'frame_%06d.png')
            cmd = [ffmpeg, '-y', '-framerate', f'{fps}', '-i', inp, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', out_path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                messagebox.showerror('ffmpeg failed', f'Encoding failed: {proc.stderr}')
                return

            self.status_var.set(f'Exported video: {out_path}')
            messagebox.showinfo('Export complete', f'Video saved to:\n{out_path}')
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    def _countdown(self, seconds_left: int):
        """Show a countdown in the status label, then begin playback."""
        if seconds_left > 0:
            try:
                self.status_var.set(f"Starting in {seconds_left}...")
            except Exception:
                pass
            # call again in 1 second (store id so we can cancel on exit)
            try:
                self._after_id = self.after(1000, lambda: self._countdown(seconds_left - 1))
            except Exception:
                pass
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
        # stop playback and cancel any scheduled after callback
        self.running = False
        try:
            if getattr(self, '_after_id', None) is not None:
                self.after_cancel(self._after_id)
                self._after_id = None
        except Exception:
            pass

    def on_close(self):
        """Called when the window is closed. Cancel callbacks and exit cleanly."""
        try:
            # stop running loops
            self.running = False
            # cancel any pending after callback
            if getattr(self, '_after_id', None) is not None:
                try:
                    self.after_cancel(self._after_id)
                except Exception:
                    pass
                self._after_id = None
            # if we ever used a subprocess, try to terminate it
            if getattr(self, '_ffmpeg_proc', None) is not None:
                try:
                    self._ffmpeg_proc.terminate()
                except Exception:
                    pass
        finally:
            try:
                # stop the Tk mainloop and destroy the window
                self.quit()
            except Exception:
                pass
            try:
                self.destroy()
            except Exception:
                pass

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
                if self.loop_var.get():
                    idx = 0
                else:
                    self.running = False
                    try:
                        self.status_var.set("Done")
                    except Exception:
                        pass
                    return
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
        ax.set_facecolor('black')
        # position the axes to leave room for labels below and make the
        # plotting area square so the semicircle is not stretched.
        # (left, bottom, width, height)
        try:
            ax.set_position([0.05, 0.08, 0.90, 0.84])
        except Exception:
            pass
        # use a symmetric square viewport and enforce equal aspect for circular gauge
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        try:
            ax.set_aspect('equal', adjustable='box')
        except Exception:
            pass
        ax.axis('off')

        # draw a continuous background using the 'turbo' colormap
        minv = float(self.min_val.get())
        maxv = float(self.max_val.get())
        if maxv <= minv:
            maxv = minv + 1
        try:
            cmap = plt.get_cmap('turbo')
        except Exception:
            cmap = plt.get_cmap('viridis')

        # draw many small wedges across the top semicircle and color them by the colormap
        nseg = 400
        r_outer = 0.65
        r_inner = 0.1
        for i, frac in enumerate(np.linspace(0.0, 1.0, nseg)):
            # compute angular span for this segment (degrees)
            start_deg = 180.0 - (i / nseg) * 180.0
            end_deg = 180.0 - ((i + 1) / nseg) * 180.0
            color = cmap(frac)
            try:
                wedge = Wedge((0, 0), r_outer, end_deg, start_deg, width=(r_outer - r_inner), facecolor=color, edgecolor='none')
                ax.add_patch(wedge)
            except Exception:
                # fallback: draw a thin line if Wedge fails
                ang = math.radians((start_deg + end_deg) / 2.0)
                x1 = r_inner * math.cos(ang)
                y1 = r_inner * math.sin(ang)
                x2 = r_outer * math.cos(ang)
                y2 = r_outer * math.sin(ang)
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=3)

        def value_to_angle(v):
            # map [minv,maxv] to angles across the TOP semicircle: left->top->right
            # We want smallest values on the left (180°), middle near north (90°),
            # and largest values on the right (0°). So angle = 180 - frac*180.
            frac = (v - minv) / (maxv - minv)
            frac = max(0.0, min(1.0, frac))
            angle_deg = 180.0 - frac * 180.0
            return math.radians(angle_deg)

        # ticks and labels
        tick = True
        for frac in np.linspace(0,1,6):
            
            # tick angles along top semicircle (left=180 -> right=0)
            ang = math.radians(180.0 - frac * 180.0)
            x1 = 0.4 * math.cos(ang)
            y1 = 0.4 * math.sin(ang)
            x2 = 0.63 * math.cos(ang)
            y2 = 0.63 * math.sin(ang)
            ax.plot([x1,x2],[y1,y2], color='white', linewidth=2)
            label = f"{minv + frac*(maxv-minv):.0f}"
            lx = 0.85 * math.cos(ang)
            ly = 0.85 * math.sin(ang)
            if tick:
                ax.text(lx, ly, label, horizontalalignment='center', verticalalignment='center', fontsize=18, fontweight='bold', color='white')
            # tick = not tick

        # draw needle
        if value is not None:
            angle = value_to_angle(value)
            nx = 0.6 * math.cos(angle)
            ny = 0.6 * math.sin(angle)
            ax.plot([0, nx], [0, ny], color='white', linewidth=3)

            # center circle
            center = matplotlib.patches.Circle((0,0), 0.05, color='white')
            ax.add_patch(center)

            ax.text(0, -0.12, f"{value:.0f}", horizontalalignment='center', verticalalignment='top', fontsize=18, fontweight='bold', color='white')
            # show units next to the numeric value
            try:
                unit = self.units.get()
            except Exception:
                unit = ''
            ax.text(0, -0.3, f"{unit}", horizontalalignment='center', verticalalignment='top', fontsize=18, fontweight='bold', color='white')

        self.canvas.draw()
    def export_cmap(self):
        """Export the current colormap to a PNG file with transparent background."""
        out_path = filedialog.asksaveasfilename(defaultextension='.svg', filetypes=[('SVG image','*.svg')])
        # determine colormap range from loaded data or min/max controls and prepare ticks
        try:
            if getattr(self, 'data', None) is not None and len(self.data) > 0:
                dmin = float(np.nanmin(self.data))
                dmax = float(np.nanmax(self.data))
            else:
                dmin = float(self.min_val.get())
                dmax = float(self.max_val.get())
        except Exception:
            dmin, dmax = 0.0, 1.0
        if dmax <= dmin:
            dmax = dmin + 1.0

        num_ticks = 3
        cmap_ticks = np.linspace(dmin, dmax, num_ticks)
        tick_formatter = lambda v: f"{v:.0f}"

        # Monkeypatch plt.subplots briefly so the following code (which calls plt.subplots,
        # ax.imshow(...) and ax.set_axis_off()) will produce an image whose x-axis is
        # scaled to [dmin, dmax] and that receives the requested ticks/labels.
        _orig_subplots = plt.subplots


        def _patched_subplots(*args, **kwargs):
            fig, ax = _orig_subplots(*args, **kwargs)
            _orig_imshow = ax.imshow

            def _imshow_override(arr, *a, **kw):
                # force the image to span the data range on X axis
                kw.setdefault('extent', (dmin, dmax, 0, 1))
                m = _orig_imshow(arr, *a, **kw)
                # set ticks and styling after the image is drawn
                ax.set_axis_on()
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xticks(cmap_ticks)
                ax.set_xticklabels([tick_formatter(v) for v in cmap_ticks], color='white', fontsize=10)
                ax.set_yticks([])
                # try to show units below the colorbar
                try:
                    unit = self.units.get()
                except Exception:
                    unit = ''
                if unit:
                    ax.set_xlabel(unit, color='white', fontsize=10)
                # clean up spines for nicer appearance on transparent background
                for side in ('top', 'right', 'left'):
                    ax.spines[side].set_visible(False)
                    ax.spines['bottom'].set_color('white')
                return m

            def _set_axis_off_noop():
                # original code calls set_axis_off(); override so we keep axis and ticks visible
                return

            ax.imshow = _imshow_override
            ax.set_axis_off = _set_axis_off_noop

            # restore plt.subplots to original so our patch is temporary
            plt.subplots = _orig_subplots
            return fig, ax


        plt.subplots = _patched_subplots
        if not out_path:
            return
        try:
            # create a temporary figure for the colormap
            fig, ax = plt.subplots(figsize=(3, 1), dpi=300)
            fig.patch.set_alpha(0.0)  # transparent background
            cmap = plt.get_cmap('turbo')
            gradient = np.linspace(0, 1, 500)
            gradient = np.vstack((gradient, gradient))
            ax.imshow(gradient, aspect='auto', cmap=cmap)
            ax.set_axis_off()
            fig.savefig(out_path, bbox_inches='tight', pad_inches=0)
        except Exception as e:
            print(f"Error exporting colormap: {e}")

if __name__ == '__main__':
    app = GaugeApp()
    app.mainloop()


# %%
