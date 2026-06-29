"""Geometry tab — editable folder table with viewer buttons and 2-D bounding-box plot."""
import glob
import os
import struct
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

from beamontarget.gui.gui_widgets import (
    make_card,
    get_project_folder as _get_project_folder,
    to_relative_path as _to_relative_path,
    resolve_path as _resolve_path,
)


class GeometryTab(ttk.Frame):
    """Treeview-based geometry folder manager."""

    def __init__(self, parent, cfg, colours, *,
                 view_geometry_fn, view_geometry_o3d_fn):
        super().__init__(parent)
        self.cfg = cfg
        self._colours = colours
        self._view_geometry = view_geometry_fn
        self._view_geometry_o3d = view_geometry_o3d_fn
        self._build()

    # ------------------------------------------------------------------
    #  Build UI
    # ------------------------------------------------------------------
    def _build(self):
        card = make_card(self, "Geometry Folders", pady=(12, 10))

        # Treeview
        cols = ("folder", "scale", "target_length", "save_details",
                "is_diagnostic", "save_impact_data", "max_impact_records")
        tree_frame = ttk.Frame(card, style="Card.TFrame")
        tree_frame.pack(fill="both", expand=True)

        self.geo_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=8)
        headers = {"folder": "Folder", "scale": "Scale", "target_length": "Target Len",
                   "save_details": "Details", "is_diagnostic": "Diagnostic",
                   "save_impact_data": "Impacts", "max_impact_records": "Max Records"}
        widths = {"folder": 140, "scale": 60, "target_length": 90, "save_details": 65,
                  "is_diagnostic": 80, "save_impact_data": 65, "max_impact_records": 100}
        for c in cols:
            self.geo_tree.heading(c, text=headers[c])
            self.geo_tree.column(c, width=widths[c], anchor="center")
        self.geo_tree.column("folder", anchor="w")

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.geo_tree.yview)
        self.geo_tree.configure(yscrollcommand=vsb.set)
        self.geo_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._populate_geo_tree()

        # Buttons
        btn_frm = ttk.Frame(card, style="Card.TFrame")
        btn_frm.pack(fill="x", pady=(10, 0))
        ttk.Button(btn_frm, text="+ Add Folder…", style="Secondary.TButton",
                    command=self._add_geo_folder).pack(side="left", padx=(0, 4))
        ttk.Button(btn_frm, text="✎ Edit Selected…", style="Secondary.TButton",
                    command=self._edit_geo_folder).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="✕ Remove", style="Secondary.TButton",
                    command=self._remove_geo_folder).pack(side="left", padx=4)
        ttk.Button(btn_frm, text="🔍 ParaView", style="Secondary.TButton",
                    command=self._view_geometry).pack(side="right", padx=(4, 0))
        ttk.Button(btn_frm, text="🔷 Open3D", style="Secondary.TButton",
                    command=self._view_geometry_o3d).pack(side="right", padx=4)

        # --- 2-D Bounding-box plot ---
        plot_card = make_card(self, "Geometry Overview Plot")

        ctrl_frm = ttk.Frame(plot_card, style="Card.TFrame")
        ctrl_frm.pack(fill="x", pady=(0, 6))

        ttk.Label(ctrl_frm, text="Projection:", style="Card.TLabel").pack(side="left")
        self._var_projection = tk.StringVar(value="X–Z")
        proj_combo = ttk.Combobox(ctrl_frm, textvariable=self._var_projection,
                                   values=["X–Y", "X–Z", "Y–Z"],
                                   state="readonly", width=8)
        proj_combo.pack(side="left", padx=(6, 12))

        ttk.Button(ctrl_frm, text="↻ Refresh Plot", style="Secondary.TButton",
                    command=self._update_plot).pack(side="left")

        self._fig = Figure(figsize=(6, 3.2), dpi=100)
        self._fig.patch.set_facecolor("#ffffff")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=plot_card)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    # ------------------------------------------------------------------
    #  Treeview helpers
    # ------------------------------------------------------------------
    def _populate_geo_tree(self):
        for item in self.geo_tree.get_children():
            self.geo_tree.delete(item)
        for folder, s in self.cfg.get("GEOMETRY_FOLDERS", {}).items():
            self.geo_tree.insert("", "end", values=(
                folder, s.get("scale", 1), s.get("target_length", 1.0),
                "✓" if s.get("save_details") else "", "✓" if s.get("is_diagnostic") else "",
                "✓" if s.get("save_impact_data") else "",
                s.get("max_impact_records") if s.get("max_impact_records") is not None else "—"))

    def _add_geo_folder(self):
        self._geo_dialog(None)

    def _edit_geo_folder(self):
        sel = self.geo_tree.selection()
        if not sel:
            return
        folder = self.geo_tree.item(sel[0])["values"][0]
        self._geo_dialog(str(folder))

    def _remove_geo_folder(self):
        sel = self.geo_tree.selection()
        if not sel:
            return
        folder = str(self.geo_tree.item(sel[0])["values"][0])
        if messagebox.askyesno("Remove", f"Remove geometry folder '{folder}'?"):
            self.cfg["GEOMETRY_FOLDERS"].pop(folder, None)
            self._populate_geo_tree()

    def _geo_dialog(self, existing_folder):
        """Pop up a dialog to add/edit a geometry folder entry."""
        dlg = tk.Toplevel(self)
        dlg.title("Edit Geometry Folder" if existing_folder else "Add Geometry Folder")
        dlg.geometry("400x360")
        dlg.transient(self)
        dlg.grab_set()

        s = self.cfg.get("GEOMETRY_FOLDERS", {}).get(existing_folder, {}) if existing_folder else {}

        entries = {}
        row = 0

        ttk.Label(dlg, text="Folder path:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        v_folder = tk.StringVar(value=existing_folder or "")
        folder_frm = ttk.Frame(dlg)
        folder_frm.grid(row=row, column=1, sticky="we", padx=8)
        e = ttk.Entry(folder_frm, textvariable=v_folder, width=24)
        e.pack(side="left", fill="x", expand=True)

        def _browse_geo_folder():
            base_dir = _get_project_folder()
            d = filedialog.askdirectory(
                initialdir=base_dir,
                title="Select geometry folder",
                parent=dlg,
            )
            if d:
                rel = _to_relative_path(d, base_dir)
                if existing_folder:
                    e.config(state="normal")
                v_folder.set(rel)
                if existing_folder:
                    e.config(state="disabled")

        ttk.Button(folder_frm, text="Browse…", command=_browse_geo_folder).pack(
            side="left", padx=(4, 0)
        )
        if existing_folder:
            e.config(state="disabled")
        entries["folder"] = v_folder

        fields = [
            ("scale", "Scale:", float, 1),
            ("target_length", "Target length:", float, 1.0),
        ]
        for key, label, typ, default in fields:
            row += 1
            ttk.Label(dlg, text=label).grid(row=row, column=0, sticky="w", padx=8, pady=4)
            v = tk.StringVar(value=str(s.get(key, default)))
            ttk.Entry(dlg, textvariable=v, width=15).grid(row=row, column=1, sticky="w", padx=8)
            entries[key] = (v, typ)

        bools = [
            ("save_details", "Save detailed reports (VTP files)", False),
            ("is_diagnostic", "Is diagnostic (transparent)", False),
            ("save_impact_data", "Save impact data", False),
            ("show_in_plot", "Show in plot", False),
        ]
        for key, label, default in bools:
            row += 1
            v = tk.BooleanVar(value=s.get(key, default))
            ttk.Checkbutton(dlg, text=label, variable=v).grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=2)
            entries[key] = (v, bool)

        row += 1
        ttk.Label(dlg, text="Max impact records:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
        mir = s.get("max_impact_records")
        v_mir = tk.StringVar(value=str(mir) if mir is not None else "")
        ttk.Entry(dlg, textvariable=v_mir, width=15).grid(row=row, column=1, sticky="w", padx=8)
        entries["max_impact_records"] = v_mir

        def _ok():
            folder_name = entries["folder"].get().strip()
            if not folder_name:
                messagebox.showwarning("Missing", "Folder path is required."); return
            new_s = {}
            for key, (var, typ) in [(k, v) for k, v in entries.items() if k not in ("folder", "max_impact_records")]:
                try:
                    new_s[key] = typ(var.get())
                except (ValueError, tk.TclError):
                    new_s[key] = var.get()
            mir_val = entries["max_impact_records"].get().strip()
            new_s["max_impact_records"] = int(mir_val) if mir_val else None
            if "GEOMETRY_FOLDERS" not in self.cfg:
                self.cfg["GEOMETRY_FOLDERS"] = {}
            # If the folder was renamed via Browse, remove the old key
            if existing_folder and folder_name != existing_folder:
                self.cfg["GEOMETRY_FOLDERS"].pop(existing_folder, None)
            self.cfg["GEOMETRY_FOLDERS"][folder_name] = new_s
            self._populate_geo_tree()
            dlg.destroy()

        row += 1
        ttk.Button(dlg, text="OK", command=_ok).grid(row=row, column=0, columnspan=2, pady=10)
        dlg.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------
    #  2-D bounding-box plot
    # ------------------------------------------------------------------
    _PLOT_COLOURS = [
        "#2563eb", "#dc2626", "#16a34a", "#d97706",
        "#7c3aed", "#db2777", "#0891b2", "#65a30d",
    ]

    @staticmethod
    def _stl_bbox(filepath, scale=1.0):
        """Return (min_xyz, max_xyz) arrays for a binary or ASCII STL file."""
        with open(filepath, "rb") as fh:
            header = fh.read(80)
            n_tri_bytes = fh.read(4)
            if len(n_tri_bytes) < 4:
                return None
            n_tri = struct.unpack("<I", n_tri_bytes)[0]
            expected = 80 + 4 + n_tri * 50
            fh.seek(0, 2)
            actual = fh.tell()
            if actual == expected and n_tri > 0:
                # Binary STL — fast numpy read
                fh.seek(84)
                dtype = np.dtype([
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attr", "<u2"),
                ])
                data = np.frombuffer(fh.read(n_tri * 50), dtype=dtype)
                verts = data["vertices"].reshape(-1, 3) * scale
                return verts.min(axis=0), verts.max(axis=0)

        # Fallback: ASCII STL
        verts = []
        with open(filepath, "r", errors="replace") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) == 4 and parts[0] == "vertex":
                    try:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except ValueError:
                        pass
        if not verts:
            return None
        arr = np.array(verts) * scale
        return arr.min(axis=0), arr.max(axis=0)

    def _compute_folder_bboxes(self):
        """Return {folder_name: (min_xyz, max_xyz)} for every configured folder."""
        result = {}
        for folder, settings in self.cfg.get("GEOMETRY_FOLDERS", {}).items():
            folder_abs = _resolve_path(folder) if not os.path.isabs(folder) else folder
            if not os.path.isdir(folder_abs):
                continue
            scale = settings.get("scale", 1.0)
            stl_files = sorted(
                glob.glob(os.path.join(folder_abs, "*.stl"))
                + glob.glob(os.path.join(folder_abs, "*.STL"))
            )
            if not stl_files:
                continue
            g_min = np.full(3, np.inf)
            g_max = np.full(3, -np.inf)
            for stl in stl_files:
                bb = self._stl_bbox(stl, scale)
                if bb is None:
                    continue
                g_min = np.minimum(g_min, bb[0])
                g_max = np.maximum(g_max, bb[1])
            if np.all(np.isfinite(g_min)):
                result[folder] = (g_min, g_max)
        return result

    def _update_plot(self):
        """Redraw the 2-D bounding-box overview."""
        ax = self._ax
        ax.clear()

        proj = self._var_projection.get()
        idx_map = {"X–Y": (0, 1), "X–Z": (0, 2), "Y–Z": (1, 2)}
        axis_labels = {"X–Y": ("X (m)", "Y (m)"),
                       "X–Z": ("X (m)", "Z (m)"),
                       "Y–Z": ("Y (m)", "Z (m)")}
        ix, iy = idx_map.get(proj, (0, 2))
        xlabel, ylabel = axis_labels.get(proj, ("", ""))

        bboxes = self._compute_folder_bboxes()

        if not bboxes:
            ax.text(0.5, 0.5, "No geometry folders with STL files found",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#64748b")
        else:
            for ci, (folder, (lo, hi)) in enumerate(bboxes.items()):
                colour = self._PLOT_COLOURS[ci % len(self._PLOT_COLOURS)]
                x0, y0 = lo[ix], lo[iy]
                w, h = hi[ix] - lo[ix], hi[iy] - lo[iy]
                rect = Rectangle((x0, y0), w, h,
                                  linewidth=1.8, edgecolor=colour,
                                  facecolor=colour, alpha=0.18)
                ax.add_patch(rect)
                # Label at top-left corner
                lbl = os.path.basename(folder.rstrip("/\\")) or folder
                ax.text(x0 + w * 0.02, y0 + h * 0.96, lbl,
                        fontsize=8, color=colour, fontweight="bold",
                        va="top", clip_on=True)

            ax.set_aspect("equal", adjustable="datalim")
            ax.autoscale_view()
            # Small margin
            ax.margins(0.06)

        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"Bounding Boxes — {proj}", fontsize=10, fontweight="bold")
        ax.grid(True, linewidth=0.4, alpha=0.5)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    #  collect / refresh
    # ------------------------------------------------------------------
    def collect(self, d):
        """Write geometry keys into *d*."""
        d["GEOMETRY_FOLDERS"] = self.cfg.get("GEOMETRY_FOLDERS", {})

    def refresh(self, c):
        """Push config *c* values back into the widgets."""
        self.cfg = c
        self._populate_geo_tree()

