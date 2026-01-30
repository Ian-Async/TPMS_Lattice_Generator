# TPMS_Lattice_Generator

**TPMS_Lattice_Generator** is an interactive desktop tool for generating Triply Periodic Minimal Surface (TPMS) lattices and *hybrid/graded* TPMS structures.  
It is designed for additive manufacturing workflows and mechanical property studies where you want precise control over lattice type, grading direction, transition behavior, and target relative density (RD).

> **One sentence:** Generate TPMS lattices (P/G/D/I/N), blend two topologies with a controllable transition region, preview in 3D, then export STL for printing/simulation.

---

## ✨ Highlights

- **TPMS library**: Primitive (P), Gyroid (G), Diamond (D), I-WP (I), Neovius (N)
- **Hybrid / graded TPMS**: blend Topology A → Topology B with a logistic transition
- **Grading directions**:
  - Z-gradient (bottom → top)
  - X-gradient (left → right)
  - Diagonal X–Z gradient
- **Target relative density (RD) control**: generate lattices based on a user-defined RD (0–1)
- **Dual resolution workflow**:
  - fast **Preview Res**
  - high-quality **Export Res** for STL
- **GUI quality-of-life**:
  - **Dark / Light theme** switch
  - **中文 / English** UI switch
  - Screenshot button for quick documentation
  - Standard views (Top / Front / Right) + Reset
- **Binary STL export** with progress + cancel support

---


##🚀 Quick Start (Users)

Option A — Run the Windows build (recommended for classmates)
	1.	Go to Releases and download the Windows .zip
	2.	Unzip it anywhere
	3.	Run TPMS_Mixer.exe

✅ No Python environment needed.

Option B — Run from source (developers)

pip install numpy scikit-image PySide6 pyvista pyvistaqt vtk
python Tpms_mixer.py


⸻

🧠 How the hybrid transition works (intuitive explanation)

You pick:
	•	Topology A (primary phase)
	•	Topology B (secondary phase)
	•	A grading direction (Z / X / XZ)

The tool computes two implicit fields ΦA and ΦB, then blends them using a logistic weight w:
	•	d0 = transition center (where A→B is half-half)
	•	k  = steepness (higher = sharper interface)

This makes it easy to study:
	•	different hybrid ratios (by shifting d0)
	•	different transition thickness (by changing k)
	•	different grading directions (Z vs X vs diagonal)

⸻

🎛️ Main Controls (What each block does)

UI
	•	Language: 中文 / English
	•	Theme: Dark / Light

Topology
	•	Topology A (primary): base structure
	•	Topology B (secondary): blended structure

Grading
	•	Direction: Z, X, or diagonal X–Z
	•	Transition center (d0): shifts the interface location
	•	Steepness (k): controls the gradient thickness

Geometry
	•	Target RD: overall relative density target (0–1)
	•	Periods Kx, Ky, Kz: number of unit cells in each axis
	•	Size (mm) Sx, Sy, Sz: physical dimensions
	•	Preview Res: fast mesh reconstruction for interactive preview
	•	Export Res: high-resolution reconstruction for STL

Render
	•	Render style: material-like visualization presets
	•	Preview: generate and display mesh
	•	Export STL: save a binary STL with progress bar + cancel

⸻

🧩 Notes & Practical Tips
	•	If preview/export is slow or memory-heavy:
	•	reduce Res, or
	•	reduce Kx/Ky/Kz
	•	A very sharp transition (high k) can create a thin interface region; for printing, consider moderate values.
	•	Export uses high-resolution marching cubes, so expect longer time than preview.

⸻

📦 What’s in a Release

A release asset typically contains:
	•	TPMS_Mixer.exe
	•	required DLLs & runtime dependencies (packed by PyInstaller in onedir mode)

Users only need to download → unzip → run.

⸻

📄 License

MIT License (or update this section if you change the license).

⸻

🙌 Acknowledgements
	•	Mesh extraction via skimage.measure.marching_cubes
	•	Visualization via PyVista + PyVistaQt
	•	GUI via PySide6

---

### 你下一步该做什么（最有效）
1) 在仓库根目录新建 `README.md`，粘贴上面内容  
2) 新建 `docs/screenshots/` 文件夹  
3) 把你软件运行截图（暗色+中文、亮色+英文、杂化案例）放进去  
4) 把 README 里的图片链接改成你真实文件名

如果你愿意，把你准备好的 2~3 张截图文件名发我（或者直接发图），我可以帮你把 README 的截图部分改成“直接可用”的最终版本（包括排列、标题、说明文字）。
