# TPMS_Lattice_Generator

**TPMS_Lattice_Generator** is an interactive desktop tool for generating Triply Periodic Minimal Surface (TPMS) lattices and hybrid/graded TPMS structures.

It is designed for additive manufacturing and mechanical property studies where you need precise control over lattice type, grading direction, transition behavior and target relative density (RD).  
You can preview the structure in real time and export high-resolution STL files for 3D printing or simulation.

> **In one sentence:** generate TPMS lattices (P/G/D/I/N), smoothly blend two topologies, preview in 3D, then export STL for printing or analysis.

---

## ✨ Highlights

- Multiple TPMS types  
  Primitive (P), Gyroid (G), Diamond (D), I-WP (I), Neovius (N)

- Hybrid / graded structures  
  Smoothly blend **Topology A → Topology B** with a controllable transition region

- Flexible grading direction  
  - Z gradient (bottom → top)  
  - X gradient (left → right)  
  - Diagonal X–Z gradient  

- Target relative density control  
  Generate structures directly from a user-defined RD (0–1)

- Two-stage resolution workflow  
  - fast low-resolution preview for interaction  
  - high-resolution reconstruction for STL export  

- User-friendly GUI  
  - Dark / Light theme switch  
  - Chinese / English UI switch  
  - One-click screenshot  
  - Standard views (Top / Front / Right) and Reset view  

- Binary STL export with progress bar and cancel support

---

🚀 Getting Started

Option 1 (recommended): Run the prebuilt executable
	1.	Go to the Releases page of this repository
	2.	Download the Windows package (zip file), e.g.
TPMS_Mixer_v1.0_Windows_x64.zip
	3.	Unzip it to any folder
	4.	Open the folder and double-click TPMS_Mixer.exe

No Python or extra installation is required.
This is the easiest way to use the software.

If Windows shows a security warning on first run, click
“More info” → “Run anyway”.

⸻

Option 2: Run from the Python source file

If you prefer to run the .py file directly (for development or research):
	1.	Clone or download this repository

git clone https://github.com/your-username/TPMS_Lattice_Generator.git
cd TPMS_Lattice_Generator

(or click Code → Download ZIP on GitHub and unzip it)
	2.	Install the required dependencies

pip install -r requirements.txt

	3.	Run the program

python TPMS_Mixer.py

The graphical interface will open and you can use the tool exactly like the executable version.

Note:
	•	Installing packages such as vtk and pyvista may take some time.
	•	If you only want to use the software, the executable in Releases is strongly recommended.

⸻

🧠 How hybrid grading works

Two implicit TPMS fields are generated:
	•	ΦA from Topology A
	•	ΦB from Topology B

They are blended using a logistic transition function controlled by:
	•	d0 – transition center (where A and B are half-half)
	•	k  – transition steepness (larger = sharper interface)

Together with the grading direction (Z, X or diagonal X–Z), this allows you to study:
	•	different hybrid ratios
	•	different transition thicknesses
	•	different spatial grading strategies

All while maintaining a global target relative density (RD).

⸻

📦 Export

The exported STL:
	•	is reconstructed at high resolution
	•	uses binary STL format
	•	shows a progress bar and supports cancellation

This makes it suitable for:
	•	metal / polymer additive manufacturing
	•	finite element analysis
	•	porous structure research

⸻
