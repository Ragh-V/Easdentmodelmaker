# EasDentModelMaker: Hybrid Dental CAD/CAM Framework

EasDentModelMaker is a specialized Computer-Aided Design (CAD) application designed for the processing, modification, and preparation of dental 3D scans. It features a hybrid architecture, leveraging Python for clinical workflow management and a C++ kernel for high-performance geometric sculpting.

## Key Features

### Clinical Workflows
* **Automated Alignment:** 3-Point pick system (Molars + Incisor) to normalize mesh orientation to the occlusal plane.
* **Maxillary & Mandibular Modes:** Dedicated pipelines for upper and lower arches, handling base extrusion direction (Up/Down) automatically.
* **Feature Recognition:** Tools to mark and extract critical regions like the Incisive Papilla and Retromolar Pads.
* **Undercut Analysis:** Real-time surveyor to visualize insertion paths and identify undercut zones.

### High-Performance Sculpting
* **Dynamic Topology (Dyntopo):** The C++ backend (`sculpt_tore`) dynamically subdivides the mesh during sculpting, adding resolution only where detail is needed.
* **Custom Interaction:** Specialized VTK interaction styles that separate camera manipulation from mesh deformation strokes.

## System Architecture

The project follows a Host-Guest architecture pattern:

* **Host (Python 3.10+):** Handles state management, UI (PySide6), Visualization (PyVista/VTK), and File I/O.
* **Guest (C++17):** Acts as the compute kernel, handling vertex-level operations, topological graph rebuilding, and spatial queries.

### Directory Structure

```text
app/
├── core/
│   ├── commands.py          # Undo/Redo command pattern implementation
│   └── interactors.py       # Custom VTK Interactor styles
├── tools/
│   ├── bezier.py            # Bezier curve marking tool
│   ├── border_tool.py       # Border deformation logic
│   ├── sculptor.py          # Python wrapper for the C++ sculpting engine
│   └── surveyor.py          # Undercut visualization tool
├── ui/
│   ├── eas_dent_model_maker.py # Workflow orchestration (Maxilla/Mandible steps)
│   ├── hierarchy.py         # Scene graph and object management
│   ├── main_window.py       # Primary application window
│   └── dialogs.py           # Pop-up dialogs and prompts
├── utils/
│   ├── config.py            # Application configuration
│   ├── main.py              # Application entry point
│   └── __init__.py
└── src_cpp/                 # C++ Source Code
    ├── sculpt_tore.cpp      # Dynamic Topology Engine
    └── CMakeLists.txt       # Build configuration
```

## Installation & Build

### Prerequisites
* Python 3.10+
* C++ Compiler (GCC 9+, Clang 10+, or MSVC 2019+)
* CMake 3.15+

### 1. Install Python Dependencies
```bash
pip install numpy pyvista vtk PySide6 pybind11
```

### 2. Compile the C++ Kernel
The sculpting engine (`sculpt_tore`) must be compiled before running the application.

**Using CMake:**

```bash
mkdir build && cd build
cmake ../app/src_cpp
make
# Copy the resulting .so (Linux/Mac) or .pyd (Windows) to the app/ directory
cp sculpt_tore* ../app/
```

### 3. Run the Application
Navigate to the root directory and execute the utility entry point as a module.

```bash
python -m app.utils.main
```

## Usage Guide

### Alignment
1. Load a scan (STL/PLY).
2. Select **Align Horizontal**.
3. Click 3 points: Left Molar Cusp, Right Molar Cusp, and Central Incisor Edge.
4. The mesh re-orients to the Z-axis.

### Base Generation
1. Select the **Maxilla** or **Mandible** workflow.
2. Use the **Manual Marking Tool** to trace the gumline.
3. Adjust **Height** and **Skirt** parameters.
4. Click **Generate Base** (Maxilla extrudes Up, Mandible extrudes Down).

### Sculpting
1. Toggle **Enable Sculpting**.
2. Choose **Add**, **Remove**, or **Smooth** modes.
3. Enable **Dynamic Topology** to automatically refine low-poly areas of the scan while brushing.

## Theoretical Framework

The current version utilizes **Extrinsic Euclidean Deformation (Gaussian RBF)** for sculpting.

### Future Roadmap (Intrinsic Methods)
To improve fidelity, the system is transitioning to Differential Geometry approaches:

* **Discrete Laplacian Editing:** Preserves local surface details by encoding vertex coordinates relative to their neighbors using Cotangent Weights.
* **As-Rigid-As-Possible (ARAP):** Ensures teeth remain rigid (rotate/translate) while soft tissue deforms.
* **Heat Method:** For computing Geodesic distance to prevent artifacts between spatially adjacent but topologically distinct features.
