Dental Wizard: Hybrid Dental CAD/CAM Framework
Dental Wizard is a specialized Computer-Aided Design (CAD) application designed for the processing, modification, and preparation of dental 3D scans. It features a hybrid architecture, leveraging Python for clinical workflow management and a C++ kernel for high-performance geometric sculpting.

##Key Features
Clinical Workflows
Automated Alignment: 3-Point pick system (Molars + Incisor) to normalize mesh orientation to the occlusal plane.

Maxillary & Mandibular Modes: Dedicated pipelines for upper and lower arches, handling base extrusion direction (Up/Down) automatically.

Feature Recognition: Tools to mark and extract critical regions like the Incisive Papilla and Retromolar Pads.

Undercut Analysis: Real-time surveyor to visualize insertion paths and identify undercut zones.

##High-Performance Sculpting
Dynamic Topology (Dyntopo): The C++ backend (sculpt_tore) dynamically subdivides the mesh during sculpting, adding resolution only where detail is needed.

Custom Interaction: specialized VTK interaction styles that separate camera manipulation from mesh deformation strokes.

##System Architecture
The project follows a Host-Guest architecture pattern:

Host (Python 3.10+): Handles state management, UI (PySide6), Visualization (PyVista/VTK), and File I/O.

Guest (C++17): Acts as the compute kernel, handling vertex-level operations, topological graph rebuilding, and spatial queries.

Directory Structure
Plaintext
app/
├── core/
│   ├── commands.py       # Undo/Redo command pattern implementation
│   └── interactors.py    # Custom VTK Interactor styles
├── tools/
│   ├── bezier.py         # Bezier curve marking tool
│   ├── border_tool.py    # Border deformation logic
│   ├── sculptor.py       # Python wrapper for the C++ sculpting engine
│   └── surveyor.py       # Undercut visualization tool
├── ui/
│   ├── dental_wizards.py # Workflow orchestration (Maxilla/Mandible steps)
│   ├── hierarchy.py      # Scene graph and object management
│   ├── main_window.py    # Primary application window
│   └── dialogs.py        # Pop-up dialogs and prompts
├── utils/
│   ├── config.py         # Application configuration
│   ├── main.py           # Application entry point
│   └── __init__.py
└── src_cpp/              # C++ Source Code
    ├── sculpt_tore.cpp   # Dynamic Topology Engine
    └── CMakeLists.txt    # Build configuration
Installation & Build
Prerequisites
Python 3.10+

C++ Compiler (GCC 9+, Clang 10+, or MSVC 2019+)

CMake 3.15+

1. Install Python Dependencies
Bash
pip install numpy pyvista vtk PySide6 pybind11
2. Compile the C++ Kernel
The sculpting engine (sculpt_tore) must be compiled before running the application.

Using CMake:

Bash
mkdir build && cd build
cmake ../src_cpp
make
# Copy the resulting .so (Linux/Mac) or .pyd (Windows) to the app/ root
cp sculpt_tore* ../app/
3. Run the Application
Navigate to the app folder and execute the utility entry point.

Bash
python -m utils.main
Usage Guide
Alignment
Load a scan (STL/PLY).

Select Align Horizontal.

Click 3 points: Left Molar Cusp, Right Molar Cusp, and Central Incisor Edge.

The mesh re-orients to the Z-axis.

Base Generation
Select the Maxilla or Mandible workflow.

Use the Manual Marking Tool to trace the gumline.

Adjust Height and Skirt parameters.

Click Generate Base. (Maxilla extrudes Up, Mandible extrudes Down).

Sculpting
Toggle Enable Sculpting.

Choose Add, Remove, or Smooth modes.

Enable Dynamic Topology to automatically refine low-poly areas of the scan while brushing.
