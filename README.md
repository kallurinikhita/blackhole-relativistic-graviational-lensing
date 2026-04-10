# Black Hole MVP: Physics-Informed Neural Rendering of Gravitational Lensing

## Overview

This project builds a **physics-informed machine learning pipeline** to simulate and visualize **gravitational lensing near black holes** in real time.

Instead of solving full general relativity equations, I construct a:

> **Surrogate model that learns how mass distributions distort light**

The system combines:

* astrophysical simulation principles
* computational physics approximations
* deep learning (CNN / U-Net)
* real-time visualization

---

## Core Idea

Approximate the mapping:

```
Mass Distribution (Σ, Φ) → Light Deflection Field (Δx, Δy)
```

Where:

* **Σ (Sigma)** = surface mass density
* **Φ (Phi)** = gravitational potential
* **Δu = (Δx, Δy)** = lensing distortion field

This enables **real-time rendering of relativistic visual effects** such as:

* gravitational lensing
* photon ring distortion
* accretion disk warping

---

## Project Structure

```
blackhole-mvp/
│
├── data/
│   ├── synthetic_images/        # Base images (stars, grids, disks)
│   ├── warped_images/           # Lensed outputs (ground truth)
│
├── physics/
│   ├── lensing_model.py         # Lensing field computation (∇Φ)
│   ├── distortion_equations.py  # Physics approximations (gravity, redshift)
│
├── ml/
│   ├── model.py                 # CNN / U-Net models
│   ├── train.py                 # Training pipeline
│   ├── dataset.py               # Dataset creation + loading
│
├── viz/
│   ├── streamlit_app.py         # Interactive UI
│   ├── renderer.py              # Image warping + visualization
│
├── utils/
│   ├── generate_dataset.py      # Synthetic / simulation dataset generator
│
└── main.py                      # Entry point (end-to-end pipeline)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/blackhole-mvp.git
cd blackhole-mvp
```

### 2. Create environment

```bash
conda create -n blackhole python=3.9
conda activate blackhole
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Generate dataset

```bash
python utils/generate_dataset.py
```

This creates:

* mass density maps (Σ)
* potential fields (Φ)
* lensing distortion maps (Δu)

---

### 2. Train model

```bash
python ml/train.py
```

Trains a CNN / U-Net to learn:

```
(Σ, Φ) → (Δx, Δy)
```

---

### 3. Visualize results

```bash
streamlit run viz/streamlit_app.py
```

Interactively:

* adjust black hole strength
* visualize lensing effects
* simulate “falling into a black hole”

---

## Physics Modeling

This project uses **approximate physics**:

### Gravitational Potential

```
Φ ≈ Σ * (1 / r)
```

### Lensing Deflection

```
Δu = ∇Φ
```

### Optional Effects

* Doppler shift
* gravitational redshift
* stochastic “quantum” perturbations (experimental)

---

## Machine Learning

### Model

* CNN (baseline model)
* U-Net (explored)

### Input

* 2-channel field: (Σ, Φ)

### Output

* 2-channel field: (Δx, Δy)

### Loss

* Mean Squared Error (MSE)
* optional smoothness regularization

---

## Evaluation

I evaluate:

* **Reconstruction Error**

  * L2 difference between predicted and true lensing field

* **Visual Fidelity**

  * realistic distortion patterns
  * photon ring approximation

* **Performance**

  * inference speed (real-time capable)

---

## Scientific Scope

This is a **physics-informed visualization system**, not a full GR solver.

I explicitly approximate:

* ✔ weak-field gravitational lensing
* ✔ surrogate ray deflection
* ✔ post-processing visualization

I do **NOT** solve:

* Einstein field equations
* Kerr metric geodesics
* full relativistic magnetohydrodynamics

---

## Future Work

* Integrate real simulation data (e.g., IllustrisTNG)
* Replace approximations with learned GR surrogate models
* GPU acceleration (CUDA)
* Integration with NVIDIA Omniverse for rendering
* Real-time immersive visualization (VR / AR)

---

## Research Motivation

This project explores:

> **How machine learning can approximate complex physical systems for real-time scientific visualization**

Applications:

* astrophysics education
* scientific visualization pipelines
* ML surrogates for expensive simulations
