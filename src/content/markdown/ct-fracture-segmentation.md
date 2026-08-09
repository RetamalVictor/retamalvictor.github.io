---
title: "From CT Scan to Fracture Map: A Practical Guide to 3D Medical Image Segmentation"
date: "2026-05-02"
tags: ["medical-imaging", "3d-segmentation", "deep-learning", "nnunet"]
summary: "How I built a pelvic fracture segmentation pipeline - and the single design decision that mattered more than the model. ABBC methodology, two-stage architecture with nnU-Net, and what actually matters when training on 3D medical data."
readTime: "25 min"
featured: true
---

*How I built a pelvic fracture segmentation pipeline - and the single design decision that mattered more than the model.*

---

## The Thesis, Up Front

I spent weeks on this project. The model architecture wasn't what made it work. The loss function wasn't what made it work. The GPU wasn't what made it work.

**What made it work was changing what I asked the network to predict.**

Instead of asking a neural network to label 30 arbitrary fragment classes, I asked it to predict 4 geometric properties - interior, surface, fracture line, background. From those 4 predictions, I recovered 30 labeled fragments using classical algorithms. This single design decision - the choice of **prediction target** - was the difference between a pipeline that couldn't learn and one that achieved state-of-the-art results.

This post walks through the full pipeline. But keep that thesis in mind: when you're stuck on a segmentation problem, before reaching for a bigger model or more data, ask yourself if you're asking the network the right question.

---

## 1. The Problem: Pelvic Fracture Segmentation

A patient arrives at a trauma center after a car crash. The CT scan reveals a shattered pelvis - the ring of bone that connects spine to legs, broken into a dozen fragments. The orthopedic surgeon has hours to plan a reconstruction: which fragments to plate, where to place screws, how to restore the ring's geometry.

The first step is understanding the fracture. A radiologist sits down with the CT volume - 400+ axial slices, each a cross-section of the pelvis - and manually traces each fragment, slice by slice. This takes 3-4 hours per case. It's tedious, error-prone, and it happens while the patient is waiting for surgery.

In many surgical planning workflows, the next step is even more manual: those traced fragments get converted into 3D meshes - STL files that can be loaded into planning software or 3D-printed as physical models. The surgeon holds the shattered pelvis in their hands, plans the reconstruction on the physical model, then walks into the operating room. The bottleneck isn't the surgery; it's the hours of manual segmentation and mesh generation that precede it.

This is the problem I'm trying to solve: **given a 3D CT volume of a fractured pelvis, automatically segment every bone fragment** - and produce the 3D meshes that surgeons actually use.

Here are real segmentation results from my pipeline - each colored mesh is a separate bone fragment, automatically extracted from a CT scan:

<div id="stl-viewer-demo" class="my-8"></div>

<div id="youtube-video" class="my-8"></div>

### The anatomy

The pelvis is three bones fused into a ring: the **sacrum** (base of the spine) at the top, connected to the **left hip** and **right hip** bones on either side. The 3D viewer above shows this structure - sacrum fragments in red tones, left hip in green, right hip in blue.

In a high-energy fracture, each bone can shatter into up to 10 fragments. The challenge labels these 1-10 (sacrum), 11-20 (left hip), 21-30 (right hip), for up to 30 fragments total. Label 0 is background - everything that isn't bone. The side panels in the viewer above show the isolated fragments for each bone region.

### The challenge

The **PENGWIN Challenge** (PElvis aNalysis With INtelligence) formalizes this task as a competition. The primary metric is **IoU-F** - fragment-wise Intersection over Union. Not just "did you find bone?" but "did you correctly separate fragment 3 from fragment 4?"

The 2024 edition had 100 training cases. The winning team achieved IoU-F = 0.9296. The 2026 edition expands to 500 clinical cases plus 16,000 simulated fractures.

---

## 2. What Makes 3D Medical Segmentation Hard

If you've worked with 2D image segmentation (COCO, Cityscapes), 3D medical volumes introduce several new challenges:

### Scale

A pelvic CT is typically $512 \times 512 \times 400$ voxels - **105 million values**. You can't fit the full volume through a network. Instead, you train on random 3D patches (e.g., $160 \times 160 \times 160$) and stitch predictions together at inference. The patch size - how much context the network can see - is one of the most impactful hyperparameters.

<div id="scale-comparison" class="my-8"></div>

### Non-square voxels

CT voxels aren't cubes. A typical pelvic scan has spacing $0.82 \times 0.82 \times 1.5$ mm - the slice axis (Z) is nearly 2x coarser than in-plane (X, Y). Every distance calculation (convolutions, distance transforms, loss functions) must account for this anisotropy.

<div id="voxel-anisotropy" class="my-8"></div>

### Small datasets

Medical datasets are small by ML standards. PENGWIN 2024 had 100 cases. Even PENGWIN 2026's 500 clinical cases would be considered tiny in computer vision. This makes framework-level decisions (augmentation, normalization, patch sampling) matter more than in data-rich settings.

### The pretrained model landscape is changing

A year ago, 3D medical segmentation meant training from scratch. Today, foundation models like **TotalSegmentator** (104 anatomical structures, 1,200+ CTs) and **STU-Net** (pretrained on TotalSegmentator, up to 1.4B parameters) offer pretrained starting points. I use TotalSegmentator as a zero-shot baseline for anatomical bone segmentation in my pipeline - no training required.

<details>
<summary>Deep Dive: 3D Segmentation Architectures (U-Net, nnU-Net, ResEncL)</summary>

### What Is Medical Image Segmentation?

Segmentation means assigning a class label to every pixel (2D) or voxel (3D) in an image. In medical imaging, that means: "this voxel is liver," "this voxel is tumor," "this voxel is background."

The output is a **label map** - the same dimensions as the input, where each element is an integer class ID.

### The U-Net: Why It Dominates Medical Segmentation

The U-Net (Ronneberger et al., 2015) [1] has been the dominant architecture in medical image segmentation for over a decade. It was designed specifically for biomedical images, and its core ideas remain the foundation of every state-of-the-art method in the field.

A segmentation network needs to answer two questions simultaneously:
- **What** is in this region? (requires abstract, high-level features)
- **Where** exactly is the boundary? (requires precise, pixel-level spatial information)

These goals are in tension. The U-Net solves this with an **encoder-decoder** architecture. The **encoder** progressively compresses spatial resolution while increasing channel depth - by the bottleneck, the network has a global view of the input. The **decoder** reverses this, progressively upsampling back to the original resolution.

**Skip connections** concatenate encoder features with decoder features at each resolution level. The decoder gets the best of both worlds: the encoder's sharp spatial detail and the decoder's semantic understanding. This is why the architecture is shaped like a "U."

Three properties made U-Net ideal for medical data:
1. **Works with small datasets.** Skip connections preserve information flow, reducing the amount of data needed.
2. **Dense prediction.** Every voxel gets a label - no region proposals or post-hoc stitching.
3. **Flexible depth.** The number of stages can be tuned to match GPU memory and structure scale.

### Going to 3D

Every 2D operation becomes its 3D counterpart: $3 \times 3$ kernels become $3 \times 3 \times 3$ (27 weights instead of 9), $2 \times 2$ max pooling becomes $2 \times 2 \times 2$. The parameter count per layer increases by 3x, and memory required for activations increases dramatically.

The solution: train on **random patches** of the volume, not the whole thing. At inference, slide overlapping patches across the full volume and average predictions in the overlap regions. nnU-Net handles this automatically with Gaussian-weighted blending.

### nnU-Net: The Self-Configuring Framework

**nnU-Net** (no-new-Net, Isensee et al., 2021) [2] is the most important practical tool in 3D medical segmentation. It is not a single architecture - it is a **framework** that automatically configures a U-Net for your specific dataset.

Given your data, it determines: target resampling spacing, normalization strategy, patch size and batch size, network depth and channels, and whether to use 2D, 3D full-res, 3D low-res, or a cascade.

The key finding: **systematic hyperparameter selection beats architectural innovation**. A properly configured standard U-Net outperforms novel architectures that are sub-optimally configured.

### Residual Encoder: Scaling Up

The "nnU-Net Revisited" paper (Isensee et al., 2024) [3] asked: what if we just make the U-Net bigger?

A residual block adds a shortcut connection from input to output:

$$\text{output} = \text{ReLU}(\text{Conv}_2(\text{ReLU}(\text{Conv}_1(\text{input}))) + \text{input})$$

The shortcut lets gradients flow directly during backpropagation, enabling much deeper networks. **ResEncL consistently beat Transformer architectures** (SwinUNETR, nnFormer) and state-space models (U-Mamba) on standard medical segmentation benchmarks. For 3D volumes with 100-1000 training cases, scaled CNNs still win.

### The Loss Function: Dice + Cross-Entropy

Almost every competitive medical segmentation system uses Dice + CE:

$$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}$$

**Cross-entropy** provides stable gradients everywhere but treats all voxels equally (background-dominated). **Dice loss** measures per-class overlap and naturally handles class imbalance, but has noisy gradients early in training. Together, they compensate for each other's weaknesses.

</details>

---

## 3. Why Direct Prediction Fails

The simplest approach to fracture segmentation: train a network with 31 output channels (background + 30 fragments), softmax, Dice + cross-entropy loss. This is standard semantic segmentation - the same approach you'd use for organs or tumors.

It doesn't work for fragments. Here's why:

### Arbitrary labels

In one training case, the largest sacrum fragment is labeled "1." In another case with a different fracture pattern, a completely different region gets labeled "1." There is no semantic consistency.

```python
# The naive approach: 31-class semantic segmentation
model = UNet(in_channels=1, out_channels=31)  # bg + 30 fragments
pred = model(ct_volume)                        # shape: [31, D, H, W]
labels = pred.argmax(dim=0)                    # fails: class "1" means 
                                               # different things per patient
```

Contrast this with organ segmentation, where "liver" always means liver. The network can learn a prior for what a liver looks like. It cannot learn a prior for what "fragment 3" looks like, because fragment 3 looks different in every patient.

---

### Variable instance count

One patient has 3 fragments, another has 15. The network must allocate 30 output channels, but most are unused in any given case - typically 25+ classes have zero voxels. This wastes capacity and creates a severe class imbalance.

> **The math of the imbalance**: With 30 fragment channels and only 5 actual fragments, 25 channels have zero training signal. The loss is dominated by "correctly predicting nothing" in 83% of the output channels.

---

### No geometric prior

The network has no incentive to learn that a voxel deep inside a bone fragment is different from a voxel on the fracture surface. But this geometric distinction is exactly what separates fragments. Two adjacent fragments are distinguishable only at their contact zone - the fracture line.

---

### The general lesson

> This isn't unique to fractures. It's a fundamental problem in **instance segmentation** - separating individual objects of the same type. When you can't assign consistent class meanings across examples, semantic segmentation breaks down.

Detection-based approaches (Mask R-CNN) struggle in 3D volumes: fragment "bounding boxes" overlap extensively, fragment sizes span 3 orders of magnitude, and the number of instances varies widely.

The winning approach takes a different path entirely.

---

## 4. ABBC: Predict Geometry, Recover Instances

This section is the core of the post. Everything else is infrastructure around this idea.

### The representation

**ABBC** (Adaptive Border Boundary Core) was introduced by the MIC-DKFZ team, winners of PENGWIN 2024 with IoU-F = 0.9296 [5]. Instead of predicting 31 fragment classes, the network predicts 4 geometric classes:

| Class | Name | Meaning |
|-------|------|---------|
| 0 | Background | Not bone (soft tissue, air, organs) |
| 1 | Boundary | Outer bone surface and shallow interior |
| 2 | Core | Deep interior of each fragment |
| 3 | Border | Fracture contact zone - where fragments nearly touch |

**Why this works**: "Core" means the same thing in every patient - deep fragment interior. "Border" means the same thing - fracture contact zone. The network learns geometry, not arbitrary IDs. And crucially, each **connected component** of the core class corresponds to exactly one fragment. Instance recovery becomes a classical algorithm problem, not a learning problem.

<div id="abbc-cross-section" class="my-8"></div>

### Generating ABBC labels: from fragment IDs to geometry

Given ground truth fragment labels, how is the ABBC representation computed? Three steps, each building on the previous.

#### Step 1: Distance Transform - how deep is each voxel?

For each fragment, compute the **Euclidean Distance Transform (EDT)** - the distance from every interior voxel to the nearest surface:

$$\text{EDT}(\mathbf{x}) = \min_{\mathbf{y} \in \partial F} \|\mathbf{x} - \mathbf{y}\|_2$$

```python
# Step 1: Distance transform for each fragment
for frag_id in unique_fragments:
    mask = (labels == frag_id)
    edt[mask] = distance_transform_edt(mask, sampling=voxel_spacing)
    # edt values: 0 at surface, increasing toward interior
```

Voxels at the bone surface have EDT = 0. Voxels deep inside have high EDT. The EDT is computed in physical coordinates (millimeters, not voxels) to account for anisotropic spacing.

---

#### Step 2: Laplacian - finding the skeleton adaptively

I could classify "core" as simply "EDT > some threshold." But what threshold?

> **The thin-bone problem**: The iliac wing is only 4 voxels thick. A fixed threshold of 6mm would give it zero core voxels. No core = no seed = no instance recovery.

The solution uses the **Laplacian** of the smoothed distance field. At the **medial axis** (the geometric "skeleton" of a shape), the distance field forms a ridge. The Laplacian is strongly negative there.

$$\nabla^2 \text{EDT}_\sigma = \frac{\partial^2}{\partial x^2} + \frac{\partial^2}{\partial y^2} + \frac{\partial^2}{\partial z^2} \;\text{of}\; (G_\sigma * \text{EDT})$$

The **adaptive core rule** combines curvature and distance:

$$\text{Core}(\mathbf{x}) = \big[-\nabla^2 \text{EDT}_\sigma(\mathbf{x}) > 0.11\big] \;\lor\; \big[\text{EDT}(\mathbf{x}) > 6\big]$$

```python
# Step 2: Adaptive core detection
edt_smooth = gaussian_filter(edt, sigma=2.0)
laplacian = laplace(edt_smooth)

core = np.zeros_like(labels, dtype=bool)
core |= (-laplacian > 0.11)   # curvature: catches thin fragments
core |= (edt > 6.0)           # distance: catches thick fragments
core &= (labels > 0)          # must be inside bone
```

- **Curvature condition** ($-\nabla^2 > 0.11$): catches thin fragments where max EDT is only 2, but the medial axis has high curvature.
- **Distance condition** (EDT > 6): catches thick fragments where any voxel more than 6mm from the surface is definitely interior.

> **Guarantee**: Every fragment gets at least one core voxel. This is verified by 34 unit tests in the label generation pipeline.

---

#### Step 3: Fracture detection - where do fragments meet?

Border voxels mark fracture contact zones. A bone voxel is in the fracture zone if it's within 6 voxels of two or more different fragments:

$$\text{fracture}(\mathbf{x}) = \sum_{i} \mathbf{1}\big[\text{dist}(\mathbf{x}, F_i) < r\big] > 1 \qquad r = 6$$

```python
# Step 3: Fracture zone detection
fracture_zone = np.zeros_like(labels, dtype=bool)
for frag_id in unique_fragments:
    inv_edt = distance_transform_edt(labels != frag_id, sampling=spacing)
    nearby = (inv_edt < 6.0)
    fracture_zone |= nearby & (labels > 0) & (labels != frag_id)

# In fracture zone: boundary -> border, but preserve at least 1 core per fragment
border = fracture_zone & boundary_mask
```

In the fracture zone, boundary voxels are promoted to border. Core voxels near fracture lines are demoted to boundary - but never all of them. Every fragment keeps at least one core voxel.

---

### From 4 classes back to 30 fragments

The ABBC representation makes the prediction problem learnable. But the challenge expects labels 0-30. I need to convert 4 geometry classes back to instance labels. This is a three-stage classical algorithm - no learning required.

#### Stage A: Seeds from core

Each connected component of the predicted core class is one fragment seed. I use 26-connectivity (including diagonals) and filter out components smaller than 50 voxels (noise):

```python
core_mask = (abbc_prediction == CORE)
seeds = cc3d.connected_components(core_mask, connectivity=26)
seeds = cc3d.dust(seeds, threshold=50)  # remove noise
```

#### Stage B: Expanding seeds with Fast Marching

Each seed is a small blob deep inside a fragment. It has to be expanded to fill the entire fragment. But "nearest" must respect anatomy: the expansion can't shortcut through a fracture line.

> **Why not Euclidean distance?** In curved bone geometry, the Euclidean nearest seed might be across a fracture line - through empty space or through another fragment. FMM follows the bone surface, respecting the actual topology.

The **Fast Marching Method (FMM)** solves the Eikonal equation - wavefronts expand from each seed through bone tissue. **Border voxels are walls.** Waves cannot cross fracture lines.

$$|\nabla T(\mathbf{x})| = \frac{1}{F(\mathbf{x})}$$

```python
# Stage B: Geodesic expansion via Fast Marching
speed = np.ones_like(labels, dtype=float)
speed[pred == BORDER] = 0.0     # fracture lines are impassable
speed[labels == 0] = 0.0        # background is impassable

# Initialize: seed voxels have T=0, everything else is unknown
phi = np.full_like(labels, dtype=float, fill_value=1e10)
for seed_id in unique_seeds:
    phi[seeds == seed_id] = 0.0

# FMM expands wavefronts; each voxel assigned to nearest seed
arrival_time = skfmm.travel_time(phi, speed, dx=voxel_spacing)
instances = assign_to_nearest_seed(arrival_time, seeds)
```

Each bone voxel is assigned to the seed whose wave arrives first - a **geodesic Voronoi diagram**.

---

#### Stage C: Anatomy assignment

The final step maps generic instance IDs to PENGWIN canonical labels (sacrum 1-10, left hip 11-20, right hip 21-30):

```python
# Stage C: Map instances to anatomical labels
for inst_id in unique_instances:
    if voxel_count(inst_id) < 1000:
        merge_into_nearest(inst_id)          # remove tiny fragments
        continue
    region = majority_vote(inst_id, anatomy)  # sacrum / L.hip / R.hip
    rank = size_rank(inst_id, region)         # largest = 1
    final_label = region_offset[region] + rank
```

1. **Merge small fragments** (< 1000 voxels) into their nearest neighbor
2. **Majority vote**: each instance is assigned to the anatomical region where most of its voxels fall
3. **Canonical numbering**: within each region, number by size descending

### The full ABBC pipeline

<div id="pipeline-diagram" class="my-8"></div>

---

## 5. The Two-Stage Pipeline

The ABBC pipeline needs an anatomical segmentation to assign fragments to bone regions. The result is a two-stage system: first identify the bones, then segment the fragments.

### Why two models?

**All top-5 PENGWIN 2024 teams used two stages.** The challenge summary paper found a Pearson correlation of r = 0.870 between anatomical accuracy and final fragment accuracy [5]. Getting the bone regions right is the foundation.

Why not one model with 7 output classes? Two reasons:

1. **Different spatial scales.** Anatomical segmentation is a coarse, whole-volume task (which blob is the sacrum?). Fragment segmentation is fine-grained (where exactly is the fracture line, down to 1-2 voxels?).
2. **Independent development.** Stage 1 can be a pretrained model (TotalSegmentator, zero training). Stage 2 can be trained, retrained, and tuned independently.

<div id="architecture-diagram" class="my-8"></div>

### Stage 1: Anatomical bone segmentation

I evaluated **TotalSegmentator** [4] - a pretrained model that segments 104 anatomical structures - as a zero-shot Stage 1:

| Structure | Dice | HD95 |
|-----------|------|------|
| Sacrum | 0.817 ± 0.016 | 9.9 ± 1.1 mm |
| Left hip | 0.929 ± 0.010 | 1.0 ± 0.2 mm |
| Right hip | 0.929 ± 0.013 | 1.0 ± 0.2 mm |

Hips are excellent (~1mm surface error). The sacrum is weaker - 9.9mm HD95 at fracture boundaries, because TotalSegmentator wasn't trained on shattered bone. For a production system, fine-tuning on fracture data would close this gap.

### Error propagation

What happens when Stage 1 is wrong? If the anatomical model misclassifies a region of sacrum as left hip, the anatomy vote in Stage C will assign fragments to the wrong bone. The fragment boundaries will still be correct (ABBC doesn't depend on anatomy), but the final labels will be wrong.

This is the main risk of two-stage systems: errors compound. The mitigation is making Stage 1 very accurate - which is easier than making the full 31-class problem accurate, because anatomical segmentation has consistent labels across patients.

---

## 6. Training: What Actually Matters

### The framework: nnU-Net

I use **nnU-Net** [2] - a framework that automatically configures a U-Net for your dataset. No manual tuning required:

```bash
# That's it. nnU-Net figures out everything else.
nnUNetv2_plan_and_preprocess -d 003 --verify_dataset_integrity
nnUNetv2_train 003 3d_fullres 0 -p nnUNetResEncUNetLPlans
```

Given your data, it determines patch size, network depth, spacing, normalization, batch size, and augmentation. The model variant is **ResEncL** (Residual Encoder, Large) [3] - a residual encoder U-Net that pushes the limits of 24GB GPU memory.

> **Key finding [3]**: A properly configured, scaled CNN (ResEncL) consistently beats Transformer architectures (SwinUNETR, nnFormer) and state-space models (U-Mamba) for 3D medical segmentation with < 1000 training cases.

---

### Loss function: Dice + Cross-Entropy

The loss combines two terms that compensate for each other's weaknesses:

$$\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}$$

**Cross-entropy** is the standard per-voxel classification loss:

$$\mathcal{L}_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} g_{ic} \log(p_{ic})$$

It provides a gradient for every voxel, everywhere, all the time. But when 95% of voxels are background, the network can minimize CE loss by mostly predicting background.

**Dice loss** measures overlap per-class:

$$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum_{i} p_{ic} \, g_{ic}}{\sum_{i} p_{ic} + \sum_{i} g_{ic} + \epsilon}$$

If the border class (< 1% of the volume) is 50% wrong, Dice notices even though CE barely cares.

### Training dynamics

<div id="training-chart-demo" class="my-8"></div>

Training on 100 cases (ABBC labels, fold 0 of 5-fold cross-validation, ResEncL planner):

| Epoch | Boundary Dice | Core Dice | Border Dice | EMA Dice |
|-------|--------------|-----------|-------------|----------|
| 50 | 0.84 | 0.85 | 0.45 | 0.71 |
| 150 | 0.89 | 0.90 | 0.60 | 0.78 |
| 250 | 0.91 | 0.92 | 0.72 | 0.81 |
| 385 | ~0.92 | ~0.93 | 0.72-0.76 | 0.860 |

> **Border is the hardest class** - it occupies 1-3 voxels at fracture lines, making it both rare (class imbalance) and difficult (requires precise localization). It's also the most clinically important: the fracture line is what separates fragments. At epoch 385, border Dice is still climbing - justifying long training runs (the original DKFZ team trained for 1000 epochs).

---

### nnU-Net training details

**Patch sampling with oversampling.** One-third of patches in each batch are guaranteed to contain foreground. For my ABBC task, the border class (< 1% of the volume) appears in training patches far more often than random sampling would provide.

---

**Deep supervision.** The network produces predictions at multiple decoder resolutions. The total loss is a weighted sum:

$$\mathcal{L}_{\text{total}} = \sum_{s=0}^{S} w_s \, \mathcal{L}^{(s)}$$

where weights $w_s$ decrease for lower-resolution stages ($1, 0.5, 0.25, \ldots$).

---

**Learning rate schedule.** Polynomial decay - no warm-up, no restarts:

$$\eta(e) = \eta_0 \cdot \left(1 - \frac{e}{E}\right)^{0.9}$$

---

**Augmentation.** Critical with only 100 training cases:

```python
# nnU-Net applies all of these on-the-fly per patch:
transforms = [
    RandomRotation(axes=[0,1,2]),       # all three axes
    ElasticDeformation(),                # smooth warping
    RandomScale(range=(0.7, 1.4)),       # size variation
    GaussianNoise() + GaussianBlur(),    # intensity variation
    GammaCorrection(),                   # contrast variation
    MirrorTransform(axes=[0,1,2]),       # flips
]
```

### The cost of a bigger model

| Planner | Epoch time | 1000 epochs | EMA Dice (fold 0) |
|---------|-----------|-------------|-------------------|
| Default | 34s | ~9.4 hours | good (baseline) |
| ResEncL | 119s | ~33 hours | 0.860 (better) |

I train on **RunPod** cloud GPUs - RTX 4090 (24GB) at $0.50/hour. A competitive pipeline from scratch costs under $100 in cloud compute.

<details>
<summary>Deep Dive: Segmentation Metrics - Dice, IoU, HD95</summary>

### Why Metrics Matter More Than You Think

Two segmentation models can have near-identical Dice scores but wildly different clinical utility. One might miss a thin boundary by 1 voxel everywhere (low surface error, slight overlap loss). The other might perfectly segment 95% of the structure but completely miss a protruding region (low overlap error, catastrophic surface error).

### Dice Coefficient (F1 Score)

The most common metric in medical image segmentation. Measures the overlap between predicted region $P$ and ground truth region $G$:

$$\text{Dice}(P, G) = \frac{2|P \cap G|}{|P| + |G|}$$

- **Range**: 0 (no overlap) to 1 (perfect overlap)
- **Symmetric**: $\text{Dice}(P, G) = \text{Dice}(G, P)$
- **Related to precision and recall**: $\text{Dice} = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$

**When Dice misleads**: A liver (100,000 voxels) with 500 voxels misclassified gets Dice $\approx$ 0.995. A small lymph node (500 voxels) with 50 voxels misclassified gets Dice $\approx$ 0.905. Always report Dice per class.

### Intersection over Union (IoU / Jaccard Index)

$$\text{IoU}(P, G) = \frac{|P \cap G|}{|P \cup G|}$$

Related to Dice: $\text{IoU} = \frac{\text{Dice}}{2 - \text{Dice}}$. They rank models identically. IoU is standard in computer vision (COCO, Pascal VOC) and is the primary metric in PENGWIN.

**IoU-F (fragment-wise)**: For instance segmentation, predicted fragments are matched to ground truth fragments by maximum IoU. Penalizes both over-segmentation (splitting) and under-segmentation (merging).

### Hausdorff Distance (HD95)

Dice and IoU measure **volume overlap**. Hausdorff distance measures **surface accuracy**:

$$\text{HD}(A, B) = \max\big(\max_{a \in A} \min_{b \in B} \|a - b\|_2, \; \max_{b \in B} \min_{a \in A} \|a - b\|_2\big)$$

**HD95** uses the 95th percentile instead of the maximum, making it robust to small outliers. A surgeon planning a bone cut needs to know the boundary location within ~2mm - HD95 tells you this directly.

### When Metrics Disagree

| Scenario | Dice | HD95 | What happened |
|----------|------|------|--------------|
| Good prediction | High | Low | Everything works |
| Uniform over-segmentation | Medium | Low | Boundaries shifted but smooth |
| Missing region | High | High | Most correct, one lobe missed |
| Fragmented prediction | Low | Low | Broken into pieces, each near boundary |

**The golden rule**: report both an overlap metric (Dice or IoU) and a distance metric (HD95).

</details>

---

## 7. What I Learned

### The prediction target is the most important design decision

> The PENGWIN 2024 results tell the story: 1st place (MIC-DKFZ, IoU-F = 0.9296) and 4th place (MedApp-AGH) both used nnU-Net. Same architecture, same loss, same augmentation. **The difference was what the model predicted**: DKFZ used ABBC; MedApp-AGH used direct fragment prediction.

When you're stuck on a segmentation problem - especially instance segmentation - the first question shouldn't be "what model should I use?" It should be "what should the model predict?"

---

### Test your label generation as rigorously as your model

My ABBC label generation pipeline has **34 unit tests**:

```python
# Critical test: every fragment must retain at least one core voxel
def test_every_fragment_has_core():
    for frag_id in unique_fragments:
        core_count = np.sum(abbc[labels == frag_id] == CORE)
        assert core_count >= 1, f"Fragment {frag_id} has no core voxels"
```

I caught a bug early where demotion of core voxels near fracture zones could remove all core from thin fragments. Without this test, it would have silently produced fragments with no seeds - invisible during training, catastrophic during inference.

---

### Postprocessing is half the system

The neural network outputs 4 probability maps. Getting from there to 30 labeled fragments requires connected components, FMM, EDT, majority voting, and size filtering.

> **Budget rule**: If you budget 2 weeks for the neural network and 2 days for postprocessing, you'll regret it. Budget equal time. My postprocessing pipeline has its own parameters, failure modes, and test suite (16 tests).

---

### Start with pretrained models where you can

```bash
# This gave me a working Stage 1 with Dice > 0.93 on hips
pip install totalsegmentator
TotalSegmentator -i patient_ct.nii.gz -o segmentation/ --task total
```

No training, no data preparation, no hyperparameter tuning. The hip results were immediately usable.

---

### The model is not the product

The hardest lesson: between a numpy array of labels and a surgeon's 3D planning tool:

| Gap | Detail |
|-----|--------|
| Input format | Clinical scanners export DICOM, not NIfTI |
| Voxel spacing | Clinical scans: 0.5-5mm. Model expects ~0.8mm |
| Output format | Surgeons need STL meshes, not label arrays |
| Robustness | One crash on unexpected input kills demo credibility |

None of this is ML work. All of it is essential.

---

### Domain shift is real - and predictable

My model was trained on 100 research CTs at ~0.8mm isotropic spacing. The real risk is **slice thickness** - you can't segment a 1-voxel fracture line if your voxels are 5mm thick.

> **Mitigation**: Get sample data from the target clinical environment early. Run it through the pipeline before committing to a demo. nnU-Net makes fine-tuning straightforward - add the new cases, re-run training, same pipeline.

---

### Cloud training needs resilience engineering

```bash
# My checkpoint sync runs every 5 minutes in the background
while true; do
    rsync checkpoints/ hf://username/model-repo/ 
    sleep 300
done
```

- **Preflight validation** checks GPU, CUDA, disk space, and dataset integrity before training starts.
- **Dependency pinning** is non-trivial. One dependency (acvl-utils) broke with setuptools >= 75.

> Infrastructure code is production code. Treat it accordingly.

---

## 8. Getting Started

If you want to try 3D medical segmentation, here's the path I'd recommend:

### Start with nnU-Net on an established benchmark

Don't start with your own data. Start with a dataset that has known good results:
- **BTCV** (multi-organ abdominal CT, 30 cases) - small enough to train fast
- **Medical Segmentation Decathlon** (10 tasks, diverse organs) - good for understanding how task properties affect performance
- **AMOS** (500 CTs + MRIs, 15 organs) - if you want a larger dataset

Install nnU-Net, format the data, train one fold. This will teach you more about 3D medical segmentation than any tutorial.

### Then modify the prediction target

Once you have a baseline, the most impactful thing to change is **what you predict**:
- Are your classes semantically consistent across cases?
- Can you reformulate them as geometric properties?
- Can you decompose instance segmentation into semantic prediction + classical postprocessing?

### Only then consider architecture changes

After the prediction target is right, you might benefit from pretraining on a larger dataset, ensemble of multiple architectures, or test-time augmentation. But these are refinements. The representation is the foundation.

---

## References

1. Ronneberger, O. et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation," *MICCAI* 2015.
2. Isensee, F. et al., "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods* 2021. [GitHub](https://github.com/MIC-DKFZ/nnUNet)
3. Isensee, F. et al., "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation," [arXiv:2404.09556](https://arxiv.org/abs/2404.09556), 2024.
4. Wasserthal, J. et al., "TotalSegmentator: Robust Segmentation of 104 Anatomical Structures in CT Images," [arXiv:2208.05868](https://arxiv.org/abs/2208.05868), 2022.
5. Cheng, C.-T. et al., "PENGWIN Challenge 2024 Summary," [arXiv:2504.02382](https://arxiv.org/abs/2504.02382), 2025.
6. He, K. et al., "Deep Residual Learning for Image Recognition," *CVPR* 2016.
7. Maier-Hein, L. et al., "Metrics Reloaded: Recommendations for Image Analysis Validation," *Nature Methods* 2024.
8. Dice, L.R., "Measures of the Amount of Ecologic Association Between Species," *Ecology* 1945.
