# Adversarial-Defense-Strategies-for-Object-Detection-In-Autonomous-Vehicle-Perception-Systems

# 🚗🔒 Adversarial Defense Strategies for Object Detection  
Robustifying YOLO v8 for Autonomous‑Vehicle Perception
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](/LICENSE)

> **Authors**: Manas Dixit · Nitish Poojari · Sharva Khandagale  
> Minnesota Robotics Institute, University of Minnesota Twin Cities

---

## ✨ Project Highlights
| ✔ | Description |
|---|-------------|
| • | Implements **FGSM**, **PGD** and (**WIP**) **JSMA** attacks on the KITTI dataset. |
| • | Benchmarks both **YOLO v8** and a **ViT‑based detector** under adversarial stress. |
| • | Measures robustness with **Prediction Change Rate** & **Mean Confidence Drop** (mAP coming). |
| • | Adds two defenses:<br>  1. **Ensemble Adversarial Training**<br>  2. **Self‑Supervised Anomaly Detector** that catches, logs, and retrains on new adversarial inputs. |
| • | Achieved 14 % ↓ confidence & 89 % PCR on YOLO v8 with PGD/FGSM (baseline numbers to beat). |

---

## 📑 Table of Contents
1. [Background](#background)
2. [Repo Structure](#repo-structure)
3. [Setup & Installation](#setup--installation)
4. [Datasets](#datasets)
5. [Running Attacks](#running-attacks)
6. [Defenses & Training](#defenses--training)
7. [Results](#results)
8. [Roadmap](#roadmap)
9. [Citation](#citation)
10. [License](#license)

---

## Background
CNN‑based detectors like **YOLO** excel at real‑time perception but are **vulnerable to imperceptible perturbations**.  
Our goal is to **quantify** that vulnerability and **close the gap** with lightweight, deployable defenses suitable for on‑board embedded hardware.

Key references:  
* Goodfellow *et al.* (ICLR 2015) – FGSM  
* Carlini & Wagner (S&P 2017) – evaluating robustness  
* Tramèr *et al.* (ICLR 2018) – Ensemble Adversarial Training  
* Xie *et al.* (ICCV 2017) – DAG for detection/segmentation

---

*WORK IN PROGRESS ...*
