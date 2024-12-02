# Predictive Maintenance Pipeline for IoT Sensor Data

## Overview
This project implements a predictive maintenance pipeline using IoT sensor data from the CMAPSSData dataset. The goal is to predict the Remaining Useful Life (RUL) of engines based on sensor readings, leveraging techniques such as feature correlation analysis, GRU neural networks, and sequence modeling.

---

## Dataset
The dataset used is **CMAPSSData**, which contains sensor readings from jet engines. Each engine's life cycle is captured until a failure occurs. The main features include:
- **Operational Settings:** Three operating conditions.
- **Sensor Measurements:** Twenty-one sensor readings.
- **RUL:** Remaining Useful Life (calculated from the number of cycles).

**Location:**  
Dataset files are stored in the `data/CMAPSSDATA` directory.

---

## Installation
To set up the environment and dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PredictiveMaintenancePipeline.git
   cd PredictiveMaintenancePipeline
