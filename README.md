# BenchmarkingDocker: Model Server Benchmarking Suite

This project benchmarks the performance of a machine learning model server under different request strategies and deployment architectures.
![image](https://github.com/user-attachments/assets/1645e72c-6024-46ef-903d-49e2de5ee473)


## 📌 Project Overview

The goal is to analyze how different request-handling methods perform when sending concurrent or sequential HTTP requests to a model-serving endpoint. Additionally, it compares performance between a **standalone server** and a **load-balanced setup using NGINX**.

## 🏗️ Architecture

The system supports two deployment configurations:

### 1. Standalone Server

A single FastAPI server running on port `8000`.

### 2. Load-Balanced NGINX Server

Three FastAPI containers behind an NGINX reverse proxy that distributes load using the `least_conn` policy.
Running on port 80



![image](https://github.com/user-attachments/assets/442f6e11-a545-487d-85ff-9e236df6b2c1)








