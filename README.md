# BenchmarkingDocker: Model Server Benchmarking Suite

This project benchmarks the performance of a machine learning model server under different request strategies and deployment architectures.
![image](https://github.com/user-attachments/assets/2dc8e36a-ec77-4d46-a0bf-21411b860854)



## 📌 Project Overview

The goal is to analyze how different request-handling methods perform when sending concurrent or sequential HTTP requests to a model-serving endpoint. Additionally, it compares performance between a **standalone server** and a **load-balanced setup using NGINX**.

## 🏗️ Architecture

The system supports two deployment configurations:

### 1. Standalone Server

A single FastAPI server running on port `8000`.

### 2. Load-Balanced NGINX Server

Three FastAPI containers behind an NGINX reverse proxy that distributes load using the `least_conn` policy.
Running on port 80

![image](https://github.com/user-attachments/assets/b5ab108d-58dc-47db-8023-260255dc4ed5)











