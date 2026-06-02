# Detección de Intrusiones en IoT con Machine Learning y Deep Learning (CICIoT2023)

## 🧠 Descripción del proyecto
Este proyecto explora la aplicación de técnicas de **Machine Learning (ML) clásico y Deep Learning (DL)** para la detección de ciberataques en redes IoT utilizando el dataset **CICIoT2023**.

El objetivo es construir y evaluar un pipeline completo de clasificación de tráfico de red, analizando el rendimiento de distintos enfoques de modelado sobre datos reales y de gran escala.

En concreto, el proyecto aborda:
* **Clasificación binaria** de tráfico (benigno vs malicioso).
* **Clasificación multiclase** de familias de ataques.
* **Detección de subtipos** de ataques más específicos.
* **Análisis exploratorio de datos (EDA)**.
* Estudio del impacto del **desbalance de clases** y la **redundancia de características**.
* Comparación de rendimiento entre modelos de ML tradicional y arquitecturas de DL.

---

## 💾 El Dataset (CICIoT2023)
Este proyecto utiliza el dataset **CICIoT2023**, un conjunto de datos a gran escala para la detección de intrusiones en dispositivos IoT, creado por el *Canadian Institute for Cybersecurity*.

### 📌 Fuente de los datos
* **Dataset en Hugging Face:** [lacg030175/CIC-IoT-2023-full](https://huggingface.co/datasets/lacg030175/CIC-IoT-2023-full)
* **Página oficial del CIC:** [CIC-IoT-2023 Dataset](https://www.unb.ca/cic/datasets/iotdataset-2023.html)

---

## 📈 Características y Etiquetas
El dataset contiene características extraídas de logs de tráfico IoT real. Cada registro representa una conexión de red descrita mediante métricas estadísticas y de protocolo.

### 🔧 Tipos de variables
* **Estadísticas de paquetes:** Tasa, tamaño y tiempos de flujo.
* **Flags TCP/IP:** SYN, ACK, FIN, RST, etc.
* **Protocolos de red:** TCP, UDP, HTTP, DNS, etc.
* **Métricas de flujo:** Media, desviación estándar, varianza, mínimos y máximos.

### 🏷️ Niveles de Granularidad (Etiquetas)
El dataset permite abordar el problema desde tres niveles de especificidad:
1. `label`: Clasificación **binaria** (Benigno / Malicioso).
2. `attack_class`: Clasificación por **familia de ataque** (DoS, DDoS, Mirai, Recon, etc.).
3. `Label`: Clasificación detallada por **subtipo específico de ataque**.

---

## 📁 Estructura del proyecto

El repositorio sigue una estructura organizada para separar datos, código, experimentos y documentación:

```text
datasets/
│
├── raw/              # Datos originales (Parquet descargados)
└── processed/        # Datos limpios y preparados para modelado

src/                  # Código fuente (pipelines, entrenamiento, utilidades)

notebooks/            # Jupyter notebooks para exploración y experimentación

models/               # Modelos entrenados y checkpoints guardados

results/
│
├── metrics/          # JSON / CSV con métricas (accuracy, F1, etc.)
└── plots/            # Visualizaciones (matrices de confusión, ROC, etc.)

docs/                 # Documentación del proyecto (memoria, presentaciones, informes)