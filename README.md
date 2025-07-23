# Evaluación de Riesgo Crediticio con LightGBM

Este proyecto pensado como MVP implementa un sistema de evaluación de riesgo crediticio utilizando un modelo de machine learning (LightGBM) y una interfaz interactiva desarrollada con Streamlit.

## Estructura del Proyecto

```

├── lghtdm_model.py
├── main.py
├── requirements.txt
├── data/
│     └── data_train_new.zip
|     └── HomeCredit_columns_description.csv
|
└── model/
    └── model_lightgbm.pkl
```

- **lghtdm_model.py**: Script para entrenamiento, ingeniería de características y guardado del modelo.
- **main.py**: Aplicación Streamlit para la evaluación interactiva del riesgo crediticio.
- **data/**: Contiene el datasets de entrenamiento.
- **model/**: Carpeta donde se almacena el modelo entrenado y artefactos.
- **requirements.txt**: Dependencias del proyecto.

## Instalación

1. Clona este repositorio y navega a la carpeta del proyecto.
2. Instala las dependencias:
   ```sh
   pip install -r requirements.txt
   ```

## Entrenamiento del Modelo

Descomprimir el archivo data_train_new.zip para extraer el archivo entrenamiento.

Ejecuta el script de entrenamiento para generar el modelo y los artefactos necesarios:

```sh
python lghtdm_model.py
```

Configurar dentro de `lghtdm_model.py` la ruta del archivo

ARCHIVO_DATOS = os.path.join(BASE_DIR, 'data', 'data_train_new.csv')

El modelo entrenado se guardará en `model/model_lightgbm.pkl`.

## lghtdm_model.py

Este script es responsable de todo el proceso de entrenamiento y preparación del modelo de riesgo crediticio. Sus principales funciones incluyen:

- **Carga y preprocesamiento de datos:** Lee los datos de entrenamiento, realiza limpieza, transformación y generación de nuevas variables relevantes para el modelo.
- **Ingeniería de características:** Selecciona y transforma variables numéricas y categóricas, calcula ratios financieros y prepara los datos para el modelo.
- **Entrenamiento del modelo:** Utiliza LightGBM (o un pipeline de scikit-learn) para entrenar el modelo de clasificación sobre los datos procesados.
- **Evaluación:** Calcula métricas de desempeño como ROC-AUC, accuracy, matriz de confusión, etc., para validar la calidad del modelo.
- **Serialización de artefactos:** Guarda el modelo entrenado, el preprocesador y la lista de columnas/variables necesarias en un archivo `.pkl` dentro de la carpeta `model/`. Esto permite que la aplicación principal (`main.py`) cargue y utilice el modelo sin necesidad de reentrenar.

Este archivo debe ejecutarse cada vez que se quiera actualizar el modelo con nuevos datos o cambios en

# HomeCredit_columns_description.csv

El archivo es una descripcion de las columnas del archivo `data_train.csv.` Puede agregar mas caracteristicas a la evaluacion del modelo, con ayuda de este archivo.


## Uso de la Aplicación

Lanza la aplicación Streamlit para evaluar el riesgo crediticio de nuevos clientes:

```sh
streamlit run main.py
```

Sigue las instrucciones en la interfaz para ingresar los datos del cliente y obtener la predicción de riesgo.

Tambien se encuentra alojada en Streamlit Cloud en la siguiente direccion:

https://homecredit-mvp.streamlit.app/

## Notas

- En el historial de credito hay tres componentes que son como puntaje externo de empresas de historial crediticio. Si sube los 3 valores vera que la evaluacion crediticia es positiva.
- Asegúrate de que los archivos de datos estén en la carpeta `data/` antes de entrenar el modelo.
- Este proyecto es personal pero tambien la idea que puedan ver la experiencia de entrenamiento de modelo para futuros proyectos que pueda participar.

## Autor

Desarrollado por Emiliano Carracedo | ecarracedo@gmail.com | 