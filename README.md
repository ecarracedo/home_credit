# Evaluación de Riesgo Crediticio con LightGBM

Este proyecto pensado como MVP implementa un sistema de evaluación de riesgo crediticio utilizando un modelo de machine learning (LightGBM) y una interfaz interactiva desarrollada con Streamlit.

## Estructura del Proyecto

```

├── lghtdm_model.py
├── main.py
├── requirements.txt
├── data/
│  │   └── data_train_new.zip
|
└── model/
    └── model_lightgbm.pkl
```

- **lghtdm_model.py**: Script para entrenamiento, ingeniería de características y guardado del modelo.
- **main.py**: Aplicación Streamlit para la evaluación interactiva del riesgo crediticio.
- **data/**: Contiene el datasets de entrenamiento.
- **model/**: Carpeta donde se almacena el modelo entrenado y artefactos.
- **EDA.ipynb**: Análisis exploratorio de datos.
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

## Uso de la Aplicación

Lanza la aplicación Streamlit para evaluar el riesgo crediticio de nuevos clientes:

```sh
streamlit run main.py
```

Sigue las instrucciones en la interfaz para ingresar los datos del cliente y obtener la predicción de riesgo.

## Notas

- Los archivos de datos no están incluidos en el repositorio por motivos de privacidad.
- Asegúrate de que los archivos de datos estén en la carpeta `data/` antes de entrenar el modelo.

## Autor

Desarrollado por [Tu