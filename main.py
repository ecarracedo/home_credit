import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

# Configuración de la página
st.set_page_config(
    page_title="Evaluación de Riesgo Crediticio (LightGBM)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título de la aplicación
st.title("🏦 Sistema de Evaluación de Riesgo Crediticio")
st.markdown("""
Esta aplicación utiliza un modelo LightGBM para predecir la probabilidad de incumplimiento crediticio.
""")

# Estilos CSS personalizados
st.markdown("""
<style>
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .st-b7 {
        color: white;
    }
    .st-c0 {
        background-color: #0E1117;
    }
    .css-1aumxhk {
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Cargar modelo y artefactos
@st.cache_resource
def cargar_modelo():
    try:
        artefactos = joblib.load('model/model_lightgbm.pkl')
        return artefactos
    except Exception as e:
        st.error(f"Error al cargar el modelo: {str(e)}")
        return None

artefactos = cargar_modelo()

if artefactos is None:
    st.stop()

# Función para formatear nombres de características
def formatear_nombre(nombre):
    return nombre.replace('_', ' ').replace('AMT', 'Monto').replace('CNT', 'Cant.').title()

# Función para generar campos de entrada
def generar_campos_entrada(artefactos):
    input_data = {}
    
    # Organizar características por categorías
    categorias = {
        "📌 Información Demográfica": [
            'DAYS_BIRTH', 'CODE_GENDER', 'CNT_CHILDREN', 
            'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE'
        ],
        "💵 Información Financiera": [
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
            'AMT_GOODS_PRICE', 'DEBT_TO_INCOME_RATIO'
        ],
        "🏡 Propiedades y Activos": [
            'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'OWN_CAR_AGE',
            'CNT_FAM_MEMBERS'
        ],
        "💼 Situación Laboral": [
            'DAYS_EMPLOYED', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
            'NAME_INCOME_TYPE'
        ],
        "📊 Historial Crediticio": [
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'DAYS_LAST_PHONE_CHANGE'
        ]
    }
    
    with st.sidebar.form("formulario_cliente"):
        st.header("📋 Datos del Cliente")
        
        for categoria, features in categorias.items():
            with st.expander(categoria):
                for feature in features:
                    if feature in artefactos['columnas_entrenamiento']:
                        # Campos numéricos
                        if feature in artefactos['numeric_features']:
                            if 'DAYS_' in feature:
                                label = formatear_nombre(feature.replace('DAYS_', '')) + " (años)"
                                years = st.number_input(
                                    label,
                                    min_value=0,
                                    max_value=100 if 'BIRTH' in feature else 50,
                                    value=30 if 'BIRTH' in feature else 5
                                )
                                input_data[feature] = -years * 365
                            elif feature == 'AMT_INCOME_TOTAL':
                                input_data[feature] = st.number_input(
                                    formatear_nombre(feature),
                                    min_value=0,
                                    value=180000,
                                    step=10000
                                )
                            elif feature == 'AMT_CREDIT':
                                input_data[feature] = st.number_input(
                                    formatear_nombre(feature),
                                    min_value=0,
                                    value=500000,
                                    step=10000
                                )
                            else:
                                input_data[feature] = st.number_input(
                                    formatear_nombre(feature),
                                    value=0
                                )
                        # Campos categóricos
                        elif feature in artefactos['categorical_features']:
                            if feature == 'CODE_GENDER':
                                input_data[feature] = st.selectbox(
                                    formatear_nombre(feature),
                                    ['M', 'F']
                                )
                            elif feature in ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
                                input_data[feature] = st.selectbox(
                                    formatear_nombre(feature),
                                    ['Y', 'N']
                                )
                            elif feature == 'OCCUPATION_TYPE':
                                ocupaciones = [
                                    'Laborers', 'Core staff', 'Accountants', 'Cleaning staff',
                                    'Managers', 'Medicine staff', 'High skill tech staff', 'Drivers',
                                    'Security staff', 'Sales staff', 'Cooking staff', 'Realty agents',
                                    'Secretaries', 'Waiters/barmen staff', 'IT staff',
                                    'Low-skill Laborers', 'Private service staff', 'HR staff'
                                ]
                                input_data[feature] = st.selectbox(
                                    formatear_nombre(feature),
                                    ocupaciones
                                )
                            elif feature == 'ORGANIZATION_TYPE':
                                occupation_options = [
                                    'School', 'Kindergarten', 'University',
                                    'Medicine', 'Emergency',
                                    'Bank', 'Insurance', 'Legal Services',
                                    'Trade', 'Realtor', 'Transport',
                                    'Construction', 'Housing',
                                    'Telecom', 'Mobile', 'Electricity',
                                    'Advertising', 'Culture',
                                    'Cleaning', 'Agriculture',
                                    'Industry', 'Services', 'Self-employed',
                                    'Restaurant', 'Hotel', 'Other', 'Business'
                                ]
                                input_data[feature] = st.selectbox(
                                    formatear_nombre(feature),
                                    sorted(occupation_options)  # Ordenado alfabéticamente (opcional)
                                )
                            elif feature == 'NAME_INCOME_TYPE':
                                income_type_options = [
                                    'Pensioner',
                                    'Working',
                                    'Commercial associate',
                                    'State servant',
                                    'Unemployed',
                                    'Student',
                                    'Maternity leave',
                                    'Businessman'
                                ]
                                input_data[feature] = st.selectbox(
                                    formatear_nombre(feature),
                                    income_type_options
                                )

                            elif feature == 'NAME_FAMILY_STATUS':
                                family_status_options = [
                                    'Married','Single',
                                    'Civil marriage', 'Separated', 'Widow','Unknown'
                                ]
                                
                                input_data[feature] = st.selectbox(
                                    formatear_nombre(feature),
                                    family_status_options
                                )

                            elif feature == 'NAME_HOUSING_TYPE':
                                housing_options = [
                                    'House / apartment',
                                    'Municipal apartment',
                                    'With parents',
                                    'Rented apartment',
                                    'Co-op apartment',
                                    'Office apartment'
                                ]
                                
                                input_data[feature] = st.selectbox(
                                    formatear_nombre(feature),  # Función para formato bonito
                                    housing_options
                                )

                            else:
                                input_data[feature] = st.text_input(
                                    formatear_nombre(feature),
                                    value=""
                                )
        
        submitted = st.form_submit_button("🔍 Evaluar Riesgo Crediticio")
    
    return input_data, submitted
    
# Función para preparar datos para predicción
def preparar_datos_para_prediccion(input_data, artefactos):

    try:
        # Crear DataFrame con las columnas exactas del entrenamiento
        input_df = pd.DataFrame(columns=artefactos['columnas_entrenamiento'])
        
        # Poblar los datos proporcionados
        for col in input_data:
            if col in input_df.columns:
                input_df[col] = [input_data[col]]
        
        # Rellenar valores faltantes con valores por defecto
        for col in input_df.columns:
            if col not in input_data:
                if col in artefactos['numeric_features']:
                    input_df[col] = 0
                elif col in artefactos['categorical_features']:
                    input_df[col] = 'missing'
        
        return input_df
    except Exception as e:
        st.error(f"Error preparando datos: {str(e)}")
        return None

# Función para hacer predicción
def hacer_prediccion(input_df, artefactos):
    try:
        modelo = artefactos['modelo']  # Ya incluye el preprocessor
        proba = modelo.predict_proba(input_df)[0][1]
        return proba
    except Exception as e:
        st.error(f"Error en la predicción: {str(e)}")
        return None

# Función para mostrar resultados
def mostrar_resultados(probabilidad):
    st.success("✅ Evaluación completada")
    
    # Mostrar probabilidad
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.metric(
            label="**Probabilidad de Incumplimiento**",
            value=f"{probabilidad:.2%}",
            help="Probabilidad estimada de que el cliente tenga dificultades para pagar el préstamo"
        )
    
    # Barra de riesgo
    riesgo = probabilidad * 100
    st.progress(int(riesgo))
    st.caption(f"Nivel de riesgo: {riesgo:.1f}%")

    # Recomendación
    st.subheader("📌 Recomendación")
    if riesgo < 25:
        st.success("""
        **✅ Cliente de bajo riesgo**  
        Préstamo recomendado con condiciones estándar.
        """)
    elif riesgo < 50:
        st.warning("""
        **⚠️ Cliente de riesgo moderado**  
        Considerar con precaución. Se recomienda:
        - Tasa de interés más alta
        - Monto menor al solicitado
        - Garantías adicionales
        """)
    else:
        st.error("""
        **❌ Cliente de alto riesgo**  
        Préstamo no recomendado. Alternativas:
        - Solicitar garantías colaterales
        - Reducir monto significativamente
        - Rechazar la solicitud
        """)
    
    # Explicación técnica
    with st.expander("📊 Detalles técnicos"):
        st.write(f"""
        **Modelo utilizado:** LightGBM  
        **Características consideradas:** {len(artefactos['columnas_entrenamiento'])}  
        **Probabilidad de incumplimiento:** {probabilidad:.4f}
        """)
        
        st.write("""
        **Interpretación del resultado:**  
        - **<25%:** Bajo riesgo - Cliente con alta probabilidad de pago oportuno  
        - **25-50%:** Riesgo moderado - Cliente con posibles dificultades de pago  
        - **>50%:** Alto riesgo - Cliente con alta probabilidad de incumplimiento  
        """)

# Interfaz principal
def main():
    # Obtener datos del formulario
    input_data, submitted = generar_campos_entrada(artefactos)
    
    # Procesar cuando se envía el formulario
    if submitted:
        with st.spinner('Analizando perfil crediticio...'):
            # Preparar datos
            input_df = preparar_datos_para_prediccion(input_data, artefactos)
            
            if input_df is not None:
                # Hacer predicción
                probabilidad = hacer_prediccion(input_df, artefactos)
                
                if probabilidad is not None:
                    # Mostrar resultados
                    mostrar_resultados(probabilidad)
                else:
                    st.error("No se pudo realizar la predicción")
            else:
                st.error("Error en la preparación de datos")

# Ejecutar la aplicación
if __name__ == "__main__":
    main()