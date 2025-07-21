import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página
st.set_page_config(page_title="Home Credit Default Risk", layout="wide")
st.title("Análisis de Riesgo de Incumplimiento de Crédito")

# Carga de datos (simplificada - en un caso real deberías cargar los datos reales)
@st.cache_data
def load_data():
    # Simulando la carga de datos (reemplazar con tus datos reales)
    data = {
        'SK_ID_CURR': np.random.randint(100000, 999999, 1000),
        'TARGET': np.random.choice([0, 1], 1000, p=[0.92, 0.08]),
        'AMT_INCOME_TOTAL': np.random.normal(150000, 50000, 1000).clip(30000, 300000),
        'AMT_CREDIT': np.random.normal(500000, 200000, 1000).clip(100000, 1000000),
        'NAME_CONTRACT_TYPE': np.random.choice(['Cash loans', 'Revolving loans'], 1000, p=[0.9, 0.1]),
        'CODE_GENDER': np.random.choice(['M', 'F', 'XNA'], 1000, p=[0.5, 0.5, 0.0]),
        'DAYS_BIRTH': -np.random.randint(365*20, 365*70, 1000),
        'DAYS_EMPLOYED': -np.random.randint(0, 365*50, 1000)
    }
    return pd.DataFrame(data)

df = load_data()

# Sidebar con controles
st.sidebar.header("Opciones de Visualización")
show_raw_data = st.sidebar.checkbox("Mostrar datos crudos")
target_analysis = st.sidebar.checkbox("Análisis de la variable objetivo", True)
feature_distribution = st.sidebar.checkbox("Distribución de características", True)
credit_analysis = st.sidebar.checkbox("Análisis de crédito", True)

# Mostrar datos crudos si se selecciona
if show_raw_data:
    st.subheader("Datos Crudos")
    st.write(df.head())

# Análisis de la variable objetivo
if target_analysis:
    st.subheader("Distribución de la Variable Objetivo (TARGET)")
    
    # Calcular distribución
    target_dist = df['TARGET'].value_counts(normalize=True)
    
    # Crear dos columnas para los gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Distribución porcentual:")
        st.write(target_dist)
    
    with col2:
        fig, ax = plt.subplots()
        target_dist.plot(kind='bar', ax=ax)
        ax.set_title('Distribución de TARGET')
        ax.set_xlabel('Target')
        ax.set_ylabel('Proporción')
        ax.set_xticklabels(['No Default (0)', 'Default (1)'], rotation=0)
        st.pyplot(fig)

# Distribución de características
if feature_distribution:
    st.subheader("Distribución de Características")
    
    # Seleccionar características para visualizar
    features = st.multiselect(
        "Selecciona características para visualizar",
        options=['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'DAYS_BIRTH', 'DAYS_EMPLOYED'],
        default=['AMT_INCOME_TOTAL', 'AMT_CREDIT']
    )
    
    for feature in features:
        st.write(f"### {feature}")
        
        if df[feature].dtype == 'object':
            # Gráfico de barras para variables categóricas
            fig, ax = plt.subplots()
            df[feature].value_counts().plot(kind='bar', ax=ax)
            ax.set_title(f'Distribución de {feature}')
            st.pyplot(fig)
        else:
            # Histograma para variables numéricas
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f'Distribución de {feature}')
            st.pyplot(fig)

# Análisis de crédito
if credit_analysis:
    st.subheader("Análisis de Crédito")
    
    # Relación entre ingresos y monto del crédito
    st.write("### Relación entre Ingresos y Monto del Crédito")
    
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='AMT_INCOME_TOTAL', y='AMT_CREDIT', hue='TARGET', alpha=0.6, ax=ax)
    ax.set_title('Ingresos vs Monto del Crédito (coloreado por Target)')
    st.pyplot(fig)
    
    # Análisis por tipo de contrato
    st.write("### Análisis por Tipo de Contrato")
    
    contract_analysis = df.groupby('NAME_CONTRACT_TYPE')['TARGET'].mean()
    fig, ax = plt.subplots()
    contract_analysis.plot(kind='bar', ax=ax)
    ax.set_title('Tasa de Incumplimiento por Tipo de Contrato')
    ax.set_ylabel('Tasa de Incumplimiento')
    st.pyplot(fig)

# Información adicional
st.sidebar.markdown("---")
st.sidebar.info("""
Esta es una aplicación simplificada basada en el proyecto AnyoneAI - Sprint Project 02.
Muestra análisis exploratorios básicos del conjunto de datos Home Credit Default Risk.
""")