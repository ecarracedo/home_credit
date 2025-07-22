# entrenamiento_lightgbm.py
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm

def cargar_datos(archivo):
    """Carga y prepara los datos iniciales con verificación de ruta"""
    print(f"Intentando cargar datos desde: {archivo}")
    
    # Verificar si el archivo existe
    if not os.path.exists(archivo):
        raise FileNotFoundError(f"El archivo {archivo} no existe en la ruta especificada")
    
    try:
        data = pd.read_csv(archivo)
        print("Columnas disponibles:", data.columns.tolist())
        
        # Verificar columnas requeridas
        required_cols = ['SK_ID_CURR', 'TARGET']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas: {missing_cols}")
        
        X = data.drop(required_cols, axis=1)
        y = data['TARGET']
        
        print(f"Datos cargados correctamente. Dimensiones: {X.shape}")
        return X, y
    except Exception as e:
        print(f"Error al cargar datos: {str(e)}")
        return None, None

def crear_features(df):
    """Ingeniería de características adicionales con manejo de errores"""
    print("Creando características adicionales...")
    try:
        df = df.copy()
        
        # Ratios financieros con manejo de división por cero
        df['DEBT_TO_INCOME_RATIO'] = np.where(
            df['AMT_INCOME_TOTAL'] > 0,
            df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'],
            np.nan
        )
        
        df['ANNUITY_TO_INCOME_RATIO'] = np.where(
            df['AMT_INCOME_TOTAL'] > 0,
            df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'],
            np.nan
        )
        
        # Conversión de días a años
        df['AGE_YEARS'] = -df['DAYS_BIRTH'] / 365.25
        df['WORKING_YEARS'] = -df['DAYS_EMPLOYED'].clip(lower=0) / 365.25
        
        # Documentos proporcionados
        doc_cols = [col for col in df.columns if 'FLAG_DOCUMENT_' in col]
        if doc_cols:
            df['DOCS_PROVIDED_RATIO'] = df[doc_cols].sum(axis=1) / len(doc_cols)
        
        print("Características adicionales creadas con éxito")
        return df
    except Exception as e:
        print(f"Error al crear características: {str(e)}")
        return df  # Devuelve el dataframe original si hay error



    """Prepara el pipeline de preprocesamiento con validación"""
    print("Preparando pipeline...")
    print(f"Columnas numéricas: {numeric_features}")

def preparar_pipeline(numeric_features, categorical_features):
    """Prepara el pipeline de preprocesamiento con validación"""
    print("Preparando pipeline...")
    print(f"Columnas numéricas: {numeric_features}")
    print(f"Columnas categóricas: {categorical_features}")
    
    try:
        transformers = []
        
        if numeric_features:
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])
            transformers.append(('num', numeric_transformer, numeric_features))
        
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])  # sparse=True para eficiencia
            transformers.append(('cat', categorical_transformer, categorical_features))
        
        preprocessor = ColumnTransformer(transformers=transformers)
        return preprocessor
    except Exception as e:
        print(f"Error al preparar pipeline: {str(e)}")
        raise

def entrenar_modelo(X_train, y_train, preprocessor):
    # Primero ajustar y transformar TODOS los datos de entrenamiento
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Dividir los datos ya transformados para validación
    X_train_trans, X_val_trans, y_train, y_val = train_test_split(
        X_train_transformed, y_train, test_size=0.1, random_state=42, stratify=y_train)
    
    # Configuración del modelo LightGBM
    count_0 = (y_train == 0).sum()
    count_1 = (y_train == 1).sum()
    scale_pos_weight = count_0 / count_1
    
    lgbm_params = {
        'objective': 'binary',
        'scale_pos_weight': scale_pos_weight,
        'boosting_type': 'gbdt',
        'n_estimators': 500,
        'learning_rate': 0.02,
        'max_depth': -1,
        'num_leaves': 31,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'metric': 'auc',
        'early_stopping_rounds': 50,
        'verbose': -1
    }
    
    # Entrenar directamente con los datos transformados
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(
        X_train_trans, y_train,
        eval_set=[(X_val_trans, y_val)],
        eval_metric='auc'
    )
    
    # Crear un pipeline que incluya el preprocesador y el modelo
    lgbm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return lgbm_pipeline

def guardar_modelo(modelo, preprocessor, X_train, nombre_archivo='modelo_lightgbm.pkl'):
    """Guarda el modelo y los artefactos necesarios con validación"""
    print(f"Intentando guardar modelo en {nombre_archivo}")
    
    try:
        # Verificaciones exhaustivas
        if not isinstance(modelo, Pipeline):
            raise ValueError("El modelo debe ser un Pipeline de sklearn")
            
        if not hasattr(preprocessor, 'transformers'):
            raise ValueError("Preprocesador inválido")
            
        if not isinstance(X_train, pd.DataFrame):
            raise ValueError("X_train debe ser un DataFrame")
        
        # Obtener información de las columnas
        numeric_cols = X_train.select_dtypes(include=['int', 'float']).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Crear diccionario de artefactos
        artefactos = {
            'modelo': modelo,
            'preprocessor': preprocessor,
            'columnas_entrenamiento': X_train.columns.tolist(),
            'numeric_features': numeric_cols,
            'categorical_features': categorical_cols,
            'fecha_entrenamiento': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'versiones': {
                'pandas': pd.__version__,
                'lightgbm': lgb.__version__,
                'sklearn': joblib.__version__
            }
        }
        
        # Guardar con compresión
        joblib.dump(artefactos, nombre_archivo, compress=3)
        print(f"Modelo guardado exitosamente en {nombre_archivo}")
        return True
    except Exception as e:
        print(f"Error al guardar el modelo: {str(e)}")
        return False

def optimizar_hiperparametros(X_train, y_train, preprocessor):
    # Definir espacio de búsqueda
    param_dist = {
        'classifier__n_estimators': [300, 500, 700],
        'classifier__learning_rate': [0.01, 0.02, 0.05],
        'classifier__max_depth': [4, 6, 8],
        'classifier__num_leaves': [20, 31, 50],
        'classifier__min_child_samples': [50, 100, 200],
        'classifier__reg_alpha': [0.1, 0.5, 1],
        'classifier__reg_lambda': [0.1, 0.5, 1],
        'classifier__scale_pos_weight': [None, 'balanced', 10, 20]
    }
    
    # Pipeline base
    lgbm_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', lgb.LGBMClassifier(objective='binary', random_state=42))
    ])
    
    # Búsqueda aleatoria
    search = RandomizedSearchCV(
        lgbm_pipeline,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    print("Mejores parámetros:", search.best_params_)
    print("Mejor F1-score:", search.best_score_)
    
    return search.best_estimator_
def main():
    try:
        # Configuración con rutas absolutas recomendadas
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ARCHIVO_DATOS = os.path.join(BASE_DIR, 'data', 'data_train_new.csv')
        MODELO_SALIDA = os.path.join(BASE_DIR, 'model', 'model_lightgbm.pkl')
        
        print("\n" + "="*50)
        print("Iniciando proceso de entrenamiento")
        print("="*50 + "\n")
        
        # Paso 1: Cargar datos
        X, y = cargar_datos(ARCHIVO_DATOS)
        if X is None or y is None:
            raise ValueError("No se pudieron cargar los datos")
        
        # Paso 2: Ingeniería de características
        X = crear_features(X)
        
        # Paso 3: Identificar tipos de características
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Verificar que todas las columnas están consideradas
        all_columns = set(numeric_features + categorical_features)
        missing_cols = set(X.columns) - all_columns
        if missing_cols:
            print(f"\nAdvertencia: Columnas no consideradas: {missing_cols}")
        
        # Paso 4: Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"\nDatos divididos - Entrenamiento: {X_train.shape}, Prueba: {X_test.shape}")
        
        # Paso 5: Preparar pipeline
        preprocessor = preparar_pipeline(numeric_features, categorical_features)
        
        # Validación adicional del preprocesador (AJUSTADO PRIMERO)
        try:
            print("\nValidando transformación de categorías...")
            # Primero ajustar el preprocesador
            preprocessor.fit(X_train)
            
            # Ahora podemos acceder a los transformadores
            if 'cat' in preprocessor.named_transformers_:
                sample = X_train[categorical_features].head() if categorical_features else pd.DataFrame()
                if not sample.empty:
                    cat_transformer = preprocessor.named_transformers_['cat']
                    transformed = cat_transformer.transform(sample)
                    print(f"Transformación exitosa. Forma: {transformed.shape}")
                else:
                    print("No hay columnas categóricas para transformar")
            else:
                print("No se encontró transformador categórico")
        except Exception as e:
            print(f"Error transformando categorías: {str(e)}")
            raise
        
        # Paso 6: Entrenar modelo
        modelo = entrenar_modelo(X_train, y_train, preprocessor)
        
        
        # Paso 7: Evaluar modelo
        print("\nEvaluando modelo...")
        y_pred = modelo.predict(X_test)
        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        
        print("\nReporte de Clasificación:")
        print(classification_report(y_test, y_pred))
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        print(f"\nAUC en conjunto de prueba: {auc_score:.4f}")
        
        # Paso 8: Guardar modelo
        if not guardar_modelo(modelo, preprocessor, X_train, MODELO_SALIDA):
            raise RuntimeError("Falló el guardado del modelo")
        
        print("\n" + "="*50)
        print("Proceso de entrenamiento completado exitosamente!")
        print("="*50 + "\n")
        
    except Exception as e:
        print("\n" + "="*50)
        print(f"ERROR: {str(e)}")
        print("="*50 + "\n")
        raise

if __name__ == "__main__":
    main()