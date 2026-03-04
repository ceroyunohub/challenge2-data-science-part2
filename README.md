## Análisis y Predicción de Abandono de Clientes (Churn Prediction)

## Descripción del Proyecto

Este proyecto tiene como objetivo identificar los factores clave que influyen en el abandono de clientes (Churn) en una empresa de telecomunicaciones y desarrollar modelos de Machine Learning capaces de predecir la probabilidad de que un cliente cancele su servicio. La predicción temprana del churn permite implementar estrategias de retención proactivas, reduciendo la pérdida de clientes y optimizando los recursos de la empresa.

## Origen de los Datos

Los datos utilizados provienen de un conjunto de datos público de una empresa de telecomunicaciones, el cual contiene información demográfica de los clientes, servicios contratados, información de la cuenta y si el cliente ha cancelado el servicio o no.

## Pasos del Análisis y Modelado

### 1. Preparación de Datos

*   **Extracción del Archivo Tratado:** Se cargó el archivo CSV `/datos_tratados.csv` en un DataFrame de Pandas.
*   **Eliminación de Columnas Irrelevantes:** Se eliminó la columna `customerID` por ser un identificador único sin valor predictivo.
*   **Codificación de Variables Categóricas:** Se aplicó `one-hot encoding` (`pd.get_dummies`) a todas las columnas de tipo 'object' para convertir las variables categóricas en un formato numérico compatible con los algoritmos de Machine Learning.
*   **Evaluación del Balance de Clases:** Se analizó la distribución de la variable objetivo `Churn_Yes` y se identificó un desbalance significativo hacia la clase de no cancelación.
*   **Balanceo de Clases con SMOTE:** Para corregir el desbalance, se aplicó la técnica SMOTE (Synthetic Minority Over-sampling Technique) para generar ejemplos sintéticos de la clase minoritaria (`Churn_Yes = True`), logrando una distribución balanceada (50/50).
*   **Estandarización de Características Numéricas:** Se utilizó `StandardScaler` para estandarizar las columnas numéricas en el DataFrame balanceado (`df_balanced`), asegurando que tuvieran una media de 0 y una desviación estándar de 1. Esto es crucial para modelos sensibles a la escala como la Regresión Logística.

### 2. Análisis Exploratorio de Datos (EDA)

*   **Análisis de Correlación:** Se calculó y visualizó la matriz de correlación de las variables estandarizadas, prestando especial atención a la correlación con `Churn_Yes`. Se identificaron factores como el método de pago (cheque electrónico), servicio de fibra óptica, facturación sin papel, cargo mensual y tiempo de contrato como los más correlacionados.
*   **Análisis Dirigido (Boxplots):** Se realizaron boxplots para visualizar la relación entre `customer_tenure` (tiempo de contrato) y `account_Charges.Total` (gasto total) con `Churn_Yes`, confirmando que una menor tenencia y menores gastos totales están asociados con una mayor probabilidad de cancelación.

### 3. Modelado Predictivo

Se dividió el conjunto de datos estandarizado en conjuntos de entrenamiento y prueba (80/20) y se entrenaron dos modelos de clasificación:

#### Modelo 1: Regresión Logística

*   **Descripción:** Modelo lineal sensible a la escala de las características. Se entrenó con los datos estandarizados.
*   **Métricas de Rendimiento:**
    *   **Exactitud (Accuracy):** 0.8193
    *   **Reporte de Clasificación:**
        *   **No Churn (False):** Precisión: 0.83, Recall: 0.81, F1-score: 0.82
        *   **Churn (True):** Precisión: 0.81, Recall: 0.83, F1-score: 0.82

#### Modelo 2: Random Forest Classifier

*   **Descripción:** Modelo de conjunto basado en árboles de decisión, menos sensible a la escala de los datos. Se entrenó con los datos estandarizados.
*   **Métricas de Rendimiento:**
    *   **Exactitud (Accuracy):** 0.8478
    *   **Reporte de Clasificación:**
        *   **No Churn (False):** Precisión: 0.85, Recall: 0.84, F1-score: 0.85
        *   **Churn (True):** Precisión: 0.84, Recall: 0.85, F1-score: 0.85

### 4. Comparación y Selección del Mejor Modelo

El **Random Forest Classifier** demostró un rendimiento superior en todas las métricas clave (exactitud, precisión, recall, F1-score) en comparación con la Regresión Logística. Su capacidad para capturar relaciones no lineales y su naturaleza de conjunto lo hacen más robusto y preciso para este problema. Por lo tanto, el **Random Forest Classifier fue seleccionado como el mejor modelo** para la predicción de abandono de clientes.

### 5. Análisis de la Relevancia de Variables

Ambos modelos concuerdan en que los factores más influyentes en la cancelación son:

*   **Antigüedad del cliente (`customer_tenure`):** A menor tiempo con la empresa, mayor probabilidad de churn.
*   **Cargos mensuales y totales (`account_Charges.Monthly`, `account_Charges.Total`):** Altos cargos mensuales y bajos cargos totales (asociados a baja tenencia) incrementan el riesgo.
*   **Servicio de Internet (`internet_InternetService_Fiber optic`):** Los clientes con fibra óptica tienen una mayor propensión a cancelar.
*   **Método de Pago (`account_PaymentMethod_Electronic check`):** El uso del cheque electrónico se asocia con un mayor riesgo de churn.
*   **Tipo de Contrato (`account_Contract_Two year`):** Los contratos a dos años reducen significativamente la probabilidad de abandono.
*   **Facturación sin papel (`account_PaperlessBilling_Yes`):** Los clientes con esta opción muestran mayor riesgo de churn.

## Estrategias de Retención de Clientes

Basándose en los factores clave de abandono identificados, se proponen las siguientes estrategias:

*   **Mejorar la Calidad del Servicio de Fibra Óptica:** Investigar y abordar las causas de insatisfacción entre los usuarios de fibra óptica (calidad, precio, soporte).
*   **Revisión de Tarifas y Ofertas:** Evaluar la estructura de precios, ofreciendo planes más competitivos y descuentos a clientes con alto riesgo de abandono o aquellos con altos cargos mensuales.
*   **Optimizar Métodos de Pago:** Mejorar la experiencia o incentivar el uso de métodos de pago más convenientes y de mayor retención, alejándose del cheque electrónico.
*   **Programas de Lealtad para Clientes Nuevos:** Implementar programas de bienvenida, ofertas exclusivas y atención proactiva durante los primeros meses para aumentar la retención inicial.
*   **Incentivos para Contratos a Largo Plazo:** Fortalecer las ofertas de contratos a uno o dos años con beneficios adicionales para fomentar la lealtad y reducir la flexibilidad de cancelación.
*   **Mejorar la Experiencia de Facturación Digital:** Optimizar la usabilidad y claridad del portal de facturación sin papel para evitar frustraciones que puedan llevar al churn.
*   **Personalizar Ofertas de Contenido:** Ofrecer recomendaciones y paquetes personalizados de streaming para aumentar la satisfacción y el apego a los servicios adicionales.

## Tecnologías Utilizadas

*   **Python:** Lenguaje de programación principal.
*   **Pandas:** Manipulación y análisis de datos.
*   **NumPy:** Operaciones numéricas.
*   **Scikit-learn:** Implementación de modelos de Machine Learning (LogisticRegression, RandomForestClassifier, StandardScaler, train_test_split).
*   **Imbalanced-learn (imblearn):** Manejo de desbalance de clases (SMOTE).
*   **Matplotlib y Seaborn:** Visualización de datos.

## Cómo Ejecutar el Proyecto

1.  Clonar el repositorio:
    `git clone [URL_DEL_REPOSITORIO]`
    `cd [NOMBRE_DEL_REPOSITORIO]`
2.  Instalar las dependencias:
    `pip install -r requirements.txt` (o instalar las librerías mencionadas manualmente)
3.  Abrir el notebook de Jupyter o Google Colab:
    `jupyter notebook nombre_del_notebook.ipynb`
4.  Ejecutar las celdas en orden para reproducir el análisis.
