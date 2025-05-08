# Creamos el archivo de la app en el interprete principal
import pandas as pd 
import plotly.express as px
import streamlit as st
import numpy as np
import matplotlib.pyplot as mt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")


# Aplicamos estilos con CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');

        html, body {
            font-family: 'Montserrat', sans-serif;
            color: #2E4053;
            background-color: #AFDDFF;
        .stApp {
            font-family: 'Arial', sans-serif;
            background-color: #FFF;
        }
        .stTitle {
            color: #1F618D;
        }
        .stHeader {
            color: #FFF;
        }
        .stSidebar {
            background-color: #67AE6E;
        }

        div[data-baseweb="select"] > div {
            background-color: #B5FCCD;
        }

    </style>
    """,
    unsafe_allow_html=True
)

# Definimos la instancia
@st.cache_resource

# Definimos la función de carga de datos
def load_data():
    df = pd.read_csv("Datos_limpios_Berlin.csv", index_col="last_scraped")
    df = df.drop(["Unnamed: 0"], axis=1)

     # Selecciono las columnas tipo numéricas del dataframe
    numeric_df = df.select_dtypes(['float','int']) # Devuelve las columnas
    numeric_cols = numeric_df.columns # Devuekve lista de columnas

    # Selecciono las columnas tipo texto
    text_df = df.select_dtypes(['object']) # Seleccionamos los datos tipo cadena (objeto)
    text_cols = text_df.columns # Mostramos las columnas

    # Selecciono algunas columnas categoricas de valores
    categorical_column_room = df['room_type']

    # Obtengo los valores unicos
    unique_categories_room = categorical_column_room.unique()

    # Codificar variables categóricas (One-Hot Encoding)
    #df_encoded = pd.get_dummies(df, drop_first=False)

    return df, numeric_cols, text_cols, unique_categories_room, numeric_df

    # return df, cols, cols_columns

# Cargo los datos obtenidos de la funcion "load_data()"
df, numeric_cols, text_cols, unique_categories_room, numeric_df = load_data()

#Creacion del dashboard

# Generamos las paginas para utilizar dentro del diseño
# Widget 1: Selectbox
# Menú desplegable de opciones de las páginas seleccionadas
# Generamos los encabezados para la barra ñlateral (sidebar)
st.sidebar.markdown("""
    <h1 style='display: flex; align-items: center; gap: 10px;'>
        <img src='https://previews.123rf.com/images/ylivdesign/ylivdesign1610/ylivdesign161001301/63631501-bandera-de-alemania-en-el-icono-de-estilo-de-dibujos-animados-aislado-en-el-fondo-blanco-del-c%C3%ADrculo.jpg' width='40' height='40'>
        BERLÍN
    </h1>
    """, unsafe_allow_html=True)
st.sidebar.title("DASHBOARD")
st.sidebar.subheader("Panel de selección")
View = st.sidebar.selectbox(label="Opciones", options=["Información","Modelado Explicativo","Modelado Predictivo"])

if View == "Información":
    st.title("Bienvenido a Berlín")
    
    # Mostrar imagen de Berlín
    st.image("logo.jpg", 
             caption="Vista panorámica de Berlín",  width=800)

    # Mostrar una breve descripción
    st.markdown("""
    Localización y sitio: 
    La ciudad de Berlín está situada en el noroeste de Alemania, a orillas de los ríos Spree y Havel. Es la capital de la República Federal de Alemania y uno de los 16 Estados federales. Con sus más de 3,4 millones de habitantes, es la urbe más poblada de Alemania y la mayor de la Unión Europea.

    Hay en Berlín tres diferentes sitios del Patrimonio Mundial: los “Palacios y Parques de Potsdam y de Berlín” fueron los primeros en ser inscritos en la Lista del Patrimonio Mundial.  Esta propiedad fue ampliada dos veces, en primer lugar en 1992, y luego en 1999 – año en que también fue inscrita la Museumsinsel (Isla de los Museos). Asimismo, las seis Ciudades del Modernismo de Berlín adquirieron el estatus de Patrimonio Mundial en 2008.
        
    Referencias históricas: 
    Ciudad jardín Falkenberg (1913-1916). Los arquitectos fueron Bruno Taut y Heinrich Tessenow, y el paisajista, Ludwig Lesser;
    Ciudad Schillerpark (1924-1930/1953-1957) concebida por los arquitectos Bruno Taut y Hans Hoffmann; su paisajista fue Walter Rossow);
    Ciudad Britz (1925-1930), diseñada por los arquitectos Bruno Taut y Martin Wagner; complementada por los paisajistas Leberecht Migge y Ottokar Wagler);
    Conjunto Carl Legien (1928-1930), cuyos arquitectos fueron Bruno Taut y Franz Hillinger);
    White City (1929-1931, proyectada por los arquitectos Otto Rudolf Salvisberg, Bruno Ahrends y Wilhelm Büning. Su paisajista fue Ludwig Lesser;
    Conjunto Siemensstadt (1929-1930), obra de los arquitectos Otto Bartning, Fred Forbart, Walter Gropius, Hugo Häring, Paul-Rudolf Henning y Hans Scharoun, y del paisajista Leberecht Migge).
    """)


# Contenido de la vista 1
if View == "Modelado Explicativo":
    #Encabezados para el dashboard
    # Título con icono de bandera
    # Mostramos el dataset
    st.subheader("Datos del dataset")
    st.write(df)
    st.subheader("Resumen estadístico de las columnas numéricas")
    # st.write(df.columns)
    st.write(df.describe())   

    # Widget extra: botón para mostrar frecuencias de todas las variables
    freq_button = st.sidebar.checkbox(label="Mostrar frecuencias de cada variable", key="freq_button")
    # # Generamos un button (Button) en la barra lateral (sidebar) para mostrar el linplot
    
    # Widgets de selección
    text_var_selected = st.sidebar.selectbox(label="Variables tipo texto: ", options=text_cols)
    numerics_var_selected = st.sidebar.selectbox(label="Variables numéricas", options=numeric_cols)

    # Botón para mostrar la gráfica
    mostrar_barplot = st.sidebar.button(label="Mostrar grafica de barras para su visualización", key="mostrar_barplot")

    # Mostrar gráfica solo si se presionó el botón
    if mostrar_barplot and text_var_selected:
        figure1 = px.bar(data_frame=df, 
                        x=df[text_var_selected],  # tomar solo una si hay varias
                        y=df[numerics_var_selected],
                        title='Gráfico de Barras')
        figure1.update_xaxes(automargin=True)
        figure1.update_yaxes(automargin=True)
        st.plotly_chart(figure1)


    # Si se presiona el botón, generamos la tabla de frecuencias
    if freq_button:
        st.subheader("Frecuencias de todas las variables")

        # Creamos un contenedor para mostrar múltiples tablas
        for col in df.columns:
            st.markdown(f"**Frecuencia de: `{col}`**")

            # Si es tipo texto o categoría
            if df[col].dtype == 'object' or df[col].dtype == 'bool':
                freq_table = df[col].value_counts(dropna=False).reset_index()
                freq_table.columns = [col, 'Frecuencia']
                st.dataframe(freq_table)

            # Si es numérica, mostramos frecuencias de valores únicos
            elif pd.api.types.is_numeric_dtype(df[col]):
                freq_table = df[col].value_counts(bins=10, dropna=False).reset_index()
                freq_table.columns = [f'Rango {col}', 'Frecuencia']
                st.dataframe(freq_table)

            # Línea separadora
            st.markdown("---")

    # Generamos el Contenido de la vista 2
elif View == "Modelado Predictivo":

        # Widget2: Checkbox
        # Generamos cuadros de seleccion en una barra lateral
        check_box1 = st.sidebar.checkbox(label="Regresión Lineal Simple", key="check_box1")
        check_box2 = st.sidebar.checkbox(label="Regresión Lineal Múltiple", key="check_box2")
        check_box3 = st.sidebar.checkbox(label="Regresión Logística", key="check_box3")

        # Mostramos el mapa de calor
        if check_box1:

            # if correlaciones_checkbox:

            # Encontramos todas las correlaciones entre las variables
            Corr_Factors = df.select_dtypes(include=['number']).corr()

            # Encontramos el valor absoluto de todas las correlaciones entre las variables
            Corr_Factors1 = abs(Corr_Factors)

            # Crear figura y mapa de calor
            fig, ax = mt.subplots(figsize=(20,10))
            sns.heatmap(Corr_Factors1, cmap='Blues', annot=True, fmt='.2f', ax=ax)

            # Mostrar en Streamlit
            st.subheader("Correlaciones entre las variables (Mapa de calor)")
            st.pyplot(fig)
            
            # if modelo:

            var_indep = st.selectbox(label="Variable independiente", options=numeric_cols, key="var_indep")
            var_dep = st.selectbox(label="Variable dependiente", options=numeric_cols, key="var_dep")

            model = LinearRegression()

            # Entrenamos el modelo con las variables independientes (X) y la variable dependiente (y)
            #model.fit(X=df[var_indep], y=df[var_dep])
            model.fit(X=df[[var_indep]], y=df[var_dep])

            # Evaluamos la eficiencia del modelo obtenido por medio del coeficiente R = Determinacion
            # coef_det = model.score(df[var_indep],df[var_dep])
            coef_det = model.score(df[[var_indep]], df[var_dep])
            st.write(f"**Coeficiente de determinación (R²):** {coef_det:.4f}")
            # Determinamos el coeficiente de correlación
            coef_Correl = np.sqrt(coef_det)
            st.write(f"**Coeficiente de correlación (R):** {coef_Correl:.4f}")

            # Predecimos los valores de la variable dep a partir de la var indep (nos da las predicciones, numero igual de filas)
            y_pred= model.predict(X=df[[var_indep]])
                
            # values_predictions = y_pred
            # Crear DataFrame para visualización con predicciones
            pred_df = df.copy()
            pred_df['Predicciones'] = y_pred

            # Mostrar tabla con valores reales y predichos
            st.write("### Valores reales vs Predicciones")
            st.dataframe(pred_df[[var_dep, 'Predicciones']])

            # Gráfico comparativo (Seaborn)
            st.write("### Gráfico de regresión")
            fig, ax = mt.subplots(figsize=(10, 5))
            sns.scatterplot(x=var_indep, y=var_dep, data=pred_df, color="blue", label="Real", ax=ax)
            sns.lineplot(x=var_indep, y='Predicciones', data=pred_df, color="red", label="Predicción", ax=ax)
            st.pyplot(fig)

        if check_box2:
            st.subheader("Regresión Lineal Múltiple")

            # Columnas numéricas para regresión
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # Selección de variables independientes
            selected_features = st.multiselect(
                label="Selecciona variables independientes",
                options=numeric_cols,
                default=numeric_cols[:2],
                key="multivar_indep"
            )

            # Selección de variable dependiente
            var_dep_multiple = st.selectbox(
                label="Selecciona variable dependiente",
                options=numeric_cols,
                key="multivar_dep"
            )

            # Validación de selección
            if selected_features:
                if var_dep_multiple in selected_features:
                    st.warning("La variable dependiente no debe estar entre las independientes.")
                else:
                    # Modelo de regresión
                    model_mult = LinearRegression()
                    X = df[selected_features]
                    y = df[var_dep_multiple]

                    # Entrenar modelo
                    model_mult.fit(X, y)

                    # Predicción
                    y_pred_mult = model_mult.predict(X)

                    # Métricas
                    r2 = model_mult.score(X, y)
                    r = np.sqrt(abs(r2))

                    st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")
                    st.write(f"**Coeficiente de correlación (R):** {r:.4f}")

                    # Coeficientes del modelo
                    # coef_df = pd.DataFrame({
                    #     "Variable": selected_features,
                    #     "Coeficiente": model_mult.coef_
                    # })
                    # st.write("### Coeficientes del modelo")
                    # st.dataframe(coef_df)

                    # DataFrame con predicciones
                    pred_df_mult = df.copy()
                    pred_df_mult['Predicciones'] = y_pred_mult

                    st.write("### Valores reales vs Predicciones")
                    st.dataframe(pred_df_mult[[var_dep_multiple, 'Predicciones']])

                    # Gráfico comparativo
                    st.write("### Gráfico comparativo de valores reales y predichos")
                    fig_mult, ax = mt.subplots(figsize=(10, 5))
                    sns.scatterplot(
                        x=pred_df_mult.index,
                        y=pred_df_mult[var_dep_multiple],
                        label="Valor Real",
                        color="blue",
                        ax=ax
                    )
                    sns.scatterplot(
                        x=pred_df_mult.index,
                        y=pred_df_mult['Predicciones'],
                        label="Predicción",
                        color="red",
                        ax=ax
                    )
                    ax.set_ylabel(var_dep_multiple)
                    ax.set_xlabel("Índice de muestra")
                    st.pyplot(fig_mult)
            else:
                st.warning("Selecciona al menos una variable independiente.")
            # st.subheader("Regresión Lineal Múltiple")

            # # Selección de variables independientes (múltiples)
            # selected_features = st.multiselect(
            #     label="Selecciona variables independientes",
            #     #options=df_encoded.columns,
            #     options=df.select_dtypes(include=np.number).columns,
            #     default=df.columns[:2],
            #     key="multivar_indep"
            # )

            # var_dep_multiple = st.selectbox(
            #     label="Selecciona variable dependiente",
            #     options=df.columns,
            #     key="multivar_dep"
            # )

            # if selected_features:
            #     model_mult = LinearRegression()

            #     X = df[selected_features]
            #     y = df[var_dep_multiple]


            #     # Entrenamos el modelo
            #     model_mult.fit(X, y)

            #     # Predicción
            #     #y_pred_mult = model_mult.predict(df[selected_features])
            #     y_pred_mult = model_mult.predict(X)

            #     # Coeficientes
            #     r2 = model_mult.score(df[selected_features], df[var_dep_multiple])
            #     r = np.sqrt(abs(r2))

            #     # Mostramos métricas
            #     st.write(f"**Coeficiente de determinación (R²):** {r2:.4f}")
            #     st.write(f"**Coeficiente de correlación (R):** {r:.4f}")

            #     # # Coeficientes del modelo
            #     # st.subheader("Coeficientes del modelo")
            #     # coef_df = pd.DataFrame({
            #     #     "Variable": selected_features,
            #     #     "Coeficiente": model_mult.coef_
            #     # })
            #     # st.dataframe(coef_df)

            #     # DataFrame con predicciones
            #     # pred_df_mult = df.copy()
            #     # pred_df_mult['Predicciones'] = y_pred_mult
            #     pred_df_mult = df.copy()
            #     pred_df_mult['Predicciones'] = y_pred_mult

            #     st.write("### Valores reales vs Predicciones")
            #     st.dataframe(pred_df_mult[[var_dep_multiple, 'Predicciones']])

            #     # Gráfico comparativo con colores diferenciados
            #     st.write("### Gráfico comparativo de valores reales y predichos")
            #     fig_mult, ax = mt.subplots(figsize=(10, 5))
            #     sns.scatterplot(
            #         x=pred_df_mult.index, 
            #         y=pred_df_mult[var_dep_multiple], 
            #         label="Valor Real", 
            #         color="blue", 
            #         ax=ax
            #     )
            #     sns.scatterplot(
            #         x=pred_df_mult.index, 
            #         y=pred_df_mult['Predicciones'], 
            #         label="Predicción", 
            #         color="red", 
            #         ax=ax
            #     )
            #     ax.set_ylabel(var_dep_multiple)
            #     ax.set_xlabel(selected_features)
            #     st.pyplot(fig_mult)
            # else:
            #     st.warning("Selecciona al menos una variable independiente.")


        if check_box3:
            st.subheader("Regresión Logística")

            # Selección de variables
            target_log = st.selectbox("Selecciona la variable objetivo (debe ser categórica)", options=numeric_cols, key="log_target")
            features_log = st.multiselect("Selecciona variables predictoras (numéricas)", options=numeric_cols, key="log_features")

            if target_log and features_log:
                # Eliminar filas con valores faltantes
                #df_log = df[[target_log] + features_log].dropna()

                # Codificar variable objetivo si es categórica
                # y_log = pd.factorize(df_log[target_log])[0]
                # X_log = df_log[features_log]

                # Dividir en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

                # Modelo
                model_log = LogisticRegression(max_iter=1000)
                model_log.fit(X_train, y_train)

                # Predicción
                y_pred_log = model_log.predict(X_test)

                # Métricas
                acc = accuracy_score(y_test, y_pred_log)
                cm = confusion_matrix(y_test, y_pred_log)
                report = classification_report(y_test, y_pred_log, output_dict=True)

                # Mostrar métricas
                st.write(f"**Exactitud (Accuracy):** {acc:.4f}")

                # Mostrar matriz de confusión
                st.write("### Matriz de Confusión")
                fig_cm, ax_cm = mt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
                st.pyplot(fig_cm)

                # Mostrar precisión y sensibilidad por clase
                st.write("### Reporte de Clasificación (Precisión y Sensibilidad por clase)")
                st.dataframe(pd.DataFrame(report).transpose())

            else:
                st.warning("Selecciona la variable objetivo y al menos una variable predictora.")
            
