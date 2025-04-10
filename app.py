# Creamos el archivo de la app en el interprete principal
import pandas as pd 
import plotly.express as px
import streamlit as st
import numpy as np
import matplotlib.pyplot as mt
import seaborn as sns

# Aplicamos estilos con CSS
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;700&display=swap');

        html, body {
            font-family: 'Montserrat', sans-serif;
            color: #2E4053;
            background-color: #e59866;
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
            background-color: #e59866;
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

    # Seleccionamos todas las columnas del dataset
    # cols = df.select_dtypes(['float','int','object'])
    # cols_columns = cols.columns # Mostramos las columnas



    return df, numeric_cols, text_cols, unique_categories_room, numeric_df
    # return df, cols, cols_columns

# Cargo los datos obtenidos de la funcion "load_data()"
df, numeric_cols, text_cols,  unique_categories_room, numeric_df = load_data()
# df, cols, cols_columns = load_data()

#Creacion del dashboard

# Generamos las paginas para utilizar dentro del diseño
# Widget 1: Selectbox
# Menú desplegable de opciones de las páginas seleccionadas
View = st.selectbox(label="Opciones", options=["Modelado Explicativo","Modelado Predictivo"])

# Contenido de la vista 1
if View == "Modelado Explicativo":
    #Encabezados para el dashboard
    # Título con icono de bandera
    st.markdown("""
    <h1 style='display: flex; align-items: center; gap: 10px;'>
        <img src='https://previews.123rf.com/images/ylivdesign/ylivdesign1610/ylivdesign161001301/63631501-bandera-de-alemania-en-el-icono-de-estilo-de-dibujos-animados-aislado-en-el-fondo-blanco-del-c%C3%ADrculo.jpg' width='40' height='40'>
        BERLÍN
    </h1>
    """, unsafe_allow_html=True)

    # Generamos los encabezados para la barra ñlateral (sidebar)
    st.sidebar.title("DASHBOARD")
    st.sidebar.header("Sidebar")
    st.sidebar.subheader("Panel de selección")

    # Widget2: Checkbox
    # Generamos un cuadro de seleccion en una barra lateral
    check_box = st.sidebar.checkbox(label="Mostrar Dataset", key="check_box")

    # Condicional para que aparezca el dataframe
    if check_box:
        # Mostramos el dataset
        st.subheader("Datos del dataset")
        st.write(df)
        st.subheader("Resumen estadístico de las columnas numéricas")
        # st.write(df.columns)
        st.write(df.describe())

    # Widget extra: botón para mostrar frecuencias de todas las variables
    freq_button = st.sidebar.checkbox(label="Mostrar frecuencias de cada variable", key="freq_button")
    # # Generamos un button (Button) en la barra lateral (sidebar) para mostrar el linplot
    mostrar_lineplot = st.sidebar.button(label="Mostrar grafica tipo lineplot", key="mostrar_lineplot")

    # Mostrar imagen y descripción solo si no están activados los checkboxes
    if not check_box and not freq_button and not mostrar_lineplot:
        st.image("https://media.istockphoto.com/id/486804588/es/vector/edificios-de-la-ciudad-de-berl%C3%ADn.jpg?s=612x612&w=0&k=20&c=I-3dwZC2-iLNZWN3mcLxTwbhTBP9LGHb5og7LmozGjs=",
                caption="Vista de Berlín", use_container_width=True)

        st.markdown("""
        **Berlín** es la capital de Alemania y una de las ciudades más vibrantes y multiculturales de Europa. 
        Con una rica historia, arquitectura icónica, y una vida cultural dinámica, Berlín es conocida por sus museos, su escena artística, 
        y su combinación única de lo moderno con lo histórico. Desde la Puerta de Brandeburgo hasta el Muro de Berlín, 
        la ciudad ofrece una experiencia única tanto para residentes como visitantes.
        """)

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



    # Widget 3
    # Generamos un cuadro de multiselección (Y) para seleccionar variables a graficar
    numerics_vars_selected = st.sidebar.multiselect(label="Variables graficales: ", options = numeric_cols)
    # variables = st.sidebar.multiselect(label="Variables a graficar (Y)", options= cols_columns)

    # Widget 3 Selectbox
    # Menu desplegable de opciones de la variable categórica seleccionada
    category_selected = st.sidebar.selectbox(label="Categorias variable room_type", options= unique_categories_room)
    # variables1 = st.sidebar.multiselect(label="Variables a graficar (X)", options= cols_columns)


    # Widget 4: Button
    # Generamos un button (Button) en la barra lateral (sidebar) para mostrar las variables tipo texto
    # Button = st.sidebar.button(label="Mostrar variables string")

    # # Condicional para que aparezca el dataframe
    # if Button:
    #     #Mostramos el dataset
    #     st.write(text_cols)

#Graph 1: LINEPLOT
# Despliqgue de unn lineplot, definiendo las variables "X caytegóricas" y "Y numéricas"
    data = df[df['room_type'] == category_selected]
    data_features = data[numerics_vars_selected]
    # figure1 = px.line(data_frame=data_features, x=data_features.index,
    #                     y=numerics_vars_selected, title=str('Room type'),
    #                     width=1600, height=600)
    

    figure1 = px.bar(data_frame=data_features, x=data_features.index,
                        y=numerics_vars_selected, 
                        title='Room type: ' + str(category_selected),
                        width=1600, height=600)
    
    st.plotly_chart(figure1)
    #Condicional para que aparezca la grafica
    # Si se presiona el botón y se han seleccionado variables
    # if mostrar_lineplot:
    #     if variables1 and variables:
    #         for x_var in variables1:
    #             fig = px.line(
    #                 data_frame=df,
    #                 x=x_var,
    #                 y=variables,
    #                 title=f'Lineplot: {", ".join(variables)} vs {x_var}',
    #                 width=1000,
    #                 height=500
    #             )
    #             st.plotly_chart(fig)
    #     else:
    #         st.warning("Selecciona al menos una variable para X y Y.")

    # Generamos el Contenido de la vista 2
elif View == "Modelado Predictivo":
        st.markdown("""
        <h1 style='display: flex; align-items: center; gap: 10px;'>
            <img src='https://previews.123rf.com/images/ylivdesign/ylivdesign1610/ylivdesign161001301/63631501-bandera-de-alemania-en-el-icono-de-estilo-de-dibujos-animados-aislado-en-el-fondo-blanco-del-c%C3%ADrculo.jpg' width='40' height='40'>
            BERLÍN
        </h1>
        """, unsafe_allow_html=True)

        # Generamos los encabezados para la barra ñlateral (sidebar)
        st.sidebar.title("DASHBOARD")
        st.sidebar.header("Sidebar")
        st.sidebar.subheader("Panel de selección")

        # Widget2: Checkbox
        # Generamos cuadros de seleccion en una barra lateral
        check_box1 = st.sidebar.checkbox(label="Regresión Lineal Simple", key="check_box1")
        check_box2 = st.sidebar.checkbox(label="Regresión Lineal Múltiple", key="check_box2")
        check_box3 = st.sidebar.checkbox(label="Regresión Logística", key="check_box3")

        # Mostramos el mapa de calor
        if check_box1:

            # Encontramos todas las correlaciones entre las variables
            Corr_Factors = df.select_dtypes(include=['number']).corr()

            # Encontramos el valor absoluto de todas las correlaciones entre las variables
            Corr_Factors1 = abs(Corr_Factors)

            # Crear figura y mapa de calor
            fig, ax = mt.subplots(figsize=(16,10))
            sns.heatmap(Corr_Factors1, cmap='Blues', annot=True, fmt='.2f', ax=ax)

            # Mostrar en Streamlit
            st.subheader("Correlaciones entre las variables (Mapa de calor)")
            st.pyplot(fig)


        # Mostrar imagen y descripción solo si no están activados los checkboxes
        if not check_box1 and not check_box2 and not check_box3:
            st.image("https://media.istockphoto.com/id/486804588/es/vector/edificios-de-la-ciudad-de-berl%C3%ADn.jpg?s=612x612&w=0&k=20&c=I-3dwZC2-iLNZWN3mcLxTwbhTBP9LGHb5og7LmozGjs=",
                    caption="Vista de Berlín", use_container_width=True)

            st.markdown("""
            **Berlín** es la capital de Alemania y una de las ciudades más vibrantes y multiculturales de Europa. 
            Con una rica historia, arquitectura icónica, y una vida cultural dinámica, Berlín es conocida por sus museos, su escena artística, 
            y su combinación única de lo moderno con lo histórico. Desde la Puerta de Brandeburgo hasta el Muro de Berlín, 
            la ciudad ofrece una experiencia única tanto para residentes como visitantes.
            """)

        
    
        # #GRAPH 2 SCATTERPLOT
        # x_selected = st.sidebar.selectbox(label="x", options=numeric_cols)
        # y_selected = st.sidebar.selectbox(label="y", options=numeric_cols)
        # figure2 = px.scatter(data_frame=numeric_df, x=x_selected, y=y_selected,
        #                         title='Dispersiones')
        # st.plotly_chart(figure2)

#     # Vista número 3

# elif View == "View3":
#         # Generamos los encabezados para el dashboard
#         st.title("TITANIC")
#         st.header("Panel Principal")
#         st.subheader("Scatter Plot")
    
#         Variable_cat = st.sidebar.selectbox(label="Varibale categórica", options=text_cols)
#         Variable_num = st.sidebar.selectbox(label="Varibale numérica", options=numeric_cols)

#         #GRAPH 3
#         # Despiegue de un pie plot, definiendo las variables "X categoricas" y "Y numéricas"
#         figure3= px.line(data_frame=data_features, x=data_features.index,
#                             y=numerics_vars_selected, title=str('Features of passengers'),
#                             width=1600, height=600)
#         st.plotly_chart(figure3)

#     # CONTENIDO DE LA VISTA 4
# elif View == "View4":
#     # Generamos los encabezados para el dashboard
#         st.title("TITANIC")
#         st.header("Panel Principal")
#         st.subheader("Scatter Plot")
    
#         Variable_cat = st.sidebar.selectbox(label="Varibale categórica", options=text_cols)
#         Variable_num = st.sidebar.selectbox(label="Varibale numérica", options=numeric_cols)

#         figure4 = px.bar(data_frame=df, x=df[Variable_cat],
#                         y=df[Variable_num],title=str('Features of') +' '+ 'Passengers')
#         figure4.update_xaxes(automargin=True)
#         figure4.update_yaxes(automargin=True)
#         st.plotly_chart(figure4)

