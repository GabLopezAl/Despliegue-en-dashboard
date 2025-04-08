# Creamos el archivo de la app en el interprete principal
import pandas as pd 
import plotly.express as px
import streamlit as st

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
        }
        .stTitle {
            color: #1F618D;
        }
        .stHeader {
            color: #2874A6;
        }
        .stSidebar {
            background-color: #D6EAF8;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Definimos la instancia
@st.cache_resource

# Definimos la función de carga de datos
def load_data():
    df = pd.read_csv("Datos_limpios_Berlin.csv")

    # Selecciono las columnas tipo numéricas del dataframe
    numeric_df = df.select_dtypes(['float','int']) # Devuelve las columnas
    numeric_cols = numeric_df.columns # Devuekve lista de columnas

    # Selecciono las columnas tipo texto
    text_df = df.select_dtypes(['object']) # Seleccionamos los datos tipo cadena (objeto)
    text_cols = text_df.columns # Mostramos las columnas

    # # Selecciono algunas columnas categoricas de valores
    # categorical_column_sex = df['Sex']

    # Obtengo los valores unicos
    # unique_categories_sex = categorical_column_sex.unique()

    return df, numeric_cols, text_cols, numeric_df

# Cargo los datos obtenidos de la funcion "load_data()"
df, numeric_cols, text_cols,  numeric_df = load_data()

#Creacion del dashboard

# Generamos las paginas para utilizar dentro del diseño
# Widget 1: Selectbox
# Menú desplegable de opciones de las páginas seleccionadas
View = st.selectbox(label="View", options=["Modelado Explicativo","Modelado predictivo"])

# Contenido de la vista 1
if View == "Modelado Explicativo":
    #Encabezados para el dashboard
    st.title("TITANIC")
    st.header("Sidebar")
    st.subheader("Line Plot")

    # Generamos los encabezados para la barra ñlateral (sidebar)
    st.sidebar.title("DASHBOARD")
    st.sidebar.header("Sidebar")
    st.sidebar.subheader("Panel de selección")

    # Widget2: Checkbox
    # Generamos un cuadro de seleccion en una barra lateral
    check_box = st.sidebar.checkbox(label="Mostrar Dataset")

    # Condicional para que aparezca el dataframe
    if check_box:
        # Mostramos el dataset
        st.write(df)
        st.write(df.columns)
        st.write(df.describe())


    # Widget 3
    # Generamos un cuadro de multiselección (Y) para seleccionar variables a graficar
    numerics_vars_selected = st.sidebar.multiselect(label="Variables graficales: ", options = numeric_cols)

    # Widget 3 Selectbox
    # Menu desplegable de opciones de la variable categórica seleccionada
    category_selected = st.sidebar.selectbox(label="Categorias", options= unique_categories_sex)

    # Widget 4: Button
    # Generamos un button (Button) en la barra lateral (sidebar) para mostrar las variables tipo texto
    Button = st.sidebar.button(label="Mostrar variables string")

    # Condicional para que aparezca el dataframe
    if Button:
        #Mostramos el dataset
        st.write(text_cols)

#Graph 1: LINEPLOT
# Despliqgue de unn lineplot, definiendo las variables "X caytegóricas" y "Y numéricas"
    data = df[df['Sex'] == category_selected]
    data_features = data[numerics_vars_selected]
    figure1 = px.line(data_frame=data_features, x=data_features.index,
                        y=numerics_vars_selected, title=str('Features of passengers'),
                        width=1600, height=600)
     
    # Generamos un button (Button) en la barra lateral (sidebar) para mostrar el linplot
    Button2 = st.sidebar.button(label="Mostrar grafica tipo lineplot")

    #Condicional para que aparezca la grafica
    if Button2:
        st.plotly_chart(figure1)

    # Generamos el Contenido de la vista 2
elif View == "View2":
        # Generamos los encabezados para el dashboard
        st.title("TITANIC")
        st.header("Panel Principal")
        st.subheader("Scatter Plot")
    
        #GRAPH 2 SCATTERPLOT
        x_selected = st.sidebar.selectbox(label="x", options=numeric_cols)
        y_selected = st.sidebar.selectbox(label="y", options=numeric_cols)
        figure2 = px.scatter(data_frame=numeric_df, x=x_selected, y=y_selected,
                                title='Dispersiones')
        st.plotly_chart(figure2)

    # Vista número 3

elif View == "View3":
        # Generamos los encabezados para el dashboard
        st.title("TITANIC")
        st.header("Panel Principal")
        st.subheader("Scatter Plot")
    
        Variable_cat = st.sidebar.selectbox(label="Varibale categórica", options=text_cols)
        Variable_num = st.sidebar.selectbox(label="Varibale numérica", options=numeric_cols)

        #GRAPH 3
        # Despiegue de un pie plot, definiendo las variables "X categoricas" y "Y numéricas"
        figure3= px.line(data_frame=data_features, x=data_features.index,
                            y=numerics_vars_selected, title=str('Features of passengers'),
                            width=1600, height=600)
        st.plotly_chart(figure3)

    # CONTENIDO DE LA VISTA 4
elif View == "View4":
    # Generamos los encabezados para el dashboard
        st.title("TITANIC")
        st.header("Panel Principal")
        st.subheader("Scatter Plot")
    
        Variable_cat = st.sidebar.selectbox(label="Varibale categórica", options=text_cols)
        Variable_num = st.sidebar.selectbox(label="Varibale numérica", options=numeric_cols)

        figure4 = px.bar(data_frame=df, x=df[Variable_cat],
                        y=df[Variable_num],title=str('Features of') +' '+ 'Passengers')
        figure4.update_xaxes(automargin=True)
        figure4.update_yaxes(automargin=True)
        st.plotly_chart(figure4)

