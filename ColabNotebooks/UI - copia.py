# Interfaz de Usuario - Sistema de Búsqueda Inteligente de ONGs
# Requiere: streamlit, pandas, plotly

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os
import sys
# Importar sistema
from Model_Enhanced import SistemaEmbeddingsONGAvanzado

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'ongs_procesadas.csv')

# Configuración de la página
st.set_page_config(
    page_title="Sistema de recomendación de ONGs",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.05);
    }
    .ong-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    h1 {
        color: #1976d2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    </style>
    """, unsafe_allow_html=True)

# Inicializar estado de sesión
if 'historial_busquedas' not in st.session_state:
    st.session_state.historial_busquedas = []
if 'sistema_cargado' not in st.session_state:
    st.session_state.sistema_cargado = False
if 'df_ongs' not in st.session_state:
    st.session_state.df_ongs = None
if 'sistema_embeddings' not in st.session_state:
    st.session_state.sistema_embeddings = None

# Función para cargar el sistema (simulada para el ejemplo)
@st.cache_resource
def cargar_sistema_embeddings():
    global sistema_embeddings, df_ongs

    with st.spinner('Cargando sistema de búsqueda inteligente...'):
        time.sleep(2)  # Simular carga

        # En producción, aquí cargarías tu sistema real:
        sistema_embeddings = SistemaEmbeddingsONGAvanzado()
        df = pd.read_csv(csv_path)
        sistema_embeddings.ajustar(df)

        return df, sistema_embeddings  # En producción, retornarías (df, sistema)

# Función simulada de búsqueda
def buscar_ongs(query, df, top_k=5):
    global sistema_embeddings
    
    """Simula la búsqueda semántica
    # Para el ejemplo, hacemos una búsqueda simple por palabras clave
    query_lower = query.lower()
    scores = []

    for idx, row in df.iterrows():
        score = 0
        # Buscar en todos los campos de texto
        for campo in ['nombre', 'categoria', 'mision', 'servicios']:
            if query_lower in str(row[campo]).lower():
                score += 1

        # Palabras clave específicas
        if 'niño' in query_lower and 'Infancia' in row['categoria']:
            score += 2
        if 'educación' in query_lower and 'Educación' in row['categoria']:
            score += 2
        if 'salud' in query_lower and 'Salud' in row['categoria']:
            score += 2
        if 'mujer' in query_lower and 'Género' in row['categoria']:
            score += 2

        scores.append(score)

    # Ordenar por score y retornar top_k
    df['score'] = scores
    df_sorted = df.sort_values('score', ascending=False)

    resultados = []
    for idx, row in df_sorted.head(top_k).iterrows():
        if row['score'] > 0:
            resultados.append({
                'nombre': row['nombre'],
                'categoria': row['categoria'],
                'mision': row['mision'],
                'servicios': row['servicios'],
                'ubicacion': row['ubicacion'],
                'contacto': row['contacto'],
                'similitud': min(row['score'] / 3, 0.99)  # Normalizar score
            })"""

    resultados = sistema_embeddings.buscar_ongs_similares(
        query, 
        top_k=top_k,
        umbral=0.1
    )
    return resultados

# INTERFAZ PRINCIPAL
def main():
    global sistema_embeddings, df_ongs

    # Header con logo e información
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("# Sistema de recomendación de ONGs")
        st.markdown("### Encuentra la ayuda que necesitas")

    # Cargar sistema si no está cargado
    if not st.session_state.sistema_cargado:
        df_ongs, sistema_embeddings = cargar_sistema_embeddings()
        st.session_state.df_ongs = df_ongs
        st.session_state.sistema_embeddings = sistema_embeddings
        st.session_state.sistema_cargado = True

    # Barra lateral con información y filtros
    with st.sidebar:
        st.markdown("## 📊 Panel de Control")

        # Información del sistema
        st.info(f"""
        **Sistema cargado**
        Total ONGs: {len(st.session_state.df_ongs)}
        Categorías: {st.session_state.df_ongs['categoria'].nunique()}
        """)

        # Filtros opcionales
        st.markdown("### 🔍 Filtros de Búsqueda")

        categorias = ['Todas'] + sorted(st.session_state.df_ongs['categoria'].unique().tolist())
        categoria_filtro = st.selectbox("Categoría:", categorias)

        # Estadísticas
        st.markdown("### 📈 Estadísticas de Uso")
        st.metric("Búsquedas realizadas", len(st.session_state.historial_busquedas))

        # Historial
        if st.session_state.historial_busquedas:
            st.markdown("### 🕐 Historial Reciente")
            for busqueda in st.session_state.historial_busquedas[-5:]:
                st.text(f"• {busqueda}")

    # Área principal de búsqueda
    st.markdown("---")

    # Caja de búsqueda prominente
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query = st.text_input(
            "¿Qué tipo de ayuda necesitas?",
            placeholder="Ej: apoyo escolar para niños, ayuda alimentaria, etc.",
            help="Describe en tus propias palabras qué tipo de asistencia buscas"
        )

        buscar_button = st.button("🔍 Buscar ONGs", type="primary", use_container_width=True)

    # Sugerencias de búsqueda
    st.markdown("#### 💡 Sugerencias de búsqueda:")
    col1, col2, col3, col4 = st.columns(4)

    sugerencias = [
        "Educación para niños",
        "Ayuda alimentaria",
        "Apoyo psicológico",
        "Salud gratuita"
    ]

    for col, sugerencia in zip([col1, col2, col3, col4], sugerencias):
        with col:
            if st.button(sugerencia, key=f"sug_{sugerencia}"):
                query = sugerencia
                buscar_button = True

    # Procesar búsqueda
    if buscar_button and query:
        # Agregar al historial
        st.session_state.historial_busquedas.append(query)

        # Mostrar spinner mientras busca
        with st.spinner('Buscando las mejores ONGs para ti...'):
            # Filtrar por categoría si se seleccionó
            df_filtrado = st.session_state.df_ongs
            if categoria_filtro != 'Todas':
                df_filtrado = df_filtrado[df_filtrado['categoria'] == categoria_filtro]

            # Realizar búsqueda
            resultados = buscar_ongs(query, df_filtrado)

        # Mostrar resultados
        st.markdown("---")
        st.markdown("## 🎯 Resultados de Búsqueda")

        if resultados:
            st.success(f"Encontré {len(resultados)} organizaciones que pueden ayudarte:")

            # Tabs para diferentes vistas
            tab1, tab2, tab3 = st.tabs(["📋 Lista Detallada", "📊 Vista Comparativa", "🗺️ Información de Contacto"])

            with tab1:
                # Vista de tarjetas
                for i, ong in enumerate(resultados, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"### {i}. {ong['nombre']}")
                            st.markdown(f"**Categoría:** {ong['categoria']} | **Relevancia:** {ong['similitud']:.0%}")

                            with st.expander("Ver detalles completos"):
                                st.markdown(f"**📋 Misión:**  \n{ong['mision']}")
                                st.markdown(f"**🛠️ Servicios:**  \n{ong['servicios']}")
                                st.markdown(f"**📍 Ubicación:**  \n{ong['ubicacion']}")
                                st.markdown(f"**📞 Contacto:**  \n{ong['contacto']}")

                        with col2:
                            st.markdown(f"<div class='metric-card'><h3>{ong['similitud']:.0%}</h3><p>Relevancia</p></div>",
                                      unsafe_allow_html=True)

                        st.markdown("---")

            with tab2:
                # Vista comparativa
                df_resultados = pd.DataFrame(resultados)

                # Gráfico de relevancia
                fig_relevancia = px.bar(
                    df_resultados,
                    x='similitud',
                    y='nombre',
                    orientation='h',
                    title='Relevancia de las ONGs encontradas',
                    labels={'similitud': 'Relevancia', 'nombre': 'Organización'},
                    color='similitud',
                    color_continuous_scale='Viridis'
                )
                fig_relevancia.update_layout(height=400)
                st.plotly_chart(fig_relevancia, use_container_width=True)

                # Distribución por categorías
                fig_categorias = px.pie(
                    df_resultados,
                    names='categoria',
                    title='Distribución por Categorías',
                    hole=0.3
                )
                st.plotly_chart(fig_categorias, use_container_width=True)

            with tab3:
                # Mapa de contactos (tabla interactiva)
                st.markdown("### 📍 Información de Contacto Rápido")

                contacto_df = df_resultados[['nombre', 'ubicacion', 'contacto']].copy()
                st.dataframe(
                    contacto_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "nombre": st.column_config.TextColumn("Organización", width="medium"),
                        "ubicacion": st.column_config.TextColumn("Dirección", width="medium"),
                        "contacto": st.column_config.TextColumn("Contacto", width="medium"),
                    }
                )

                # Botón para exportar
                csv = contacto_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar información de contacto",
                    data=csv,
                    file_name=f'contactos_ongs_{datetime.now().strftime("%Y%m%d")}.csv',
                    mime='text/csv'
                )

        else:
            st.warning("No encontré organizaciones que coincidan exactamente con tu búsqueda.")
            st.info("Intenta con términos más generales o explora las categorías disponibles.")

    # Sección de información adicional
    with st.expander("ℹ️ Acerca de este sistema"):
        st.markdown("""
        ### 🤖 Sistema de Búsqueda Inteligente

        Este buscador utiliza Inteligencia Artificial y Redes Neuronales de tipo Transformers para entender tu consulta y encontrar
        las organizaciones más relevantes para tus necesidades.

        **Características principales:**
        - 🧠 Comprensión de lenguaje natural en español
        - 🎯 Búsqueda semántica avanzada
        - 📊 Ranking por relevancia
        - 🔍 Filtros por categoría

        **¿Cómo funciona?**
        1. Describe tu necesidad en tus propias palabras
        2. El sistema analiza tu consulta usando embeddings semánticos
        3. Compara con la base de datos de ONGs
        4. Retorna las organizaciones más relevantes ordenadas por similitud

        **Tecnología:** Sentence-BERT multilingual, procesamiento de lenguaje natural
        """)

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown(
            "<p style='text-align: center; color: gray;'>Version 1.0.0 - Investigación Operativa I - Facultad de Ciencias Exactas - UNICEN</p>",
            unsafe_allow_html=True
        )

# Ejecutar la aplicación
if __name__ == "__main__":
    main()