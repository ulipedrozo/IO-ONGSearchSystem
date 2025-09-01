# -*- coding: utf-8 -*-
# Interfaz de Usuario - Sistema de Búsqueda de ONGs

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, 'ongs_procesadas.csv')

# Importar el sistema
try:
    from Model_Enhanced import SistemaEmbeddingsONGAvanzado
    SISTEMA_DISPONIBLE = True
except ImportError:
    SISTEMA_DISPONIBLE = False
    st.error("No se pudo importar el sistema de embeddings")

# Configuración
st.set_page_config(
    page_title="Sistema de Recomendación de ONGs",
    page_icon="🤝",
    layout="wide"
)

# CSS personalizado
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
    }
    h1 { color: #1976d2; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# Estado de sesión
if 'sistema' not in st.session_state:
    st.session_state.sistema = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'historial' not in st.session_state:
    st.session_state.historial = []

@st.cache_resource
def cargar_sistema():
    """Carga el sistema de embeddings"""
    try:
        # Cargar datos
        if not os.path.exists(csv_path):
            st.error(f"No se encontró el archivo {csv_path}")
            return None, None
        
        df = pd.read_csv(csv_path)
        
        # Crear sistema
        if SISTEMA_DISPONIBLE:
            sistema = SistemaEmbeddingsONGAvanzado()
            sistema.ajustar(df)
            return sistema, df
        else:
            return None, df
            
    except Exception as e:
        st.error(f"Error al cargar el sistema: {str(e)}")
        return None, None

def busqueda_simple(query, df, top_k=5):
    """Búsqueda simple por palabras clave (fallback)"""
    query_lower = query.lower()
    df_temp = df.copy()
    df_temp['score'] = 0
    
    # Buscar en campos principales
    for campo in ['nombre', 'mision', 'servicios', 'categoria']:
        if campo in df_temp.columns:
            mask = df_temp[campo].fillna('').str.contains(query_lower, case=False, regex=False)
            df_temp.loc[mask, 'score'] += 1
    
    # Filtrar y ordenar
    df_resultados = df_temp[df_temp['score'] > 0].nlargest(top_k, 'score')
    
    # Formato de salida
    resultados = []
    for _, row in df_resultados.iterrows():
        resultados.append({
            'nombre': row['nombre'],
            'categoria': row.get('categoria', ''),
            'mision': str(row.get('mision', ''))[:200],
            'servicios': str(row.get('servicios', '')),
            'ubicacion': str(row.get('ubicacion', '')),
            'contacto': str(row.get('contacto', '')),
            'similitud': row['score'] / 4
        })
    
    return resultados

def main():
    # Título
    st.markdown("# 🤝 Sistema de Recomendación de ONGs")
    st.markdown("### Encuentra la ayuda que necesitas")
    
    # Cargar sistema
    if st.session_state.sistema is None:
        with st.spinner('Cargando sistema...'):
            sistema, df = cargar_sistema()
            st.session_state.sistema = sistema
            st.session_state.df = df
    
    if st.session_state.df is None:
        st.error("No se pudo cargar la base de datos")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📊 Información")
        st.metric("Total ONGs", len(st.session_state.df))
        st.metric("Búsquedas", len(st.session_state.historial))
        
        if st.session_state.historial:
            st.markdown("### 📜 Historial")
            for busqueda in st.session_state.historial[-5:]:
                st.text(f"• {busqueda}")
    
    # Búsqueda
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        query = st.text_input(
            "¿Qué tipo de ayuda necesitas?",
            placeholder="Ej: apoyo escolar, ayuda alimentaria..."
        )
        
        buscar = st.button("🔍 Buscar ONGs", type="primary", use_container_width=True)
    
    # Sugerencias
    st.markdown("#### 💡 Búsquedas sugeridas:")
    col1, col2, col3, col4 = st.columns(4)
    
    sugerencias = ["Educación", "Salud", "Alimentación", "Apoyo psicológico"]
    for col, sug in zip([col1, col2, col3, col4], sugerencias):
        with col:
            if st.button(sug):
                query = sug
                buscar = True
    
    # Procesar búsqueda
    if buscar and query:
        st.session_state.historial.append(query)
        
        with st.spinner('Buscando...'):
            # Intentar con sistema de embeddings
            if st.session_state.sistema and SISTEMA_DISPONIBLE:
                try:
                    resultados = st.session_state.sistema.buscar_ongs_similares(query)
                except:
                    # Fallback a búsqueda simple
                    resultados = busqueda_simple(query, st.session_state.df)
            else:
                # Búsqueda simple
                resultados = busqueda_simple(query, st.session_state.df)
        
        # Mostrar resultados
        st.markdown("---")
        st.markdown("## 🎯 Resultados")
        
        if resultados:
            st.success(f"Encontré {len(resultados)} organizaciones:")
            
            # Tabs
            tab1, tab2 = st.tabs(["📋 Lista", "📊 Análisis"])
            
            with tab1:
                for i, ong in enumerate(resultados, 1):
                    with st.expander(f"**{i}. {ong['nombre']}** - {ong['similitud']:.0%} relevante"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Categoría:** {ong['categoria']}")
                            st.write(f"**Misión:** {ong['mision']}")
                        with col2:
                            st.write(f"**Servicios:** {ong['servicios']}")
                            st.write(f"**Contacto:** {ong['contacto']}")
            
            with tab2:
                # Gráfico de relevancia
                df_res = pd.DataFrame(resultados)
                fig = px.bar(df_res, x='similitud', y='nombre', orientation='h',
                           title='Relevancia de Resultados')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No encontré resultados para tu búsqueda")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center'>Investigación Operativa I - UNICEN</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()