# -*- coding: utf-8 -*-
# Sistema Avanzado de Embeddings con Fine-tuning para ONGs

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path
import pickle
from typing import Dict, List, Any
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SistemaEmbeddingsONGAvanzado:
    """Sistema mejorado de embeddings para ONGs"""
    
    def __init__(self, nombre_modelo='paraphrase-multilingual-MiniLM-L12-v2', usar_cache=True):
        self.nombre_modelo = nombre_modelo
        self.usar_cache = usar_cache
        self.modelo = None
        self.embeddings = None
        self.df = None
        
        # Directorios
        self.dir_cache = Path('cache_embeddings')
        self.dir_cache.mkdir(exist_ok=True)
        
        self._cargar_modelo()
    
    def _cargar_modelo(self):
        """Carga el modelo"""
        logger.info(f"Cargando modelo: {self.nombre_modelo}")
        self.modelo = SentenceTransformer(self.nombre_modelo)
    
    def crear_texto_combinado(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crea texto combinado de los campos principales"""
        def combinar_campos(row):
            campos = ['nombre', 'mision', 'servicios', 'categoria']
            textos = []
            
            for campo in campos:
                if campo in row and pd.notna(row[campo]):
                    texto = str(row[campo]).strip()
                    if texto:
                        textos.append(texto)
            
            return ' '.join(textos)
        
        df['texto_combinado'] = df.apply(combinar_campos, axis=1)
        return df
    
    def ajustar(self, df: pd.DataFrame, fine_tune: bool = False):
        """Ajusta el sistema con los datos"""
        logger.info("Ajustando sistema de embeddings...")
        
        self.df = df.copy()
        self.df = self.crear_texto_combinado(self.df)
        
        # Cache
        cache_file = self.dir_cache / f"embeddings_{len(self.df)}_{self.nombre_modelo.replace('/', '_')}.pkl"
        
        if self.usar_cache and cache_file.exists():
            logger.info("Cargando embeddings desde cache...")
            with open(cache_file, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            logger.info("Generando embeddings...")
            textos = self.df['texto_combinado'].tolist()
            self.embeddings = self.modelo.encode(
                textos,
                show_progress_bar=True,
                convert_to_tensor=False
            )
            
            if self.usar_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.embeddings, f)
        
        logger.info(f"Sistema ajustado. Shape: {self.embeddings.shape}")
        return self
    
    def buscar_ongs_similares(self, consulta: str, top_k: int = 5, umbral: float = 0.1) -> List[Dict]:
        """Busca ONGs similares a la consulta"""
        if self.embeddings is None:
            raise ValueError("El sistema no ha sido ajustado")
        
        # Generar embedding de la consulta
        embedding_consulta = self.modelo.encode([consulta])
        
        # Calcular similitudes
        similitudes = cosine_similarity(embedding_consulta, self.embeddings)[0]
        indices = np.argsort(similitudes)[::-1]
        
        # Construir resultados
        resultados = []
        for idx in indices[:top_k]:
            if similitudes[idx] >= umbral:
                ong = self.df.iloc[idx]
                resultados.append({
                    'nombre': ong.get('nombre', 'Sin nombre'),
                    'categoria': ong.get('categoria', 'Sin categoría'),
                    'mision': str(ong.get('mision', ''))[:200] + '...' if len(str(ong.get('mision', ''))) > 200 else str(ong.get('mision', '')),
                    'servicios': str(ong.get('servicios', '')),
                    'ubicacion': str(ong.get('ubicacion', '')),
                    'contacto': str(ong.get('contacto', '')),
                    'similitud': float(similitudes[idx])
                })
        
        return resultados