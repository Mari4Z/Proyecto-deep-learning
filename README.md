# Proyecto integrador de Deep Learning: Entrega final

Auditoría de un modelo BETO para detectar discurso racista en español.

**Integrantes:** María José Cortés Romero y María Isabell Guzmán Faneyte. Universidad Externado de Colombia.

Esta entrega (Fase 3: auditoría y comunicación) parte de la Fase 2: https://github.com/Mari4Z/deep-learning-fase2

## Contenido

| Archivo o carpeta | Descripción |
|---|---|
| `cuaderno fase 3.ipynb` | Cuaderno ejecutado: 8.1 modelo final, 8.2 auditoría con matrices de atención, 8.3 análisis de fallas, 8.4 dictamen y 8.5 limitaciones |
| `entrenar_modelo_final.py` | Reentrena el modelo final (BETO uncased, ajuste completo, semilla 1) y guarda sus pesos |
| `requirements.txt` | Versiones exactas de las librerías |
| `insumos/` | Resultados de la Fase 2 que usa el cuaderno: partición congelada, configuración elegida y corridas |
| `salidas/` | Resultados de esta entrega: tablas en CSV, figuras, predicciones y la revisión manual de errores |

## Cómo reproducirlo

1. Crear un entorno con Python 3.12 e instalar `pip install -r requirements.txt`. En Mac con chip Apple se usa la GPU con `tensorflow-metal`.
2. Copiar a `insumos/` los dos archivos de datos que no se incluyen en el repositorio:
   - `datos_local.json`: entrenamiento y validación del corpus local. Lo genera el cuaderno de la Fase 2 (sección 7.2) en `salidas_71/datos_local.json`.
   - `corpus_articulo.csv`: el archivo `data/input_tweets.csv` del repositorio de los autores (González-Jiménez & Benítez-Andrades, 2022), en Zenodo: https://doi.org/10.5281/zenodo.5886624
3. Ejecutar el cuaderno desde la carpeta del proyecto. El corpus local se descarga de Hugging Face (`piuba-bigdata/contextualized_hate_speech`). El reentrenamiento de la sección 8.1 tarda unos 30 minutos en GPU y guarda los pesos en `salidas/modelo_final/`; si ya existen, no se vuelve a entrenar.

## Datos

- Corpus local: *Contextualized Hate Speech* (Pérez et al., 2023), Hugging Face.
- Corpus del artículo: Benítez-Andrades et al. (2022), Zenodo.

Los datos no se redistribuyen en este repositorio. `salidas/revision_manual_fallas.csv` y algunas figuras citan textualmente algunos tuits del corpus para el análisis.

## Uso de inteligencia artificial generativa

Se utilizó Claude (Anthropic) como apoyo en la revisión del código, la preparación del cuaderno y la redacción del informe. Las integrantes revisaron los resultados y pueden sustentar cada decisión técnica.
