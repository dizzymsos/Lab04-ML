# Lab04-ML-2026-01

Implementación base del Laboratorio 04: clasificadores de ensamble sobre el dataset `.sav` de 15 atributos binarios.

## Alcance

- Se replica la estrategia experimental del Lab03 para los seis objetivos: `GDS`, `GDS_R1`, `GDS_R2`, `GDS_R3`, `GDS_R4` y `GDS_R5`.
- Se mantiene validación cruzada anidada para comparar modelos de forma justa.
- Se implementan cuatro clasificadores de ensamble:
  - `BaggingClassifier` con árboles de decisión.
  - `AdaBoostClassifier` con árboles débiles.
  - `RobustStackingClassifier` con árbol, K-NN y regresión logística como modelos base.
  - `GradientBoostingClassifier`.
- La configuración de hiperparámetros está fuera del código, en `config/paths.yaml`.
- Se ejecutan dos experimentos de búsqueda:
  - `grid_all`: todos los clasificadores con `GridSearchCV`.
  - `random_all`: todos los clasificadores con `RandomizedSearchCV`.
- `RandomizedSearchCV` usa distribuciones explícitas desde YAML.

## Idea central del laboratorio

Cada modelo se evalúa con el mismo protocolo:

1. Se toma un objetivo, por ejemplo `GDS_R3`.
2. Se separan los datos en folds externos.
3. Dentro de cada entrenamiento externo se buscan hiperparámetros.
4. El mejor modelo interno se evalúa en el fold externo.
5. Al final se reportan promedios, desviaciones estándar, matrices de confusión y métricas por clase.

Esto se llama **validación cruzada anidada**. La parte interna selecciona hiperparámetros; la parte externa estima rendimiento.

## Crear ambiente conda

```bash
conda env create -f environment.yml
conda activate lab04_ml_2026_01
```

Si se quiere reutilizar el ambiente del Lab03:

```bash
conda activate lab03_ml_2026_01
```

## Ejecutar experimentos

Ejecutar todos los objetivos y modelos configurados:

```bash
python main.py
```

Ejecutar solo algunos objetivos:

```bash
python main.py --targets GDS_R2 GDS_R5
```

Ejecutar solo algunos modelos:

```bash
python main.py --models bagging gradient_boosting
```

Ejecutar solo un tipo de experimento:

```bash
python main.py --experiments grid_all
python main.py --experiments random_all
```

Combinar ambas opciones:

```bash
python main.py --targets GDS_R3 --models adaboost gradient_boosting --experiments random_all
```

## Configuración principal

La configuración está en:

```text
config/paths.yaml
```

Ahí se definen:

- ruta del dataset;
- carpetas de salida;
- objetivos a ejecutar;
- experimentos a ejecutar;
- número de folds;
- métrica principal;
- semillas aleatorias;
- búsqueda de hiperparámetros.

## Semillas aleatorias

Las semillas se centralizan para que los experimentos sean reproducibles y comparables:

```yaml
experiment:
  random_state:
    global: 42
    outer_cv: 42
    inner_cv: 123
    model: 42
    search: 42
```

Lectura didáctica:

- `outer_cv`: controla las particiones externas de evaluación.
- `inner_cv`: controla las particiones internas de selección de hiperparámetros.
- `model`: controla modelos con aleatoriedad, como Bagging, AdaBoost, Stacking y Gradient Boosting.
- `search`: controla `RandomizedSearchCV`.
- `global`: valor de respaldo si alguna semilla específica no está definida.

## Experimentos configurados

El orden de ejecución se configura así:

```yaml
experiment:
  experiments:
    - name: "grid_all"
      search_type: "grid"
      models:
        - "bagging"
        - "adaboost"
        - "stacking"
        - "gradient_boosting"

    - name: "random_all"
      search_type: "random"
      models:
        - "bagging"
        - "adaboost"
        - "stacking"
        - "gradient_boosting"
```

Esto significa que `python main.py` ejecuta primero todas las búsquedas exhaustivas con `GridSearchCV` y después todas las búsquedas aleatorias con `RandomizedSearchCV`.

Si se quiere desactivar un modelo sin borrar su configuración, se hace dentro del bloque correspondiente:

```yaml
hyperparameter_search:
  grid:
    models:
      stacking:
        enabled: false
```

## Hiperparámetros fuera del código

Cada modelo tiene dos posibles configuraciones bajo `hyperparameter_search`:

- `grid`: grillas finitas para `GridSearchCV`.
- `random`: distribuciones para `RandomizedSearchCV`.

Ejemplo con `GridSearchCV`:

```yaml
hyperparameter_search:
  grid:
    models:
      bagging:
        enabled: true
        params:
          clf__n_estimators: [25, 50]
          clf__max_samples: [0.7, 1.0]
          clf__estimator__max_depth: [2, null]
```

`GridSearchCV` prueba todas las combinaciones posibles. En este ejemplo:

- 2 valores para `n_estimators`;
- 2 valores para `max_samples`;
- 2 valores para `max_depth`;
- total: 2 x 2 x 2 = 8 combinaciones.

Ejemplo con `RandomizedSearchCV`:

```yaml
hyperparameter_search:
  random:
    models:
      adaboost:
        enabled: true
        distributions:
          clf__n_estimators:
            dist: "randint"
            low: 25
            high: 151
          clf__learning_rate:
            dist: "uniform"
            low: 0.5
            high: 1.0
          clf__estimator__max_depth:
            values: [2, 3]
```

`RandomizedSearchCV` no prueba todas las combinaciones. Toma `n_iter` muestras desde distribuciones, usando la semilla `search`.

Tipos soportados en el YAML:

- `randint`: enteros aleatorios. `high` es exclusivo, igual que en `scipy.stats.randint`.
- `uniform`: valores reales uniformes entre `low` y `high`.
- `loguniform`: valores reales positivos en escala logarítmica.
- `values`: lista discreta de candidatos, útil para opciones como `[1, 2, 3]` o `[2, 3, null]`.

## Métrica de selección

La métrica principal es:

```yaml
scoring: "f1_macro"
```

Se usa `f1_macro` porque el dataset tiene clases desbalanceadas. Esta métrica calcula el F1 por clase y luego promedia, evitando que la clase mayoritaria domine toda la evaluación.

## ICN

El ICN es un índice comparativo normalizado. En esta versión se calcula por objetivo comparando cada corrida completa:

```text
experimento + modelo
```

Por ejemplo, `grid_all / Bagging` y `random_all / Bagging` son corridas distintas. Esto permite comparar no solo clasificadores, sino también la estrategia de búsqueda de hiperparámetros.

## Gradient Boosting

El modelo nuevo se configura igual que los demás:

```yaml
gradient_boosting:
  enabled: true
  params:
    clf__n_estimators: [50, 100]
    clf__learning_rate: [0.05, 0.1]
    clf__max_depth: [2, 3]
    clf__subsample: [0.8, 1.0]
```

Este modelo entrena árboles de forma secuencial. Cada nuevo árbol intenta corregir errores del conjunto anterior.

## Salidas generadas

- `outputs/tables/resultados_experimentos.tex`: tablas en LaTeX por experimento.
- `outputs/tables/resultados_experimentos.pdf`: tablas en PDF simple.
- `outputs/tables/resumen_resultados.csv`: resumen tabular de todos los modelos.
- `outputs/tables/resultados_detallados.json`: resultados completos.
- `outputs/tables/distribucion_clases.csv`: soporte por clase.
- `outputs/confusion_matrices/`: matrices de confusión agregadas por modelo.
- `outputs/per_class/`: precisión, recall y F1 por clase.
- `outputs/advertencias.txt`: advertencias metodológicas de validación.

El archivo `resumen_resultados.csv` incluye, entre otras columnas:

- modelo;
- objetivo;
- experimento (`grid_all` o `random_all`);
- F1 macro;
- balanced accuracy;
- estabilidad;
- ICN;
- tipo de búsqueda (`grid` o `random`);
- métrica usada para seleccionar hiperparámetros;
- `n_iter`, si se usó `RandomizedSearchCV`;
- hiperparámetros más frecuentes.

## Nota metodológica

Para cada objetivo se usa:

```text
k_outer = min(5, n_min)
```

donde `n_min` es el soporte de la clase menos frecuente. Esto evita pedir más folds externos que ejemplos disponibles en la clase minoritaria.

Cuando una clase queda con solo un ejemplo dentro del entrenamiento externo, no existe una partición interna estratificada válida para esa clase. En ese caso el código usa un `KFold` interno no estratificado y deja la advertencia en `outputs/advertencias.txt`.

El stacking implementado usa predicciones out-of-fold cuando la distribución de clases lo permite. Si el soporte mínimo dentro de un entrenamiento es menor que 2, usa predicciones in-sample para entrenar el meta-clasificador y conserva la evaluación externa para estimar generalización.
