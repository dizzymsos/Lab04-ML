# Lab04-ML-2026-01

Implementación completa del Laboratorio 04: clasificadores de ensamble sobre el dataset `.sav` de 15 atributos binarios.

## Alcance

- Se replica la estrategia experimental del Lab03 para los seis objetivos: `GDS`, `GDS_R1`, `GDS_R2`, `GDS_R3`, `GDS_R4` y `GDS_R5`.
- Se mantiene validación cruzada anidada para comparar modelos de forma justa.
- Se implementan cuatro clasificadores de ensamble:
  - `BaggingClassifier` con árboles de decisión.
  - `AdaBoostClassifier` con árboles débiles.
  - `RobustStackingClassifier` con árbol, K-NN y regresión logística como modelos base.
  - `GradientBoostingClassifier`.
- Se ejecutan dos experimentos de búsqueda:
  - `grid_all`: todos los clasificadores con `GridSearchCV`.
  - `random_all`: todos los clasificadores con `RandomizedSearchCV`.
- La configuración de hiperparámetros está fuera del código, en `config/paths.yaml`.
- `RandomizedSearchCV` usa distribuciones explícitas de `scipy.stats` definidas en el YAML.

## Resultados principales

| Objetivo | Mejor modelo | Búsqueda | F1 macro |
|---|---|---|---|
| GDS | Stacking | grid | 0.364 |
| GDS_R1 | Gradient Boosting | random | 0.727 |
| GDS_R2 | Gradient Boosting | grid | 0.673 |
| GDS_R3 | AdaBoost | random | 0.803 |
| GDS_R4 | Bagging | random | 0.609 |
| GDS_R5 | Bagging | random | 0.556 |

GDS presenta el desbalance más extremo (clase minoritaria con 2 ejemplos, k_outer=2) y debe interpretarse como evidencia exploratoria.

## Crear ambiente conda

```bash
conda env create -f environment.yml
conda activate lab04_ml_2026_01
```

## Ejecutar experimentos

Ejecutar todos los objetivos y modelos:

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

## Salidas generadas

- `outputs/tables/resultados_experimentos.tex`: tablas en LaTeX por experimento.
- `outputs/tables/resultados_experimentos.pdf`: tablas en PDF.
- `outputs/tables/resumen_resultados.csv`: resumen tabular de todos los modelos.
- `outputs/tables/resultados_detallados.json`: resultados completos.
- `outputs/tables/distribucion_clases.csv`: soporte por clase.
- `outputs/confusion_matrices/`: matrices de confusión agregadas por modelo y experimento.
- `outputs/per_class/`: precisión, recall y F1 por clase.
- `outputs/advertencias.txt`: advertencias metodológicas de validación.

## Nota metodológica

Para cada objetivo se usa `k_outer = min(5, n_min)`, donde `n_min` es el soporte de la clase menos frecuente. En el ciclo interno se usa una grilla reducida o distribuciones aleatorias según el experimento (`grid_all` o `random_all`), ambas definidas en `config/paths.yaml`.

Cuando una clase queda con solo un ejemplo dentro del entrenamiento externo, el código usa `KFold` interno no estratificado y registra la advertencia en `outputs/advertencias.txt`. Esto ocurre principalmente en el experimento `GDS`.