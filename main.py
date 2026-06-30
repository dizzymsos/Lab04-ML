from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from src.data_loader import TARGETS, load_sav_dataset, prepare_xy
from src.evaluation import assign_icn, compute_outer_folds, run_nested_cv, unimplemented_result
from src.models import build_model_registry
from src.reports import (
    write_auxiliary_tables,
    write_json_results,
    write_latex_tables,
    write_pdf_tables,
    write_summary_csv,
    write_warnings,
)
from src.settings import DEFAULT_CONFIG_PATH, ensure_output_dirs, load_config


def parse_args() -> ArgumentParser:
    # El laboratorio se puede ejecutar completo o sobre un subconjunto de
    # objetivos. Los argumentos permiten cambiar esa decisión sin modificar
    # el código fuente.
    parser = ArgumentParser(description="Ejecuta el Laboratorio 04 con clasificadores de ensamble.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Ruta al archivo YAML de configuración.")
    parser.add_argument(
        "--targets",
        nargs="*",
        default=None,
        help="Objetivos a ejecutar. Por defecto usa los seis objetivos del PDF.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Modelos a ejecutar. Por defecto usa el orden definido en config/paths.yaml.",
    )
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Experimentos a ejecutar. Por defecto usa grid_all y luego random_all.",
    )
    return parser.parse_args()


def main() -> None:
    # 1) Cargar configuración y asegurar que existan las carpetas de salida.
    #    El YAML concentra rutas, objetivos, semillas, número de folds y
    #    paralelismo para que la lógica experimental quede reproducible.
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    experiment_config = config["experiment"]
    validation_config = experiment_config["validation"]
    random_state_config = experiment_config["random_state"]
    hyperparameter_config = config["hyperparameter_search"]

    # 2) Definir qué objetivos, experimentos y modelos se evaluarán. El YAML
    #    ordena primero todas las búsquedas grid y luego todas las random.
    target_names = args.targets or experiment_config.get("targets", TARGETS)
    model_registry = build_model_registry(random_state=_random_state(random_state_config, "model"))
    experiment_runs = _selected_experiments(
        requested_experiments=args.experiments,
        configured_experiments=experiment_config["experiments"],
    )

    # 3) Leer una sola vez el dataset SPSS. Luego, para cada objetivo, se
    #    construyen X e y con prepare_xy().
    df = load_sav_dataset(config["dataset"]["path"])
    results_by_target = {target_name: [] for target_name in target_names}

    for experiment_run in experiment_runs:
        experiment_name = experiment_run["name"]
        search_type = experiment_run["search_type"]
        model_order = _selected_models(
            requested_models=args.models or experiment_run["models"],
            model_registry=model_registry,
            search_models=hyperparameter_config[search_type]["models"],
        )

        for target_name in target_names:
            # 4) Preparar el problema supervisado para el objetivo actual.
            #    k_outer depende del soporte de la clase minoritaria para evitar
            #    particiones estratificadas inválidas.
            X, y = prepare_xy(df, target_name)
            n_min, k_outer = compute_outer_folds(y, validation_config["max_outer_folds"])
            distribution = {int(label): int(count) for label, count in y.value_counts().sort_index().items()}

            for model_key in model_order:
                # 5) Ejecutar validación cruzada anidada para cada ensamble.
                #    El ciclo interno selecciona hiperparámetros con f1_macro y el
                #    ciclo externo estima el desempeño del modelo seleccionado.
                spec = model_registry[model_key]
                if spec.implemented:
                    search_config = _search_config_for_model(
                        model_key=model_key,
                        search_type=search_type,
                        experiment_run=experiment_run,
                        hyperparameter_config=hyperparameter_config,
                        validation_config=validation_config,
                    )
                    result = run_nested_cv(
                        X,
                        y,
                        target_name,
                        spec,
                        validation_config,
                        search_config,
                        random_state_config,
                        experiment_name=experiment_name,
                    )
                else:
                    result = unimplemented_result(
                        target_name,
                        spec,
                        distribution,
                        n_min,
                        k_outer,
                        experiment_name=experiment_name,
                    )
                results_by_target[target_name].append(result)

    # 6) Calcular el ICN después de evaluar todos los experimentos y modelos
    #    del objetivo, porque este índice normaliza métricas comparando entre
    #    ensambles y entre estrategias de búsqueda.
    for target_results in results_by_target.values():
        assign_icn(target_results)

    # 7) Resolver rutas de salida y exportar resultados en formatos útiles
    #    para revisión: CSV/JSON para análisis, LaTeX/PDF para reporte, y
    #    archivos auxiliares para matrices de confusión y métricas por clase.
    output_dirs = {name: Path(path) for name, path in config["outputs"].items()}
    tables_dir = output_dirs["tables"]

    write_summary_csv(results_by_target, tables_dir / "resumen_resultados.csv")
    write_json_results(results_by_target, tables_dir / "resultados_detallados.json")
    write_auxiliary_tables(results_by_target, output_dirs)
    write_warnings(results_by_target, output_dirs["root"] / "advertencias.txt")
    write_latex_tables(results_by_target, tables_dir / "resultados_experimentos.tex")
    write_pdf_tables(results_by_target, tables_dir / "resultados_experimentos.pdf")

    print("Experimentos finalizados.")
    print(f"Tabla LaTeX: {tables_dir / 'resultados_experimentos.tex'}")
    print(f"Tabla PDF:   {tables_dir / 'resultados_experimentos.pdf'}")


def _selected_experiments(
    requested_experiments: list[str] | None,
    configured_experiments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not requested_experiments:
        return configured_experiments

    by_name = {experiment["name"]: experiment for experiment in configured_experiments}
    selected = []
    for experiment_name in requested_experiments:
        if experiment_name not in by_name:
            available = ", ".join(sorted(by_name))
            raise ValueError(f"Experimento no reconocido: {experiment_name}. Disponibles: {available}.")
        selected.append(by_name[experiment_name])
    return selected


def _selected_models(
    requested_models: list[str],
    model_registry: dict[str, Any],
    search_models: dict[str, Any],
) -> list[str]:
    selected = []
    for model_key in requested_models:
        if model_key not in model_registry:
            available = ", ".join(sorted(model_registry))
            raise ValueError(f"Modelo no reconocido: {model_key}. Modelos disponibles: {available}.")
        if model_key not in search_models:
            raise ValueError(f"Falta configuración de hiperparámetros para el modelo {model_key}.")
        if search_models[model_key].get("enabled", True):
            selected.append(model_key)

    if not selected:
        raise ValueError("No hay modelos habilitados para ejecutar.")
    return selected


def _search_config_for_model(
    model_key: str,
    search_type: str,
    experiment_run: dict[str, Any],
    hyperparameter_config: dict[str, Any],
    validation_config: dict[str, Any],
) -> dict[str, Any]:
    default_config = hyperparameter_config.get("default", {})
    model_config = hyperparameter_config[search_type]["models"][model_key]
    merged = {**default_config, **experiment_run, **model_config}
    merged["type"] = search_type
    merged["scoring"] = merged.get("scoring", validation_config.get("scoring", "f1_macro"))
    if search_type == "random":
        merged["n_iter"] = int(merged.get("n_iter", default_config.get("random_n_iter", 10)))
    return merged


def _random_state(random_state_config: dict[str, int], key: str) -> int:
    if key in random_state_config:
        return int(random_state_config[key])
    return int(random_state_config.get("global", 42))


if __name__ == "__main__":
    main()
