from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import textwrap


SUMMARY_COLUMNS = [
    "experiment_name",
    "target",
    "model_name",
    "status",
    "n_min",
    "k_outer",
    "f1_macro_mean",
    "f1_macro_std",
    "balanced_accuracy_mean",
    "recall_macro_mean",
    "precision_macro_mean",
    "stability",
    "icn",
    "search_type",
    "search_scoring",
    "search_n_iter",
    "best_params_mode",
    "message",
]

PLAIN_TABLE_WIDTHS = [26, 16, 15, 15, 15, 15, 15, 28]


def write_summary_csv(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    rows = [item for results in results_by_target.values() for item in results]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in SUMMARY_COLUMNS})


def write_json_results(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(results_by_target, handle, ensure_ascii=False, indent=2)


def write_auxiliary_tables(results_by_target: dict[str, list[dict[str, Any]]], output_dirs: dict[str, Path]) -> None:
    distributions_path = output_dirs["tables"] / "distribucion_clases.csv"
    with distributions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target", "class", "support"])
        for target, results in results_by_target.items():
            distribution = results[0]["class_distribution"]
            for label, support in distribution.items():
                writer.writerow([target, label, support])

    for target, results in results_by_target.items():
        for item in results:
            if not item["implemented"]:
                continue
            _write_confusion_matrix(target, item, output_dirs["confusion_matrices"])
            _write_class_report(target, item, output_dirs["per_class"])


def write_warnings(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    lines: list[str] = []
    for target, results in results_by_target.items():
        for item in results:
            for warning in item.get("warnings", []):
                experiment_name = item.get("experiment_name", "experiment")
                lines.append(f"[{experiment_name} | {target} | {item['model_name']}] {warning}")
    if not lines:
        lines.append("No se registraron advertencias.")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_tables(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    lines = [
        r"\documentclass{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage[spanish]{babel}",
        r"\usepackage{booktabs}",
        r"\usepackage{geometry}",
        r"\geometry{margin=1.5cm, landscape}",
        r"\begin{document}",
        r"\section*{Laboratorio 04: resultados de clasificadores de ensamble}",
        (
            "Se implementan Bagging, AdaBoost, Stacking y Gradient Boosting usando el mismo protocolo "
            "de validaci\\'on cruzada anidada del Laboratorio 03."
        ),
        "",
    ]

    for target, results in results_by_target.items():
        distribution = _format_distribution(results[0]["class_distribution"])
        lines.extend(
            [
                rf"\subsection*{{Experimento {latex_escape(target)}}}",
                rf"\noindent\textbf{{Distribuci\'on de clases:}} {latex_escape(distribution)}. "
                rf"\textbf{{n\_min:}} {results[0]['n_min']}. "
                rf"\textbf{{k externo:}} {results[0]['k_outer']}.",
                r"\begin{table}[h]",
                r"\centering",
                rf"\caption{{Resultados para {latex_escape(target)}}}",
                r"\scriptsize",
                r"\begin{tabular}{p{3.2cm}p{2.0cm}p{1.6cm}p{1.6cm}p{1.8cm}p{1.5cm}p{1.4cm}p{3.4cm}}",
                r"\toprule",
                r"Experimento / modelo & F1 macro & Balanced Acc. & Recall macro & Precisi\'on macro & Estab. & ICN & Hiperpar\'ametros / estado \\",
                r"\midrule",
            ]
        )
        for item in results:
            lines.append(_latex_row(item))
        lines.extend(
            [
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
                "",
            ]
        )

    lines.extend([r"\end{document}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_pdf_tables(results_by_target: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    pages: list[list[str]] = []
    current: list[str] = [
        "Laboratorio 04: resultados de clasificadores de ensamble",
        "Bagging, AdaBoost, Stacking y Gradient Boosting evaluados con validación cruzada anidada.",
        "",
    ]

    for target, results in results_by_target.items():
        block = _plain_table_block(target, results)
        if len(current) + len(block) > 48:
            pages.append(current)
            current = []
        current.extend(block)
        current.append("")
    if current:
        pages.append(current)

    _write_simple_pdf(pages, output_path)


def _write_confusion_matrix(target: str, item: dict[str, Any], output_dir: Path) -> None:
    labels = item["labels"]
    matrix = item["confusion_matrix"]
    experiment_name = item.get("experiment_name", "experiment")
    path = output_dir / f"matriz_confusion_{experiment_name}_{target}_{item['model_key']}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["real/predicho", *labels])
        for label, row in zip(labels, matrix, strict=True):
            writer.writerow([label, *row])


def _write_class_report(target: str, item: dict[str, Any], output_dir: Path) -> None:
    report = item["classification_report"]
    experiment_name = item.get("experiment_name", "experiment")
    path = output_dir / f"metricas_por_clase_{experiment_name}_{target}_{item['model_key']}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clase", "precisión", "recall", "f1-score", "support"])
        for label in item["labels"]:
            row = report[str(label)]
            writer.writerow(
                [
                    label,
                    _format_float(row["precision"]),
                    _format_float(row["recall"]),
                    _format_float(row["f1-score"]),
                    int(row["support"]),
                ]
            )


def _latex_row(item: dict[str, Any]) -> str:
    model = latex_escape(f"{item.get('experiment_name', '')} / {item['model_name']}")
    if not item["implemented"]:
        status = latex_escape(item["status"])
        return (
            f"{model} & {status} & {status} & {status} & {status} & "
            f"{status} & {status} & {latex_escape(item['message'])} \\\\"
        )

    return (
        f"{model} & "
        f"{_format_mean_std(item['f1_macro_mean'], item['f1_macro_std'])} & "
        f"{_format_float(item['balanced_accuracy_mean'])} & "
        f"{_format_float(item['recall_macro_mean'])} & "
        f"{_format_float(item['precision_macro_mean'])} & "
        f"{_format_float(item['stability'])} & "
        f"{_format_float(item['icn'])} & "
        f"{latex_escape(item['best_params_mode'])} \\\\"
    )


def _plain_table_block(target: str, results: list[dict[str, Any]]) -> list[str]:
    distribution = _format_distribution(results[0]["class_distribution"])
    lines = [
        f"Experimento {target}",
        f"Distribución: {distribution} | n_min={results[0]['n_min']} | k_outer={results[0]['k_outer']}",
        "",
        _plain_row(["Experimento/modelo", "F1 macro", "BalAcc", "Recall", "Precisión", "Estab", "ICN", "Estado"], header=True),
        "-" * (sum(PLAIN_TABLE_WIDTHS) + len(PLAIN_TABLE_WIDTHS) - 1),
        "",
    ]
    for item in results:
        if item["implemented"]:
            values = [
                f"{item.get('experiment_name', '')} / {item['model_name']}",
                _format_mean_std(item["f1_macro_mean"], item["f1_macro_std"]),
                _format_float(item["balanced_accuracy_mean"]),
                _format_float(item["recall_macro_mean"]),
                _format_float(item["precision_macro_mean"]),
                _format_float(item["stability"]),
                _format_float(item["icn"]),
                item["best_params_mode"],
            ]
        else:
            values = [f"{item.get('experiment_name', '')} / {item['model_name']}", *["No implementado"] * 6, "No implementado"]
        lines.append(_plain_row(values))
        if item["implemented"]:
            for wrapped in textwrap.wrap(f"Hiperparámetros más frecuentes: {item['best_params_mode']}", width=120):
                lines.append(f"  {wrapped}")
    return lines


def _plain_row(values: list[str], header: bool = False) -> str:
    cells = []
    for value, width in zip(values, PLAIN_TABLE_WIDTHS, strict=True):
        text = str(value)
        if len(text) > width:
            text = text[: width - 3] + "..."
        cells.append(text.ljust(width))
    row = " ".join(cells)
    return row.upper() if header else row


def _write_simple_pdf(pages: list[list[str]], output_path: Path) -> None:
    objects: list[bytes] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    catalog_id = add_object(b"<< /Type /Catalog /Pages 2 0 R >>")
    pages_id = add_object(b"")
    font_id = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier /Encoding /WinAnsiEncoding >>")
    page_ids: list[int] = []

    for page_lines in pages:
        content = _pdf_content_stream(page_lines)
        content_id = add_object(
            b"<< /Length " + str(len(content)).encode("ascii") + b" >>\nstream\n" + content + b"\nendstream"
        )
        page_id = add_object(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 842 595] "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        page_ids.append(page_id)

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode("ascii")
    assert catalog_id == 1

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for idx, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{idx} 0 obj\n".encode("ascii"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(bytes(pdf))


def _pdf_content_stream(lines: list[str]) -> bytes:
    content = ["BT", "/F1 8 Tf", "40 555 Td", "10 TL"]
    for line in lines:
        content.append(f"({_pdf_escape(line)}) Tj")
        content.append("T*")
    content.append("ET")
    return "\n".join(content).encode("cp1252", errors="replace")


def _pdf_escape(text: str) -> str:
    safe = str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe.encode("cp1252", errors="replace").decode("cp1252")


def _format_distribution(distribution: dict[int, int]) -> str:
    return ", ".join(f"{label}: {support}" for label, support in distribution.items())


def _format_mean_std(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "No implementado"
    return f"{mean:.3f} +/- {std:.3f}"


def _format_float(value: float | None) -> str:
    if value is None:
        return "No implementado"
    return f"{value:.3f}"


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6f}"
    if value is None:
        return ""
    return value


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)
