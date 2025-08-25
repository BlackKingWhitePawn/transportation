import click
import src.dataOps as dataOps
import src.writer as writer
import logging
import os
from tqdm import tqdm
from pathlib import Path
import shutil
import re
from datetime import datetime, date, time
from dateutil.parser import parse

logger = logging.getLogger(__name__)

# ДОБАВЛЯЕМ настройки ширины терминала
CONTEXT_SETTINGS = {"max_content_width": shutil.get_terminal_size().columns}


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    logging.basicConfig(filename="log.txt", level=logging.INFO)
    pass


@main.command("data-operations")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format",
    type=click.Choice(["cd", "xg"]),
    default="cd",
    show_default=True,
    help=(
        '"xg" — сырые данные из иксолоджи (CSV);\n'
        '"cd" — экспорт из ЦУСАД (первый лист Excel).'
    ),
)
@click.option(
    "-o",
    "--out-dir",
    "out_base",  # <= сразу в out_base
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Куда сохранять результат. Если не указано — сохраняем рядом с исходником.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Перезаписывать существующие файлы назначения.",
)
@click.option(
    "--suffix",
    default="_processed",
    show_default=True,
    help="Суффикс имени файла, когда --out-dir не задан.",
)
def data_operations(
    path: Path, format: str, out_base: Path | None, overwrite: bool, suffix: str
):
    """
    Обрабатывает файл/директорию PATH рекурсивно.
    Сохраняет структуру подпапок при использовании --out-dir.
    """
    # дальше используйте out_base внутри своей обработки:
    #   - если out_base задан: dst = out_base / src.relative_to(path)
    #   - иначе: dst = src.with_name(src.stem + suffix + src.suffix)
    #   - учитывайте overwrite при записи
    #
    # ваш существующий код обработки здесь ⤵
    ...
    extensions = {".xls", ".xlsx"} if (format == "cd") else {".csv"}
    path_object = Path(path)

    def is_datetime(value) -> bool:
        if isinstance(value, (datetime, date)):
            return True
        try:
            parse(str(value), fuzzy=False)
            return True
        except Exception:
            return False

    def process_file(src_path: Path):
        if src_path.name.startswith("~$"):
            return

        if format == "xg":
            df = dataOps.process_csv_dataframe(src_path)
            # пути сохранения
            if out_base:
                dst_path = out_base / src_path.relative_to(path_object)
            else:
                dst_path = src_path.with_name(src_path.stem + "_processed.csv")
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(dst_path, index=False, encoding="utf-8-sig")

        elif format == "cd":
            print(f"Обработка {src_path}")
            initial_df, predicted = dataOps.process_xslx_dataframe(src_path)

            # КУДА ПИСАТЬ: формируем ПУТЬ НОВОГО ФАЙЛА
            if out_base:
                dst_path = out_base / src_path.relative_to(path_object)
            else:
                dst_path = src_path.with_name(
                    src_path.stem + "_processed" + src_path.suffix
                )

            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # write_to_predict_sheet САМ СОЗДАЁТ НОВЫЙ ФАЙЛ ПО ЭТОМУ ПУТИ
            print(f"Запись в новый файл {dst_path}")
            writer.write_to_predict_sheet(dst_path, initial_df, predicted)

    # собрать файлы рекурсивно
    files = []
    if path_object.is_dir():
        files = [
            f
            for f in path_object.rglob("*")
            if f.is_file() and f.suffix.lower() in extensions
        ]
    elif path_object.is_file() and path_object.suffix.lower() in extensions:
        files = [path_object]

    for filePath in tqdm(files, desc="Обработка файлов"):
        try:
            process_file(filePath)
        except BaseException as e:
            print(f"Ошибка при обработке {filePath}: {e}")


def check_missing():
    pass


if __name__ == "__main__":
    main()
