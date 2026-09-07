# Рекомендательная система на T-ECD

Проект адаптирован для **[T-ECD (T-Tech E-commerce Cross-Domain Dataset)](https://huggingface.co/datasets/t-tech/T-ECD)**. Сохранена исходная архитектура: анализ данных → implicit ALS → модель второго уровня Random Forest → FastAPI → Prometheus/Grafana. Новый функционал не добавлялся.

## Данные

T-ECD содержит около 135 млрд событий, 44 млн пользователей и 30 млн товаров в пяти доменах. Для локального запуска используйте **T-ECD Small** либо ограниченный диапазон дней. Код модели использует товарные события Marketplace, Retail и Offers; Payments и Reviews не имеют единого `item_id` во всех событиях и не подходят к исходной item-рекомендательной постановке без изменения функциональности.

> Не коммитьте датасет: полный архив занимает 2.81 ТБ. Путь `data/` исключён из Git.

Скачайте официальный `tecd_downloader.py` со страницы датасета, задайте `HF_TOKEN` и загрузите несколько последовательных дней (минимум три):

```python
from tecd_downloader import download_dataset
import os

download_dataset(
    token=os.environ["HF_TOKEN"],
    dataset_path="dataset/small",
    local_dir="data/t_ecd",
    domains=["retail", "marketplace", "offers"],
    day_begin=1250,
    day_end=1308,
    max_workers=10,
)
```

Структура внутри `local_dir` сохраняется как `dataset/small/...`; загрузчик проекта автоматически находит её.

## Установка и запуск

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python tecd_pipeline.py data/t_ecd --max-rows 2000000
```

Артефакты создаются в `fastaip/ml_service/artifacts/`. Ограничение строк защищает pandas-пайплайн от исчерпания памяти; для содержательного эксперимента выбирайте последовательный диапазон дней и увеличивайте лимит с учётом RAM.

Запуск сервиса:

```bash
cd fastaip/ml_service
uvicorn app2:app --host 0.0.0.0 --port 8079
```

Пример запроса (используйте `user_id` из обучающей выборки):

```bash
curl -X POST http://localhost:8079/predict \
  -H 'Content-Type: application/json' \
  -d '{"user_id": 123, "top_k": 5, "domain": "marketplace"}'
```

`GET /health` показывает наличие модели, `/metrics` отдаёт метрики Prometheus. Docker Compose в `fastaip/` запускает сервис, Prometheus и Grafana.

## Тетради

1. `loader.ipynb` — выборочная загрузка T-ECD с Hugging Face.
2. `eda.ipynb` — EDA дневных Parquet-партиций и каталогов.
3. `modeling.ipynb` — обучение сохранённой двухуровневой модели.
4. `test.ipynb` — проверка и нагрузочный тест API новым контрактом.

## Методика и выводы

Разбиение выполняется **по дням**, а не случайно: последние дни оставляются для проверки. При подготовке prediction timestamp необходимо также соблюдать опубликованное авторами T-ECD правило 12-часового зазора против утечки. Метрики записываются в `metadata.json` только после запуска на фактически скачанной выборке; прежние ROC-AUC 0.969 и выводы о банковских продуктах удалены, поскольку они не относятся к T-ECD.

На T-ECD задача стала implicit-feedback item recommendation. Поэтому accuracy/ROC-AUC сами по себе недостаточны для оценки ALS; результаты зависят от доменов, диапазона дней и лимита строк. Репозиторий не заявляет заранее вычисленных результатов на полном T-ECD. Для воспроизводимых выводов фиксируйте эти параметры вместе с `metadata.json`.

## Лицензия данных

T-ECD распространяется авторами по **CC BY-NC-SA 4.0**. При публикации результатов используйте цитирование из dataset card и учитывайте некоммерческое ограничение лицензии.
