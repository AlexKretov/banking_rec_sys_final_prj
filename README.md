# Рекомендательная система на T-ECD

Проект адаптирован для **[T-ECD (T-Tech E-commerce Cross-Domain Dataset)](https://huggingface.co/datasets/t-tech/T-ECD)**. Сохранена исходная архитектура: анализ данных → implicit ALS → модель второго уровня Random Forest → FastAPI → Prometheus/Grafana. `eda.ipynb` расширен аудитом качества данных и протокола моделирования; ограничения текущего прототипа описаны ниже и в ноутбуке.

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

Артефакты создаются в `fastaip/ml_service/artifacts/`. В текущем training-reader лимит уменьшает итоговый DataFrame, но отдельная Parquet-партиция сначала читается целиком: `max_rows` не гарантирует защиту от исчерпания RAM и полноту временного окна. Перед содержательным экспериментом учтите ограничения модели ниже; EDA использует отдельное батчевое чтение.

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
2. `eda.ipynb` — расширенный EDA: покрытие и схемы Parquet, смещение выборки, пропуски, действия, время/дрейф, long tail, повторы, каталоги, cross-domain overlap, embeddings, числовые аномалии и cold-start; подробные выводы и backlog моделирования.
3. `modeling.ipynb` — обучение сохранённой двухуровневой модели.
4. `test.ipynb` — проверка и нагрузочный тест API новым контрактом.

## Расширенный EDA

Для анализа не нужно устанавливать ALS/MLflow и запускать API:

```bash
pip install -r requirements-eda.txt
jupyter nbconvert --to notebook --execute --inplace eda.ipynb --ExecutePreprocessor.timeout=600
python -m pytest -q tests
```

В интерактивном Jupyter после изменения параметров используйте **Restart Kernel → Run All**. Параметры находятся в начале ноутбука; для headless-запуска поддерживаются переменные окружения:

```bash
TECD_EDA_DATA_DIR=data/t_ecd TECD_EDA_DAY_BEGIN=1250 TECD_EDA_DAY_END=1308 \
TECD_EDA_MAX_ROWS=1000000 TECD_EDA_REPORT_DIR=data/eda_reports/run_01 \
jupyter nbconvert --to notebook --execute --inplace eda.ipynb --ExecutePreprocessor.timeout=600
```

Также доступны `TECD_EDA_VARIANT=small|full`, `TECD_EDA_DOMAINS` (через запятую), `TECD_EDA_SEED`. Small по умолчанию включает Marketplace, Retail, Offers и Reviews; Payments/receipts анализируются отдельно только для Full. Корень может содержать непосредственно домены или вложенную структуру `dataset/small`. Файлы не скачиваются автоматически. Для временного разбиения по умолчанию нужно хотя бы **7 последовательных дней**, для содержательных трендов желательно несколько недель. Глубина истории и полнота выгрузки проверяются по каждому домену.

### Достоверность результатов

- **CARD** — опубликованные сведения авторов T‑ECD; источник и редакция закреплены в ноутбуке.
- **SAVED** — агрегаты из outputs прежнего `eda.ipynb` на коммите `6f6b122c245f24be9a313c1491b753359f449abb`, сохранённые с provenance в `eda_reference.json`. Это **не новый запуск**: прежний лимит 2 млн строк оставил лишь день 1250 и Marketplace/Offers. 96,84% этих событий — просмотры; 20 078 пользователей встречаются в обоих наблюдаемых доменах.
- **LOCAL** — новые вычисления только по фактически доступным Parquet. Если данных нет, ноутбук выполняется до конца и отмечает raw-проверки как **«не выполнено»**, не заменяя данные синтетическими. В сохранённом исполнении этой работы raw-файлов нет; новые распределения/частоты аномалий не заявлены.

`tecd_eda.py` читает footer и делает воспроизводимую случайную выборку **из всей длины каждой доступной партиции**, а не префикс. Обратные вероятности отбора используются только для аддитивных event-частот; unique/overlap/Gini/cold-start относятся к наблюдаемому sample. Для полных цепочек нужен census последовательного окна либо стабильная user-cohort выборка. Каталоги сканируются целиком батчами для lookup выбранных ID; неоднозначные ключи не размножают события. Бюджет ограничивает память, но I/O зависит от всего выбранного окна — не направляйте локальный pandas EDA на Full целиком.

Числовой timestamp требует явных `TIME_CONFIG` unit/origin; `DAY_ZERO`, `PREDICTION_TIME` и шкала rating не угадываются. Агрегаты, параметры, manifest и реестр выводов экспортируются в игнорируемый `data/eda_reports/latest/` (перезаписывается) либо заданный каталог. Manifest fingerprint не является hash содержимого Parquet: HF revision загрузки нужно фиксировать отдельно. Сырые события в Git не добавляются. Тестовые synthetic fixtures проверяют код, **не являются статистикой T‑ECD** и не перезаписывают outputs ноутбука.

## Методика и ограничения текущей модели

Исходный `tecd_pipeline.py` делит события по дням, но **пока не реализует обязательный минимум 12 часов между историей и prediction timestamp**. EDA показывает схему `train | gap | validation | gap | test` с целыми gap-днями и отдельную clock-based проверку. Правило должно соблюдаться во всех доменах и при расчёте исторических признаков, а не только на самом конце датасета.

Аудит выявил другие P0 ограничения прототипа: `max_rows` обрывает ранние партиции, loader отбрасывает `timestamp`, RF обучается и получает `classification_report` на одних и тех же строках, API приводит строковые item ID к `int`, а параметр `domain` не ограничивает candidates. **Эта работа расширяет EDA, но не изменяет обучающий код и API.** До содержательного эксперимента выполните backlog из раздела 12 ноутбука. Текущий RF report — resubstitution, не честная out-of-time оценка.

На T-ECD задача — implicit-feedback recommendation с неодинаковой семантикой действий и доменов. Accuracy/ROC-AUC недостаточно для оценки retrieval: нужны Recall@K/NDCG@K, domain/user-macro метрики, coverage и cold-start срезы. Метрики в `metadata.json` зависят от фактически скачанного окна и протокола. Репозиторий не заявляет воспроизведённых результатов на полном T-ECD; прежние ROC-AUC 0.969 и выводы о банковских продуктах к нему не относятся.

## Лицензия данных

T-ECD распространяется авторами по **CC BY-NC-SA 4.0**. При публикации результатов используйте цитирование из dataset card и учитывайте некоммерческое ограничение лицензии.
