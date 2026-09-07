# Развёртывание

Каталог содержит Docker Compose для существующего контура FastAPI + Prometheus + Grafana. Перед сборкой обучите модель T-ECD из корня репозитория: артефакты должны находиться в `fastaip/ml_service/artifacts/`.

Создайте `fastaip/.env`:

```dotenv
AUTHOR=project
THE_HOST=0.0.0.0
THE_PORT=8079
VM_PORT=8079
PROMETHEUS_PORT=9090
GRAFATA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASS=admin
```

Затем выполните `docker compose up --build`. API доступен на порту 8079, Prometheus — 9090, Grafana — 3000. Конфигурация сбора метрик находится в `prometheus/prometheus.yml`.
