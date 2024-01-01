#FastAPI Application for Satelite Images classification (Pet Project)

### Ссылка на сервис (временная)

[Ссылка](http://91.206.15.25:1767/docs)


### Описание API, примеры запросов

Cервис поднят с помощью приложения FastAPI

Реализованы следующие запросы:

`'POST' '/barcode/predict'`  - детекция и распознавание баркодов на фото, принимает на вход несколько изображений

`'GET' '/barcode/health_check'`  - Проверка работает ли сервис


### Как развернуть сервис локально питоном 

* Сначала создать и активировать venv:
  
```bash
python3 -m venv venv
. venv/bin/activate
```
* Поставить зависимости:
```bash
make install
```

* Cкачать веса моделек
```bash
make download_weights
```

* Запуск сервиса
```bash
make run_app
```
Можно с аргументом `APP_PORT`


### Как развернуть сервис локально докером

* `make build` - собрать образ. Можно с аргументами `DOCKER_TAG`, `DOCKER_IMAGE`

* `make docker_run` - запустить контейнер с приложением


### Где искать докер-образы сервиса и как сбилдить свежий образ

`registry.deepschool.ru/cvr-aug23/j.polushina/hw-02-service`


### Статический анализ - Линтеры

* `make lint` - запуск линтеров


#### Тестирование
* `make run_unit_tests` - запуск юнит-тестов
* `make run_integration_tests` - запуск интеграционных тестов
* `make run_all_tests` - запуск всех тестов
* `make generate_coverage_report` - сгенерировать coverage-отчет
