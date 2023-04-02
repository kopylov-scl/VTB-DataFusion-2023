## Применение вариационного автоэнкодера (LSTM-VRAE) для задачи Защита - Data Fusion Contest 2023

## [Data Fusion Contest 2023](https://ods.ai/tracks/data-fusion-2023-competitions)

[Описание задачи](https://ods.ai/tracks/data-fusion-2023-competitions/competitions/data-fusion2023-defence)

#### Описание решения

- Решение основано на применении модели CatBoost для классификации дефолта пользователя по кредиту
- Скрытое представление LSTM-VRAE используется в качестве признаков для модели CatBoost
- Предсказание RNN так же используется в качестве признака для модели CatBoost

#### Файлы

- run.py - решение, использующее предобученные модели
- training.ipynb - обучение VRAE и CatBoost

#### Метрика соревнования
Mean Harm ROC-AUC. Это среднее гармоническое ROC-AUC на исходных данных и на атакованных. Метрика сочетает в себе компромисс между повышением защищенности модели и потенциальным снижением ее качества. 
