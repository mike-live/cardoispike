# CardoiSpike

Задача определения ковидных аномаий по данным кардиинтервалографии

Установка зависимостей

```
pip install -r requirements.txt
```

### Запуск

```
git clone https://github.com/mike-live/cardoispike.git
cd cardoispike/main
python run.py --data_path '..data/test.csv' --output_path 'output/' --output_filename "output.csv"
```

```..data/test.csv``` - должен иметь следующий формат:

|    |   id |   time |   x |
|---:|-----:|-------:|----:|
|  0 |   81 |      0 | 576 |
|  1 |   81 |    568 | 568 |
|  2 |   81 |   1140 | 572 |
|  3 |   81 |   1716 | 576 |

Тогда 'output/output.csv' будет иметь вид:

|    |   id |   time |   x |   y |
|---:|-----:|-------:|----:|----:|
|  0 |   81 |      0 | 576 |   0 |
|  1 |   81 |    568 | 568 |   0 |
|  2 |   81 |   1140 | 572 |   1 |
|  3 |   81 |   1716 | 576 |   0 |

### Функциональность
1. Автоматическая разметка ковидных аномалий по информации о R-R интервалах ЭКГ
2. Автоматическая фильтрация с помощью постпроцессинга
3. Визуализация и инструмент для сравнения полученной и целевой разметки

### Особенность проекта
1. Скорость инференса
2. Простота интерпретируемости
3. Анализ и коррекция ошибок

### Основной стек технологий:
1. Python3
2. Pandas + NumPy
3. CatBoost


## Установка

### Требуемые зависимости для тестирования решения:
1. Anaconda + Python 3.8
2. Jupyter notebook

### Пакеты python
1. seaborn
2. pandas
3. numpy
4. sci-kit learn
5. matplotlib
6. catboost
7. tqdm
8. tensorflow
9. tensorflow_probability

## Разработчики

[Капралова Анастасия](https://www.github.com/stasysp)
Технический директор GorkyAI

[Вадим Альперович](https://www.github.com/VirtualRoyalty)
Ведущий инженер-исследователь GorkyAI

[Кривоносов Михаил](https://www.github.com/mike_live)
М.н.с. кафедры нейротехнологий ИББМ ННГУ им. Н.И.Лобачевского
