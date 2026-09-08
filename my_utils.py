import pandas as pd
import numpy as np

def add_datetime_features(df):
    """ Функция принимает на вход датасет (данные таксопарка(ов)) в котором 
    имеется колонка с датой и временем начала поездки, добавляет в эту таблицу 
    3 новых признака:
    * pickup_date - дата включения счетчика - начала поездки (без времени);
    * pickup_hour - час дня включения счетчика;
    * pickup_day_of_week - поряд-ый номер дня недели (в виде числа), 
      в который был включен счетчик. 

    Args:
        df (pandas.DataFrame): Входной датасет 
        Столбец с датой и временем должен иметь тип данных datetime.

    Returns:
        df (pandas.DataFrame): Исходный датасет с 3-мя новыми признаками:
        * pickup_date
        * pickup_hour 
        * pickup_day_of_week
    """
    # Сделаем копию входного датасета
    df_copy = df.copy()
    
    # Добавляем необходимые признака в DataFrame
    df_copy['pickup_date'] = df_copy['pickup_datetime'].dt.date
    df_copy['pickup_hour'] = df_copy['pickup_datetime'].dt.hour
    df_copy['pickup_day_of_week'] = df_copy['pickup_datetime'].dt.dayofweek
    
    return df_copy

def add_holiday_features(df_trips, df_holiday):
    """ Функция принимает на вход два датасета df_trips, df_holiday и 
    возвращает обновленный датасет df_trips с добавленным в него столбцом 
    pickup_holiday - бинарным признаком того, начата ли поездка в праздничный 
    день или нет (1 - да, 0 - нет). 

    Args:
        df_trips (pandas.DataFrame): Входной датасет в котором представлена 
        информация о поездках
        df_holiday (pandas.DataFrame): Входной датасет в которм представлена 
        информация о праздничных днях в 2016 г.
        
    Returns:
        df_trips (pandas.DataFrame): Исходный датасет с добавленным в него 
        новым признаком pickup_holiday
    """
    # Сделаем копии входных датасетов для подстраховки
    df_trips_copy = df_trips.copy()
    df_holiday_copy = df_holiday.copy()
    
    # Удалим ненужную колонку из df_holiday
    df_holiday_copy = df_holiday_copy.drop(['day'], axis=1)
    
    # Переведем даты в тип datetime и оставим только дату
    df_holiday_copy['date'] = pd.to_datetime(df_holiday_copy['date']).dt.date
    
    # Переименуем колонки входящего датасета df_holiday для удобства
    df_holiday_copy = df_holiday_copy.rename(
        columns={'date': 'pickup_date', 'holiday': 'pickup_holiday'})
    
    # Объединим две таблицы в одну под названием "df_merge". Объединение будем 
    # производить по одному ключевому столбцу
    df_merge = df_trips_copy.merge(df_holiday_copy, on=['pickup_date'], how='left')
    
    # Заменим значения в колонке 'pickup_holiday' по принципу: 
    # если ячейка заполнена присваиваем 1, а если нет, то 0 
    df_merge['pickup_holiday'] = df_merge['pickup_holiday'].notna().astype(int)
    
    return df_merge

def add_osrm_features(df_trips, df_osrm):
    """ Функция принимает на вход два датасета и возвращает обновленный датасет
    df_trips с добавленными в него тремя столбцами:
    * total_distance - кратчайшее дорожное расстояние (в метрах) между точками
    * total_travel_time - наименьшее время поездки (в секундах) между точками
    * number_of_steps - количество дискретных шагов, которые должен выполнить 
      водитель (поворот налево/поворот направо/ехать прямо и т. д.).

    Args:
        df_trips (pandas.DataFrame): Входной датасет в котором представлена 
        информация о поездках
        df_osram (pandas.DataFrame): Входной датасет в которм представлена 
        информация для построения кратчайшего маршрута.
        
    Returns:
        pandas.DataFrame: Исходный датасет с добавленными в него 
        новыми признаками.
    """
    # Сделаем копии входных датасетов для подстраховки
    df_trips_copy = df_trips.copy()
    df_osrm_copy = df_osrm.copy()
    
    # Обновим датасет, оставив только нужные нам признаки
    df_osrm_copy = df_osrm_copy[
        ['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
    
    # Объединим две таблицы в одну под названием "df_merge_2". Объединение будем 
    # производить по одному ключевому столбцу 
    df_merge = df_trips_copy.merge(df_osrm_copy, on=['id'], how='left')
    
    return df_merge 

def get_haversine_distance(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # радиус земли в километрах
    EARTH_RADIUS = 6371 
    # считаем кратчайшее расстояние h по формуле Хаверсина
    lat_delta = lat2 - lat1
    lng_delta = lng2 - lng1
    d = np.sin(lat_delta * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng_delta * 0.5) ** 2
    h = 2 * EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def get_angle_direction(lat1, lng1, lat2, lng2):
    # переводим углы в радианы
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    # считаем угол направления движения alpha по формуле угла пеленга
    lng_delta_rad = lng2 - lng1
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    alpha = np.degrees(np.arctan2(y, x))
    return alpha

def add_geographical_features(df_trips):
    """ Функция принимает на вход датасет с данными о поездках и возвращает 
    обновленную таблицу с добавленными в нее двумя столбцами:
    * haversine_distance - расстояние Хаверсина между точкой, в которой был включен 
      счетчик, и точкой, в которой счетчик был выключен;
    * direction - направление движения из точки, в которой был включен счетчик,
      в точку, в которой счетчик был выключен.
      
    Args:
        df_trips (pandas.DataFrame): Входной датасет в котором представлена 
        информация о поездках
        
    Returns:
        pandas.DataFrame: Исходный датасет с добавленными в него 
        новыми признаками.
    """
    # Делаем копию для безопасности
    df_trips_copy = df_trips.copy()
    
    # Извлекаем координаты в NumPy массивы
    lat1 = np.array(df_trips_copy['pickup_latitude'])
    lng1 = np.array(df_trips_copy['pickup_longitude'])
    lat2 = np.array(df_trips_copy['dropoff_latitude'])
    lng2 = np.array(df_trips_copy['dropoff_longitude'])
    
    # Расчитываем и записываем два новых признака
    df_trips_copy['haversine_distance'] = get_haversine_distance(lat1, lng1, lat2, lng2)
    df_trips_copy['direction'] = get_angle_direction(lat1, lng1, lat2, lng2)
    
    return df_trips_copy

def add_cluster_features(df_trips, alg):
    """ Функция принимает на вход датасет с данными о поездках и обученный 
        алгоритм кластеризации

    Args:
        df_trips (pandas.DataFrame): Входной датасет в котором представлена 
        информация о поездках
        kmean (_type_): Обучпенный алгоритм кластеризации
        
    Returns:
        pandas.DataFrame: Исходный датасет с добавленным в него 
        новым признаком.
    """
    # Делаем копию для безопасности
    df_trips_copy = df_trips.copy()
    
    # Собираем координаты ИМЕННО ТОЙ таблицы, которую передали в функцию
    # Это делает функцию независимой от внешних переменных
    coords_current = np.hstack((df_trips_copy[['pickup_latitude', 'pickup_longitude']],
                                df_trips_copy[['dropoff_latitude', 'dropoff_longitude']]))
    
    df_trips_copy['geo_cluster'] = alg.predict(coords_current)
    
    return df_trips_copy 

def add_weather_features(df_trips, df_weather):
    """ Функция принимает на вход два датасета: основной, с данными о поездках 
    и вспомогательный, с данными о погоде. Возвращает обновленный датасет
    df_trips с добавленными в него пятью столбцами:
    * temperature - температура;
    * visibility - видимость;
    * wind speed - средняя скорость ветра;
    * precip - количество осадков;
    * events - погодные явления.

    Args:
        df_trips (pandas.DataFrame): Входной датасет в котором представлена 
        информация о поездках.
        df_weather (pandas.DataFrame): Входной датасет в которм представлены 
        почасовые данные о погодных условиях.
        
    Returns: 
        pandas.DataFrame: Исходный датасет с добавленными в него 
        новыми признаками.
    """
    # Делаем копию входящих DF для безопасности
    df_trips_copy = df_trips.copy()
    df_weather_copy = df_weather.copy()
    
    # Приведем данные в колоках с указанием даты к одному типу
    df_trips_copy['pickup_date'] = pd.to_datetime(df_trips_copy['pickup_date']).dt.date
    df_weather_copy['date'] = pd.to_datetime(df_weather_copy['date']).dt.date
    
    # Создаем список нужных нам признаков из датасета df_weather
    columns_need_lst = ['temperature', 'visibility', 'wind speed', 'precip', 
                        'events', 'date', 'hour']
    # Создаем список признаков подлежащих удалению
    columns_dell_lst = df_weather_copy.columns.difference(columns_need_lst)
    # Удаляем из df_weather_copy все ненужные признаки 
    # (согласно условиям задания для коррекции df_weather_copy используем drop)
    df_weather_copy = df_weather_copy.drop(columns = columns_dell_lst)
    
    # Объединяем две таблицы по разным названиям ключевых столбцов
    # согласно условиям задания 
    df_merge = df_trips_copy.merge(
        df_weather_copy, 
        left_on = ['pickup_date', 'pickup_hour'], 
        right_on = ['date', 'hour'],
        how = 'left'
    )
    # Удаляем дубликаты ключевых столбцов
    df_merge = df_merge.drop(columns =['date', 'hour'])
    
    return df_merge

def fill_null_weather_data(df_trips):
    """ Функция заполняет пропущенные значения в столбцах

    Args:
        taxi_data (pandas.DataFrame): Входной датасет в котором представлена 
        информация о поездках. Датасет имеет столбцы с пропущенными значениями.
        
    Returns:
        pandas.DataFrame: Исходный датасет без пропусков в столбцах
    """
    # Создаем копию для безопасности
    df_trips_copy = df_trips.copy()
    
    # Составим список столбцов с погодными условиями с пропущенными значениями
    weather_miss_lst= ['temperature', 'visibility', 'wind speed', 'precip']
    # Группируем по дате и заполняем пропуски во всех столбцах сразу
    df_trips_copy[weather_miss_lst] = df_trips_copy[weather_miss_lst].fillna(
        df_trips_copy.groupby('pickup_date')[weather_miss_lst].transform('median'))
    
     # Составим список столбцов связанных с osrm c пропущенными значениями
    osrm_miss_lst = ['total_distance', 'total_travel_time', 'number_of_steps']
    # Заполняем пропуски медианным значением во всех столбцах сразу
    df_trips_copy[osrm_miss_lst] = df_trips_copy[osrm_miss_lst].fillna(
        df_trips_copy[osrm_miss_lst].median())
    
    # Заполняем пропуски в событиях строкой None
    df_trips_copy['events'] = df_trips_copy['events'].fillna('None')
    
    return df_trips_copy