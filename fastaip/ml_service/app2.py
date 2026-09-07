from fastapi import FastAPI
import pandas as pd
import joblib
import numpy as np
import featuretools as ft
import woodwork
import json
import time
import datetime
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY
import psutil
app = FastAPI()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
cpu_usage = Gauge('cpu_usage', 'CPU usage')
memory_usage = Gauge('memory_usage', 'Memory usage')
latency = Histogram('latency', 'Request latency')
main_app_predictions = Histogram(
    # имя метрики
    "main_app_predictions",
    #описание метрики
    "Histogram of product recommendations",
    #указаываем корзины для гистограммы
    buckets=(0,1,2,3)
)

error_counter = Counter("Error counter", "Count fo errors in recomendation service")
model = joblib.load('saved_model.pkl')

with open('replacer.json', 'r', encoding='utf-8') as f:
    replacer = json.load(f)
personal_recs = pd.read_parquet('personal_als.parquet')
medians = {'age' : 39.0, 'antiguedad' : 50.0, 'renta' : 101850.0}
modes = {
    'ind_empleado': 'N',
    'pais_residencia': 'ES',
    'sexo': 'V',
    'fecha_alta': '2014-07-28',
    'ind_nuevo': 0,
    'indrel': 1,
    'ult_fec_cli_1t': '2015-12-24',
    'indrel_1mes': 1.0,
    'tiprel_1mes': 'I',
    'indresi': 'S',
    'indext': 'N',
    'conyuemp': 'N',
    'canal_entrada': 'KHE',
    'indfall': 'N',
    'tipodom': 1,
    'cod_prov': 28,
    'nomprov': 'MADRID',
    'ind_actividad_cliente': 0,
    'segmento': '02 - PARTICULARES',
    'ind_ahor_fin_ult1': 0,
    'ind_aval_fin_ult1': 0,
    'ind_cco_fin_ult1': 1,
    'ind_cder_fin_ult1': 0,
    'ind_cno_fin_ult1': 0,
    'ind_ctju_fin_ult1': 0,
    'ind_ctma_fin_ult1': 0,
    'ind_ctop_fin_ult1': 0,
    'ind_ctpp_fin_ult1': 0,
    'ind_deco_fin_ult1': 0,
    'ind_deme_fin_ult1': 0,
    'ind_dela_fin_ult1': 0,
    'ind_ecue_fin_ult1': 0,
    'ind_fond_fin_ult1': 0,
    'ind_hip_fin_ult1': 0,
    'ind_plan_fin_ult1': 0,
    'ind_pres_fin_ult1': 0,
    'ind_reca_fin_ult1': 0,
    'ind_tjcr_fin_ult1': 0,
    'ind_valo_fin_ult1': 0,
    'ind_viv_fin_ult1': 0,
    'ind_nomina_ult1': 0,
    'ind_nom_pens_ult1': 0,
    'ind_recibo_ult1': 0
 }

ult_fec_cli_1t_intervals = [ '2015-06-30 – 2015-07-09',
       '2015-07-21 – 2015-08-03', '2015-07-09 – 2015-07-21',
       '2015-08-03 – 2015-09-14', '2015-09-14 – 2015-10-19',
       '2015-10-19 – 2015-11-18', '2015-11-18 – 2015-12-21',
       '2015-12-21 – 2016-01-11', '2016-01-11 – 2016-02-10',
       '2016-02-10 – 2016-03-16', '2016-03-16 – 2016-04-26',
       '2016-04-26 – 2016-05-30']
fecha_alta_intervals = [
    '2014-08-13 – 2015-02-27', '2012-07-23 – 2012-12-10',
       '2013-10-18 – 2014-08-13',
       '2012-12-10 – 2013-10-18', '2004-04-23 – 2006-07-11',
       '2002-02-16 – 2004-04-23', '2011-09-01 – 2012-07-23',
       '2006-07-11 – 2008-09-30', '2008-09-30 – 2011-09-01',
       '2000-05-16 – 2002-02-16', '1995-01-15 – 2000-05-16',
       '2015-02-27 – 2016-05-31'
]
prediction_stats = {}
def update_resource_metrics():
    """Обновление метрик использования ресурсов"""
    cpu_usage.set(psutil.cpu_percent())
    memory_usage.set(psutil.virtual_memory().percent)

def find_interval(input_date, intervals):
    # Конвертируем input_date в datetime.date, если это строка или datetime
    if isinstance(input_date, str):
        input_date = datetime.datetime.strptime(input_date, "%Y-%m-%d").date()
    elif isinstance(input_date, datetime.datetime):
        input_date = input_date.date()

    for interval in intervals:
        # Исправлен разделитель на длинное тире '–' (U+2013)
        start_str, end_str = interval.split(' – ')  # Важно: длинное тире с пробелами!
        
        # Парсим даты
        start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()
        
        if start_date <= input_date <= end_date:
            return interval
    
    return None


def feature_engineering(df):
    """Автоматическая генерация признаков для всех типов переменных"""
    
    # Сохраняем оригинальные данные
    original_df = df.copy()
    
    # Создаем EntitySet
    es = ft.EntitySet(id='bank_data')
    es = es.add_dataframe(
        dataframe_name='main',
        dataframe=df,                # Исходный DataFrame без reset_index()
        index='unique_id',           # Название для нового индекса
        make_index=True,             # Сгенерировать уникальный индекс
        logical_types={
            'antiguedad': woodwork.logical_types.Double,
            'renta': woodwork.logical_types.Double,
            **{col: woodwork.logical_types.Categorical for col in df.columns 
            if col not in ['antiguedad', 'renta']}  # Убрали 'index' из исключений
        }
    )


    # Определяем примитивы для разных типов признаков
    trans_primitives = [
        'add_numeric',
        'multiply_numeric',
        'divide_numeric',
        'natural_logarithm',  # Правильное название для log
        'square_root'         # Правильное название для sqrt
    ]
    
    agg_primitives = [
        'mean',
        'median',
        'std',
        'max',
        'min',
        'count',
        'num_unique'
    ]

    # Генерация признаков
    feature_matrix, _ = ft.dfs(
        entityset=es,
        target_dataframe_name='main',
        trans_primitives=trans_primitives,
        agg_primitives=agg_primitives,
        where_primitives=['count'],
        max_depth=2,
        features_only=False,
        verbose=True
    )

    # Объединение признаков
    try:
        new_features = feature_matrix.drop(columns=['unique_id'])
    except:
        new_features = feature_matrix
    try:
        new_features = feature_matrix.drop(columns=['index'])
    except:
        new_features = feature_matrix
    #df = pd.concat([original_df, new_features], axis=1)
    df = new_features
    #print(f'Новые признаки: {new_features.columns}')
    # Ручные трансформации
    df = manual_transformations(df)
    
    # Удаление дубликатов
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def manual_transformations(df):
    """Ручные преобразования и кастомные фичи"""
    # Для числовых
    #print(df.columns)
    df['renta_antiguedad_ratio'] = df['renta'] / (df['antiguedad'] + 1)
    df['log_renta'] = np.log1p(df['renta'])
    
    # Для категориальных
    for col in ['pais_residencia', 'segmento']:
        df[f'mean_renta_by_{col}'] = df.groupby(col)['renta'].transform('mean')
        df[f'median_antiguedad_by_{col}'] = df.groupby(col)['antiguedad'].transform('median')
    
    # Взаимодействие категорий
    df['renta_vs_country_mean'] = df['renta'] / df['mean_renta_by_pais_residencia']
    
    return df
def log_metrics_to_md(raw_request: dict, prediction: int, metrix: dict):
    request_json = json.dumps(raw_request, indent=2, ensure_ascii=False)
    # Получаем метрики из Prometheus
    try:
        metrics_section = f"""
                                ### Prometheus Metrics
                                ```prometheus
                                {metrix}
                                ```
                            """
    except Exception as e:
        metrics_section = f"\n### Metrics Error\nFailed to collect metrics: {str(e)}"
    md_content = f"""
                ## Recommendation Metrics Report  
                **Date**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
                ### Request Data  
                ```json  
                {request_json}```
                ### Prediction  
                **Product ID**: {prediction}  
                {metrics_section}
                """
    
    try:
        with open("/home/ml_service/metrics_reports/metrics_report.md", "a") as f:
            f.write(md_content)
    except:
        with open("metrics_report.md", "a") as f:
            f.write(md_content)

@app.on_event("startup")
async def startup_event():
    """Запуск фоновой задачи для обновления метрик ресурсов"""
    import asyncio
    async def update_metrics():
        while True:
            update_resource_metrics()
            await asyncio.sleep(5)
    asyncio.create_task(update_metrics())
@app.post("/predict")
async def predict(data: dict):
    start_time = time.time()
    update_resource_metrics()
    try:
            #MODEL_PREDICTIONS.labels(product_id=prediction).inc()
            
            # Конвертация входных данных в DataFrame
            rf = pd.DataFrame([data])
            print('Got data')
            
            rf = rf.drop(['indrel','indext'],axis=1) 
            print('Dropped indrel,indext')
            for col in ['age','antiguedad','renta']:
                rf[col] = pd.to_numeric(rf[col], errors='coerce')
            print('Corrected numberic formats')
            rf['antiguedad']=rf['antiguedad'].replace(np.float64(-999999.0),np.nan)
            intervals = [(2, 23), (24, 28), (29, 36), (37, 41), (42, 45), (46, 49), (50, 54), (55, 63), (64, 164)]
            rf['age_interval'] = pd.cut(rf['age'], bins=[interval[0] for interval in intervals] + [intervals[-1][1]], 
                                        labels=[f"{interval[0]}-{interval[1]}" for interval in intervals])
            print('Got age intervals')
            rf['fecha_alta']=rf['fecha_alta'].apply(lambda x: find_interval(x, fecha_alta_intervals))
            rf['ult_fec_cli_1t']=rf['ult_fec_cli_1t'].apply(lambda x: find_interval(x, ult_fec_cli_1t_intervals))
            print('Got fecha_alta and ult_fec_cli_1t intervals')
            cats = [x for x in list(rf.columns) if x not in ['age','antiguedad','fecha_dato','ncodpers','renta']]
            for col in cats[:-24]:
                rf[col].map(replacer[col])
            print('Mapped cat columns')
            rf['renta'] = rf['renta'].clip(lower=26449.65, upper=337117.17)
            rf['antiguedad'] = rf['antiguedad'].clip(lower=1.0, upper=207.0)
            print('Clipped num columns')
            for nc in ['age', 'antiguedad', 'renta']:
                rf[nc] = rf[nc].fillna(medians[nc])
            print('Filled nans with medians')
            for col, value in modes.items():
                try:
                    rf[col] = rf[col].fillna(value)
                except:
                    pass

            try:
                recommended_product_id = personal_recs.loc[rf['ncodpers'],'recommended_product_id']
            except:
                recommended_product_id = 0
            print('Filled nans with modes')
            rf['recommended_product_id'] = recommended_product_id
            print('Got recommended_product_id')
            # 2. Преобразование типов
            for col in ['age', 'antiguedad', 'renta']:
                rf[col] = pd.to_numeric(rf[col], errors='coerce')
            
            # 3. Генерация признаков
                products=['ind_ahor_fin_ult1',
            'ind_aval_fin_ult1',
            'ind_cco_fin_ult1',
            'ind_cder_fin_ult1',
            'ind_cno_fin_ult1',
            'ind_ctju_fin_ult1',
            'ind_ctma_fin_ult1',
            'ind_ctop_fin_ult1',
            'ind_ctpp_fin_ult1',
            'ind_deco_fin_ult1',
            'ind_deme_fin_ult1',
            'ind_dela_fin_ult1',
            'ind_ecue_fin_ult1',
            'ind_fond_fin_ult1',
            'ind_hip_fin_ult1',
            'ind_plan_fin_ult1',
            'ind_pres_fin_ult1',
            'ind_reca_fin_ult1',
            'ind_tjcr_fin_ult1',
            'ind_valo_fin_ult1',
            'ind_viv_fin_ult1',
            'ind_nomina_ult1',
            'ind_nom_pens_ult1',
            'ind_recibo_ult1']
            for c in products:
                rf[c] = pd.to_numeric(rf[c], errors='coerce')
            rf['total_products'] = rf[products].sum(axis=1)
            print('Got total products')
            rf = rf[['ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'antiguedad',
                'indrel_1mes', 'tiprel_1mes', 'indresi', 'conyuemp', 'canal_entrada',
                'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento',
                'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
                'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
                'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
                'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
                'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
                'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
                'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
                'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',
                'total_products', 'age_interval', 'recommended_product_id']]
            print('Finalized columnds')
            rf['antiguedad'] = rf['antiguedad'].astype(int)
            rf = feature_engineering(rf)
            print('Feature engineering complete')
            
            # 4. Преобразование типов для модели
            nums = ['antiguedad', 'renta', 'antiguedad + renta', 'antiguedad / renta', 'renta / antiguedad', 'antiguedad * renta', 'NATURAL_LOGARITHM(antiguedad)', 'NATURAL_LOGARITHM(renta)', 'SQUARE_ROOT(antiguedad)', 'SQUARE_ROOT(renta)', 'renta_antiguedad_ratio', 'log_renta', 'renta_vs_country_mean']
            cats = ['ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'conyuemp', 'canal_entrada', 'indfall', 'cod_prov', 'ind_actividad_cliente', 'segmento', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1', 'total_products', 'age_interval', 'recommended_product_id', 'mean_renta_by_pais_residencia', 'median_antiguedad_by_pais_residencia', 'mean_renta_by_segmento', 'median_antiguedad_by_segmento']
        
            for nu in nums:
                rf[nu] = pd.to_numeric(rf[nu], errors='coerce')
            
            for cat in cats:
                rf[cat] = rf[cat].astype('str')
            print('Got final cats and nums') 
             # Предсказание модели
            prediction = int(model.predict(rf)[0])
            latency.observe(time.time() - start_time)
            try:
                main_app_predictions.observe(prediction)
            except:
                print(r'Ошибка в main_app_predictions')
            
            metrix = {
                        'CPU usage': cpu_usage._value.get() if hasattr(cpu_usage, '_value') else 0,
                        "Memory usage": memory_usage._value.get() if hasattr(memory_usage, '_value') else 0,
                        "Number of errors": error_counter._value.get() if hasattr(error_counter, '_value') else 0,
                        "Latency": (REGISTRY.get_sample_value('latency_sum') / REGISTRY.get_sample_value('latency_count')) 
                                    if REGISTRY.get_sample_value('latency_count') else 0,
                        "Predictions distribution": {
                            k: v.get() 
                            for k, v in main_app_predictions._metrics.items()
                        } if hasattr(main_app_predictions, '_metrics') else {}
                    }

            log_metrics_to_md(data, prediction, metrix)
            return {"prediction": prediction}
    except:
            try:
                error_counter.inc()
            except:
                print("Ошибка в обработчике ошибок")
            raise