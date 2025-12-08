<body>

<h1>Aero NBO — Прототип 2 (ML + LLM)</h1>

<p>
  Второй прототип Aero NBO реализует гибридную архитектуру, в которой 
  <b>ML-модель ранжирует офферы по вероятности конверсии (CTR)</b>, 
  а <b>LLM (GigaChat, с fallback на OpenAI)</b> формирует персонализированное маркетинговое сообщение,
  адаптированное под поведение клиента и канал коммуникации. 
  Дополнительно используется бизнес-фильтр рентабельности:
  <code>p_click ≥ cost / offer_AOV</code>.
</p>

<h2>1. Цели прототипа</h2>
<ul>
  <li>Сформировать единый датасет <code>common_dataset.csv</code> с историей поведения клиентов.</li>
  <li>Построить ML-модель с обучением с учителем (таргет <code>conversion</code>), прогнозирующую вероятность клика.</li>
  <li>Ранжировать офферы и выбирать best offer / top-N, с учётом бизнес-фильтра рентабельности.</li>
  <li>Создать генератор текстового профиля клиента.</li>
  <li>С помощью LLM сформировать персонализированные сообщения (push/email/SMS) с автоматическим переключением на OpenAI при ошибке GigaChat.</li>
  <li>Реализовать единый пайплайн <i>client → ML ranking → бизнес-фильтр → best offer → LLM message → JSON-response</i>.</li>
</ul>

<h2>2. Входные и выходные данные</h2>

<h3>2.1. Оффлайн-данные (обучение и батч-прогноз)</h3>
<ul>
  <li><code>common_dataset.csv</code> — общий датасет:
    <ul>
      <li>демография,</li>
      <li>RFM-признаки,</li>
      <li>история взаимодействий с офферами,</li>
      <li>ценовые сегменты и предпочтения,</li>
      <li>каналы коммуникации,</li>
      <li>история скидок и покупок.</li>
    </ul>
  </li>
  <li><code>offers.csv</code> — активные офферы:
    <ul>
      <li>title, product_name, short_description, conditions,</li>
      <li>тип и категория,</li>
      <li><code>cost</code> — стоимость/размер скидки,</li>
      <li><code>offer_AOV</code> — средний чек по категории оффера.</li>
    </ul>
  </li>
</ul>

<p>
  На их основе формируется <code>ml_training_dataset.csv</code> с признаками и таргетом 
  <code>conversion</code>, на котором обучается модель ранжирования (train/test = 80%/20%).
</p>

<h3>2.2. Онлайн-вход в NBO-пайплайн</h3>

<p>Пайплайн поддерживает два сценария использования.</p>

<h4>А) Режим «по client_id» (внутренние данные)</h4>

<p>На вход подаётся только идентификатор клиента и параметры вызова. Строки client × offer берутся из подготовленного ML-датасета.</p>

<ul>
  <li>Параметры запроса:
    <ul>
      <li><code>client_id</code> — идентификатор клиента;</li>
      <li><code>top_n</code> — сколько офферов вернуть (по умолчанию 3);</li>
      <li><code>channel</code> — формат сообщения: <code>push</code> / <code>email</code> / <code>sms</code>;</li>
      <li><code>provider</code> (опционально) — выбор LLM-провайдера 
        (<code>gigachat</code>, <code>openai</code>, <code>gigachat_with_openai_fallback</code>).
      </li>
    </ul>
  </li>
</ul>

<p>Пример запроса:</p>

<pre><code>
curl -X POST http://localhost:8000/api/v1/nbo/by-client \
  -H "Content-Type: application/json" \
  -d '{
        "client_id": 12490,
        "top_n": 3,
        "channel": "push",
        "provider": "gigachat_with_openai_fallback"
      }'
</code></pre>

<h4>Б) Режим «по фичам» (внешняя система шлёт строки датасета)</h4>

<p>
  Внешний сервис может передать набор строк с признаками 
  (аналог строк <code>ml_training_dataset.csv</code> без таргета <code>conversion</code>).
  Отсутствующие признаки сервис автоматически заполняет дефолтными значениями 
  (числовые — 0.0, категориальные — <code>"unknown"</code>).
</p>

<ul>
  <li>Параметры запроса:
    <ul>
      <li><code>rows</code> — список JSON-объектов с признаками <code>client × offer × context</code>;</li>
      <li><code>client_id</code> — необязательно, может быть взят из первой строки <code>rows</code>;</li>
      <li><code>top_n</code>, <code>channel</code>, <code>provider</code> — аналогично режиму А.</li>
    </ul>
  </li>
</ul>

<p>Пример запроса (один оффер):</p>

<pre><code>
curl -X POST http://localhost:8000/api/v1/nbo/by-rows \
  -H "Content-Type: application/json" \
  -d '{
        "client_id": 12490,
        "top_n": 3,
        "channel": "push",
        "provider": "gigachat",
        "rows": [
          {
            "client_id": 12490,
            "offer_id": 0,
            "offer_type": "discount_10",
            "offer_category": "category_1",
            "cost": 5.0,
            "offer_AOV": 30.0,
            "price_segment": "mid",
            "recency_days": 10,
            "frequency_90d": 5,
            "monetary_90d": 5000,
            "discounts_used_90d": 2,
            "avg_discount_percent_90d": 15.0,
            "favorite_category": "category_1",
            "email_open_rate_30d": 0.4
            // ... остальные фичи по схеме обучающего датасета
          }
        ]
      }'
</code></pre>

<p>Пример запроса (несколько офферов для одного клиента):</p>

<pre><code>
curl -X POST http://localhost:8000/api/v1/nbo/by-rows \
  -H "Content-Type: application/json" \
  -d '{
        "client_id": 16189,
        "top_n": 3,
        "channel": "push",
        "rows": [
          {
            "client_id": 16189,
            "offer_id": 1,
            "offer_type": "discount_20",
            "offer_category": "category_8",
            "cost": 2.09,
            "offer_AOV": 19.59,
            "recency_days": 15,
            "frequency_90d": 11,
            "monetary_90d": 215.48,
            "discounts_used_90d": 9,
            "favorite_category": "category_8",
            "email_open_rate_30d": 0.07,
            "price_segment": "budget"
          },
          {
            "client_id": 16189,
            "offer_id": 2,
            "offer_type": "bonus_points",
            "offer_category": "category_5",
            "cost": 1.5,
            "offer_AOV": 25.0,
            "recency_days": 20,
            "frequency_90d": 8,
            "monetary_90d": 310.0,
            "discounts_used_90d": 4,
            "favorite_category": "category_5",
            "email_open_rate_30d": 0.12,
            "price_segment": "mid"
          }
        ]
      }'
</code></pre>

<h3>2.3. Выходные данные (JSON-ответ NBO-сервиса)</h3>

<p>
  В обоих режимах пайплайн возвращает одинаковый JSON, 
  сформированный модулем <code>src/service/nbo_pipeline.py</code>.
  Все офферы предварительно проходят бизнес-фильтр рентабельности:
  <code>p_click ≥ cost / offer_AOV</code>. Нерентабельные предложения в выдачу не попадают.
</p>

<pre><code>{
  "client_id": 12345678,
  "channel": "push",
  "user_profile": "Краткий текстовый профиль клиента на естественном языке...",
  "best_offer": {
    "offer_id": 987,
    "p_click": 0.27,
    "title": "Скидка 20% на электронику",
    "short_description": "Краткое описание оффера...",
    "conditions": "Условия действия предложения...",
    "personalized_message": "Ваш персональный бонус на электронику, подобранную под ваш обычный уровень трат..."
  },
  "alternative_offers": [
    {
      "offer_id": 654,
      "p_click": 0.19,
      "title": "Бесплатная доставка",
      "short_description": "Доставка без доплат при заказе от 5000 ₽",
      "conditions": "Только для онлайн-заказов до конца недели.",
      "personalized_message": "Для вас действует бесплатная доставка на ближайший заказ — без лишних условий."
    }
  ]
}
</code></pre>

<p>
  Этот формат предназначен для прямой интеграции с сервисом рассылок или оркестратором кампаний.
</p>


<h2>3. Структура проекта</h2>

<pre><code>aero_nbo_llm/
├── data/
│   ├── raw/
│   │   ├── common_dataset.csv       # единый датасет поведения пользователей
│   │   └── offers.csv               # каталог офферов (в т.ч. cost и offer_AOV)
│   ├── processed/
│   │   └── ml_training_dataset.csv  # финальный ML-датасет (признаки + conversion)
│   └── prompts/                     # шаблоны промптов для LLM (push/email/sms)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_build_training_dataset.ipynb
│   ├── 03_train_ranking_model.ipynb
│   ├── 04_user_profile_builder.ipynb
│   ├── 05_llm_generation.ipynb
│   └── 06_nbo_llm_demo.ipynb
│
├── src/
│   ├── data_prep/        # подготовка данных, фичи, кодирование
│   ├── ml/               # тренировка и инференс ML-модели (CatBoost)
│   ├── llm/              # генерация текстов GigaChat (+ fallback на OpenAI)
│   ├── evaluation/       # метрики и сравнение подходов
│   ├── service/
│   │   └── nbo_pipeline.py     # формирование итогового JSON-ответа
│   └── utils/            # конфиги, загрузчики, утилиты
│
├── models/
│   └── ranking_model.pkl
│
├── reports/
└── README.md
</code></pre>


<h2>4. Описание подхода</h2>

<ol>
  <li>
    На основе <code>common_dataset.csv</code> и <code>offers.csv</code> формируется обучающий датасет 
    <code>ml_training_dataset.csv</code>, включающий:
    <ul>
      <li>историю покупок и взаимодействий,</li>
      <li>RFM-фичи и промо-чувствительность,</li>
      <li>трафик и каналы коммуникации,</li>
      <li>категории и типы офферов,</li>
      <li>демографию и поведенческие признаки,</li>
      <li>таргет <code>conversion</code> (клик/конверсия).</li>
    </ul>
  </li>

  <li>
    Обучается ML-модель (CatBoostClassifier), прогнозирующая 
    <code>P(conversion = 1 | client, offer, context)</code>. 
    Данные делятся на train/test в пропорции 80/20.
  </li>

  <li>
    Для конкретного клиента (по <code>client_id</code> или по переданным строкам <code>rows</code>) 
    модель считает <code>p_click</code> для всех доступных офферов.
    Далее применяется бизнес-фильтр рентабельности:
    <ul>
      <li>рассчитывается минимально необходимая вероятность: <code>min_required_p = cost / offer_AOV</code>,</li>
      <li>в выдаче остаются только офферы, где <code>p_click ≥ min_required_p</code>.</li>
    </ul>
    Из оставшихся офферов выбираются лучший и top-N альтернатив.
  </li>

  <li>
    На основе признаков клиента формируется краткий текстовый профиль:
    <ul>
      <li>давность и частота покупок,</li>
      <li>сумма трат за период,</li>
      <li>любимая категория и сегмент цены,</li>
      <li>чувствительность к скидкам,</li>
      <li>открываемость писем и прочие поведенческие характеристики.</li>
    </ul>
  </li>

  <li>
    LLM получает:
    <ul>
      <li>текстовый профиль клиента,</li>
      <li>данные оффера (title, description, conditions, cost, offer_AOV),</li>
      <li>канал коммуникации (push/email/SMS) и соответствующий промпт-шаблон.</li>
    </ul>
    По умолчанию используется GigaChat; при ошибке запроса автоматически используется OpenAI как fallback.
  </li>

  <li>
    Модуль <code>src/service/nbo_pipeline.py</code> объединяет результаты ML и LLM 
    в единый JSON-ответ, удобный для интеграции с сервисом рассылок:
    <ul>
      <li><code>user_profile</code>,</li>
      <li><code>best_offer</code>,</li>
      <li><code>alternative_offers</code>,</li>
      <li><code>p_click</code> и <code>personalized_message</code> для каждого оффера.</li>
    </ul>
  </li>
</ol>


<h2>5. Как запустить</h2>

<ol>
  <li>
    Установить зависимости:
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li>
    Настроить файл <code>.env</code> (ключи для GigaChat и OpenAI, переменная <code>LLM_PROVIDER</code> и т.д.).
  </li>
  <li>
    При необходимости воспроизвести обучение — выполнить ноутбуки по порядку:
    <ul>
      <li><b>01_eda.ipynb</b> — анализ <code>common_dataset.csv</code></li>
      <li><b>02_build_training_dataset.ipynb</b> — формирование ML-датасета</li>
      <li><b>03_train_ranking_model.ipynb</b> — обучение ML-ранжировщика</li>
      <li><b>04_user_profile_builder.ipynb</b> — генерация профиля клиента</li>
      <li><b>05_llm_generation.ipynb</b> — генерация LLM-сообщений</li>
      <li><b>06_nbo_llm_demo.ipynb</b> — демонстрация полного пайплайна ML → LLM → JSON</li>
    </ul>
  </li>
  <li>
    Запустить API-сервис:
    <pre><code>uvicorn src.service.api:app --reload</code></pre>
    Swagger UI: <code>http://localhost:8000/docs</code>
  </li>
</ol>


<h2>6. Результат</h2>

<ul>
  <li>обученная ML-модель CTR-ранжирования (CatBoost, train/test = 80/20);</li>
  <li>компонент текстового профиля клиента;</li>
  <li>LLM-модуль на базе GigaChat с автоматическим fallback на OpenAI;</li>
  <li>бизнес-фильтр рентабельности офферов <code>p_click ≥ cost / offer_AOV</code>;</li>
  <li>устойчивость к «укороченным» онлайн-payload’ам (автодозаполнение отсутствующих признаков);</li>
  <li>модуль <code>nbo_pipeline.py</code> для формирования JSON-ответов;</li>
  <li>две схемы интеграции: по <code>client_id</code> и по внешним фичам (<code>rows</code>);</li>
  <li>демо-пайплайн и готовая основа для аналитической записки/презентации.</li>
</ul>

</body>