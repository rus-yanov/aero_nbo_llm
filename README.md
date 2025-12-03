<body>

<h1>Aero NBO — Прототип 2 (ML + LLM)</h1>

<p>
  Второй прототип Aero NBO реализует гибридную архитектуру, в которой 
  <b>ML-модель определяет лучший оффер</b> для клиента (на основе CTR-прогноза), 
  а <b>LLM (ЯндексGPT или GigaChat)</b> формирует персонализированное маркетинговое сообщение,
  адаптированное под поведение клиента и канал коммуникации.
</p>

<h2>1. Цели прототипа</h2>
<ul>
  <li>Сформировать единый датасет <code>common_dataset.csv</code> с историей поведения клиентов.</li>
  <li>Построить ML-модель, прогнозирующую вероятность конверсии (CTR).</li>
  <li>Ранжировать офферы и выбирать best offer / top-N.</li>
  <li>Создать генератор текстового профиля клиента.</li>
  <li>С помощью LLM сформировать персонализированные сообщения (push/email/SMS).</li>
  <li>Реализовать единый пайплайн <i>client → ML ranking → best offer → LLM message → JSON-response</i>.</li>
</ul>

<h2>2. Входные и выходные данные</h2>

<h3>Входные данные</h3>
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
      <li>title, product_name,</li>
      <li>short_description,</li>
      <li>conditions,</li>
      <li>категория и тип.</li>
    </ul>
  </li>
  <li>Параметры запроса в пайплайн:
    <ul>
      <li><code>client_id</code></li>
      <li><code>top_n</code> (по умолчанию 3)</li>
      <li><code>channel</code> (push/email/SMS)</li>
    </ul>
  </li>
</ul>

<h3>Выходные данные (JSON-ответ NBO-сервиса)</h3>

<pre><code>{
  "client_id": 12345678,
  "best_offer": {
    "offer_id": 987,
    "p_click": 0.27,
    "title": "Скидка 20% на электронику",
    "personalized_message": "Ваш персональный бонус…"
  },
  "alternative_offers": [
    {
      "offer_id": 654,
      "p_click": 0.19,
      "title": "Бесплатная доставка",
      "personalized_message": "Для вас доставка…"
    }
  ]
}
</code></pre>

<p>
Ответ формирует модуль <code>src/service/nbo_pipeline.py</code>.
</p>


<h2>3. Структура проекта</h2>

<pre><code>aero_nbo_llm/
├── data/
│   ├── raw/
│   │   ├── common_dataset.csv       # единый датасет поведения пользователей
│   │   └── offers.csv               # каталог офферов
│   ├── processed/
│   │   └── ml_training_dataset.csv  # финальный ML-датасет
│   └── prompts/                     # шаблоны промптов для LLM
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
│   ├── ml/               # тренировка и инференс ML-модели
│   ├── llm/              # генерация текстов ЯндексGPT/ГигаЧат
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
    На основе <code>common_dataset.csv</code> формируется обучающий датасет 
    <code>ml_training_dataset.csv</code>, включающий:
    <ul>
      <li>историю покупок и взаимодействий,</li>
      <li>RFM-фичи и промо-чувствительность,</li>
      <li>трафик и каналы коммуникации,</li>
      <li>категории и типы офферов,</li>
      <li>демографию, device и поведенческие признаки.</li>
    </ul>
  </li>

  <li>
    Обучается ML-модель (CatBoost/XGBoost), прогнозирующая 
    <code>P(conversion = 1 | client, offer, context)</code>.
  </li>

  <li>
    Для конкретного <code>client_id</code> модель считает <code>p_click</code> для всех офферов 
    из <code>offers.csv</code>, ранжирует их и выбирает лучший или top-N.
  </li>

  <li>
    Далее создаётся текстовый профиль клиента:
    <ul>
      <li>поведение, активность, интересы;</li>
      <li>любимые категории;</li>
      <li>чувствительность к скидкам;</li>
      <li>динамика заказов;</li>
      <li>device-поведение и каналы.</li>
    </ul>
  </li>

  <li>
    LLM получает:
    <ul>
      <li>текстовый профиль клиента,</li>
      <li>данные оффера (title, description, conditions),</li>
      <li>канал коммуникации (push/email/SMS),</li>
      <li>шаблон промпта.</li>
    </ul>
    На основе этого формируется персонализированное сообщение.
  </li>

  <li>
    Модуль <code>src/service/nbo_pipeline.py</code> объединяет результаты ML и LLM 
    в единый JSON-ответ:
    <ul>
      <li><code>best_offer</code>,</li>
      <li><code>alternative_offers</code>,</li>
      <li><code>personalized_message</code>,</li>
      <li><code>p_click</code>.</li>
    </ul>
  </li>
</ol>


<h2>5. Как запустить</h2>

<ol>
  <li>
    Установить зависимости:
    <pre><code>pip install -r requirements.txt</code></pre>
  </li>
</ol>

<p>Затем выполнить ноутбуки по порядку:</p>

<ul>
  <li><b>01_eda.ipynb</b> — анализ <code>common_dataset.csv</code></li>
  <li><b>02_build_training_dataset.ipynb</b> — формирование ML-датасета</li>
  <li><b>03_train_ranking_model.ipynb</b> — обучение ML-ранжировщика</li>
  <li><b>04_user_profile_builder.ipynb</b> — генерация профиля клиента</li>
  <li><b>05_llm_generation.ipynb</b> — генерация LLM-сообщений</li>
  <li><b>06_nbo_llm_demo.ipynb</b> — полный пайплайн ML → LLM → JSON</li>
</ul>


<h2>6. Результат</h2>

<ul>
  <li>обученная ML-модель CTR-ранжирования;</li>
  <li>компонент текстового профиля клиента;</li>
  <li>LLM-модуль на базе ЯндексGPT или GigaChat;</li>
  <li>модуль <code>nbo_pipeline.py</code> для формирования JSON-ответов;</li>
  <li>демо-пайплайн и материалы для аналитической записки.</li>
</ul>

</body>