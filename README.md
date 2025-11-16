<body>

  <h1>Aero NBO — Прототип 2 (ML + LLM)</h1>

  <p>
    Этот прототип реализует современный гибридный подход к Next Best Offer (NBO), объединяя 
    машинное обучение (ML) и большие языковые модели (LLM). ML выбирает <i>что</i> предложить 
    пользователю, а LLM формирует персонализированное текстовое сообщение: <i>как</i> именно 
    предложить оффер для максимального CTR.
  </p>

  <h2>1. Цели прототипа</h2>
  <ul>
    <li>Построить ML-модель ранжирования офферов по вероятности клика (CTR).</li>
    <li>Для каждого пользователя формировать топ-N офферов.</li>
    <li>Построить генератор текстового профиля пользователя.</li>
    <li>С помощью LLM генерировать персонализированные маркетинговые сообщения под каждый оффер.</li>
    <li>Сравнить качество ML+LLM с rule-based baseline.</li>
  </ul>

  <h2>2. Стек технологий</h2>

  <h3>Язык / среда</h3>
  <ul>
    <li>Python 3.10+</li>
    <li>Jupyter Notebook</li>
    <li>VS Code / PyCharm</li>
  </ul>

  <h3>ML-библиотеки</h3>
  <ul>
    <li>pandas, numpy</li>
    <li>scikit-learn</li>
    <li>catboost или xgboost</li>
  </ul>

  <h3>LLM-модуль</h3>
  <ul>
    <li>Яндекс GPT — вариант A</li>
    <li>Giga Chat — вариант B</li>
    <li>HuggingFace Inference или локальный запуск</li>
  </ul>

  <h2>3. Структура проекта</h2>

  <pre><code>aero_nbo_llm/
├── data/
│   ├── raw/                     # исходные данные
│   ├── processed/               # ml_training_dataset.csv
│   └── prompts/                 # шаблоны промптов для LLM
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
│   ├── data_prep/
│   ├── ml/
│   ├── llm/
│   ├── evaluation/
│   └── utils/
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
    Формируется финальный обучающий датасет 
    <code>ml_training_dataset.csv</code> как декартово произведение пользователей, офферов 
    и контекста, дополненное признаками взаимодействий (<code>clicked</code>).
  </li>

  <li>
    Обучается ML-модель (CatBoost/XGBoost), прогнозирующая вероятность клика на каждый оффер.
  </li>

  <li>
    Для пользователя вычисляется <code>p_click</code> по всем офферам, после чего они 
    ранжируются и выбирается лучший (или топ-N).
  </li>

  <li>
    Генерируется текстовый профиль пользователя — краткое описание его поведения и 
    предпочтений на естественном языке.
  </li>

  <li>
    LLM (ЯндексGPT или GigaChat) получает:
    <ul>
      <li>текстовый профиль пользователя,</li>
      <li>данные оффера,</li>
      <li>шаблон промпта для выбранного канала (push / email / SMS).</li>
    </ul>
    На выходе генерируется персонализированное маркетинговое сообщение.
  </li>

  <li>
    В демонстрационном ноутбуке для заданного <code>user_id</code> система выдаёт:
    <ul>
      <li>лучший оффер,</li>
      <li>его вероятность клика <code>p_click</code>,</li>
      <li>сгенерированное LLM-сообщение.</li>
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
    Последовательно пройти ноутбуки в каталоге <code>notebooks/</code>:
    <ul>
      <li><b>01_eda.ipynb</b>: EDA и анализ данных</li>
      <li><b>02_build_training_dataset.ipynb</b>: формирование датасета для ML</li>
      <li><b>03_train_ranking_model.ipynb</b>: обучение ML-ранжировщика</li>
      <li><b>04_user_profile_builder.ipynb</b>: генерация текстового профиля пользователя</li>
      <li><b>05_llm_generation.ipynb</b>: генерация LLM-сообщений</li>
      <li><b>06_nbo_llm_demo.ipynb</b>: демонстрация NBO (user → best offer → LLM message)</li>
    </ul>
  </li>
</ol>

  <h2>6. Результат</h2>

  <ul>
    <li>обученная модель ранжирования офферов;</li>
    <li>компонент генерации пользовательского профиля;</li>
    <li>LLM-модуль для персонализированных сообщений;</li>
    <li>демо-ноутбук, показывающий полный ML → LLM пайплайн;</li>
    <li>структурированный отчёт для аналитической записки.</li>
  </ul>

</body>
