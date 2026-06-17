-- Category taxonomy (справочник) + learnable merchant→category rules.
--
-- Phase A categorizer rework: the taxonomy and merchant mappings move OUT of
-- code INTO extensible reference tables.
--   * `category`               — справочник, рендерится в промпт категоризатора.
--                                 enum `Category` остаётся каноном (type-safety,
--                                 evals, парсеры) — это таблица-зеркало для промпта.
--   * `merchant_category_rule` — каскад «узнать без LLM»: нормализованный продавец
--                                 → категория. fuzzy-поиск через pg_trgm
--                                 (word_similarity). Подтверждённые пользователем
--                                 ответы дописываются сюда (source='user') —
--                                 система учится на правках (learning loop).
--
-- Re-running безопасно: IF NOT EXISTS + ON CONFLICT DO NOTHING.

-- Триграммный fuzzy-матч продавцов (встроен в Postgres, отдельный образ не нужен).
CREATE EXTENSION IF NOT EXISTS pg_trgm;


-- === Справочник категорий ===

CREATE TABLE IF NOT EXISTS category (
    code         TEXT PRIMARY KEY,          -- совпадает со значением enum Category
    group_name   TEXT NOT NULL,
    description  TEXT NOT NULL,             -- человекочитаемое описание для промпта
    active       BOOLEAN NOT NULL DEFAULT TRUE,
    sort_order   INT NOT NULL DEFAULT 0
);

INSERT INTO category (code, group_name, description, sort_order) VALUES
    ('food.groceries',              'Еда',          'супермаркеты (Пятёрочка, Магнит, ВкусВилл, Перекрёсток)', 10),
    ('food.restaurant',             'Еда',          'рестораны, кафе, столовые, фастфуд', 11),
    ('food.delivery',               'Еда',          'доставка еды (Самокат, Яндекс.Еда, СберМаркет, Delivery)', 12),
    ('food.coffee',                 'Еда',          'кофейни, кофе с собой (Cofix, Starbucks, Кофе Хауз)', 13),
    ('transport.fuel',              'Транспорт',    'АЗС (Лукойл, Роснефть, Газпром нефть)', 20),
    ('transport.taxi',              'Транспорт',    'Яндекс.Такси, Ситимобил, DiDi', 21),
    ('transport.public',            'Транспорт',    'метро, автобус, Аэроэкспресс, электричка', 22),
    ('transport.carparts',          'Транспорт',    'запчасти, шиномонтаж, автосервис', 23),
    ('transport.carshare',          'Транспорт',    'каршеринг (Яндекс.Драйв, Делимобиль, BelkaCar)', 24),
    ('kids.clothes',                'Дети',         'детская одежда и обувь (Детский мир)', 30),
    ('kids.toys',                   'Дети',         'игрушки, конструкторы', 31),
    ('kids.school',                 'Дети',         'школьные принадлежности, учебники, канцелярия', 32),
    ('kids.activities',             'Дети',         'секции, кружки, репетиторы, развивающие курсы', 33),
    ('shopping.clothes',            'Покупки',      'одежда и обувь для взрослых (Wildberries, ZARA, H&M, Lamoda)', 40),
    ('shopping.generic',            'Покупки',      'прочие покупки (Ozon, AliExpress, маркетплейсы)', 41),
    ('home.utilities',              'Дом',          'ЖКХ, коммуналка, электричество, вода, отопление', 50),
    ('home.telecom',                'Дом',          'мобильная связь, интернет, ТВ (МТС, Билайн, Мегафон, Ростелеком)', 51),
    ('home.rent',                   'Дом',          'аренда жилья, съёмная квартира', 52),
    ('home.furniture',              'Дом',          'мебель (IKEA, Hoff, Lazurit)', 53),
    ('home.repair',                 'Дом',          'ремонт, стройматериалы (Леруа Мерлен, OBI, СТД Петрович)', 54),
    ('home.household',              'Дом',          'бытовая химия, хозтовары, уборка', 55),
    ('health.pharmacy',             'Здоровье',     'аптеки (36.6, АСНА, Ригла)', 60),
    ('health.generic',              'Здоровье',     'медицина без подтверждённого вычета: ОМС, БАДы, медтехника, оптика', 61),
    ('health.fitness',              'Здоровье',     'фитнес, спортзал, бассейн (без налогового вычета)', 62),
    ('entertainment.subscriptions', 'Развлечения',  'Яндекс.Плюс, Netflix, Spotify, ChatGPT Plus, подписки', 70),
    ('entertainment.events',        'Развлечения',  'кино, театр, концерты, экскурсии', 71),
    ('entertainment.hobbies',       'Развлечения',  'спорттовары, хобби, книги', 72),
    ('entertainment.games',         'Развлечения',  'видеоигры, Steam, PlayStation, внутриигровые покупки', 73),
    ('education.courses',           'Образование',  'курсы, онлайн-обучение, книги для взрослых (Skillbox, Coursera) без вычета', 75),
    ('pets',                        'Питомцы',      'ветеринария, зоотовары (Зоомагазин, ВетМир)', 80),
    ('beauty.care',                 'Красота',      'салоны красоты, парикмахерские, барбершоп, косметика, уход', 82),
    ('travel.tickets',              'Путешествия',  'авиа- и ж/д билеты, путешествия', 84),
    ('travel.lodging',              'Путешествия',  'отели, апартаменты, гостиницы, Airbnb', 85),
    ('gifts',                       'Подарки',      'подарки, цветы', 86),
    ('charity',                     'Благотворительность', 'благотворительность, донаты, пожертвования', 87),
    ('government.fees',             'Госплатежи',   'налоги, штрафы ГИБДД, госпошлины, госуслуги (НЕ вычеты)', 88),
    ('tax_ded.medical',             'Налоги',       'платная медицина с правом вычета: клиники, анализы (Инвитро, Гемотест, Медси, КДЛ), стоматология, ДМС (ст.219 НК РФ)', 90),
    ('tax_ded.education',           'Налоги',       'платное образование с лицензией: вузы, автошколы, лицензированные курсы (ст.219 НК РФ)', 91),
    ('tax_ded.sport',               'Налоги',       'фитнес/спорт с правом вычета (ст.219 НК РФ, с 2022)', 92),
    ('tax_ded.iis',                 'Налоги',       'пополнение ИИС, право на вычет (ст.219.1 НК РФ)', 93),
    ('tax_ded.property',            'Налоги',       'покупка жилья, имущественный вычет (ст.220 НК РФ)', 94),
    ('finance.fees',                'Финансы',      'банковские комиссии, обслуживание карты, эквайринг', 95),
    ('finance.loan',                'Финансы',      'платежи по кредитам, ипотеке, рассрочке', 96),
    ('finance.insurance',           'Финансы',      'страхование (ОСАГО, КАСКО, ДМС, страховки)', 97),
    ('finance.cash',                'Финансы',      'снятие наличных в банкомате', 98),
    ('finance.investment',          'Финансы',      'брокер, акции, инвестиции (Тинькофф/ВТБ Инвестиции, не ИИС)', 99),
    ('income.salary',               'Доход',        'зарплата, аванс', 100),
    ('income.other',                'Доход',        'кэшбек, возвраты, прочие доходы', 101),
    ('transfer.internal',           'Спец',         'перевод между своими картами/счетами', 110),
    ('unclassified',                'Спец',         'категорию определить невозможно', 120)
ON CONFLICT (code) DO NOTHING;


-- === Правила «продавец → категория» (learning loop) ===

CREATE TABLE IF NOT EXISTS merchant_category_rule (
    rule_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- NULL = глобальный сид; конкретный family_id = выученное правило семьи.
    family_id        UUID REFERENCES family(family_id) ON DELETE CASCADE,
    merchant_norm    TEXT NOT NULL,                 -- нормализованный продавец (ключ поиска)
    merchant_sample  TEXT,                          -- оригинальный текст (для Phase B embeddings)
    category_code    TEXT NOT NULL REFERENCES category(code),
    source           TEXT NOT NULL DEFAULT 'seed'   -- seed | user | llm
        CHECK (source IN ('seed', 'user', 'llm')),
    hit_count        INT NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- PG17 NULLS NOT DISTINCT: глобальные сиды (family_id NULL) тоже уникальны.
    UNIQUE NULLS NOT DISTINCT (family_id, merchant_norm)
);

-- GIN-индекс для word_similarity-поиска (<%) при росте таблицы.
CREATE INDEX IF NOT EXISTS idx_merchant_rule_trgm
    ON merchant_category_rule USING gin (merchant_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_merchant_rule_family
    ON merchant_category_rule(family_id);

-- Глобальный сид известных продавцов (был захардкожен в промпте/парсере).
INSERT INTO merchant_category_rule (family_id, merchant_norm, category_code, source) VALUES
    (NULL, 'пятерочка',     'food.groceries',              'seed'),
    (NULL, 'магнит',        'food.groceries',              'seed'),
    (NULL, 'вкусвилл',      'food.groceries',              'seed'),
    (NULL, 'перекресток',   'food.groceries',              'seed'),
    (NULL, 'самокат',       'food.delivery',               'seed'),
    (NULL, 'яндекс еда',    'food.delivery',               'seed'),
    (NULL, 'сбермаркет',    'food.delivery',               'seed'),
    (NULL, 'лукойл',        'transport.fuel',              'seed'),
    (NULL, 'роснефть',      'transport.fuel',              'seed'),
    (NULL, 'газпром нефть', 'transport.fuel',              'seed'),
    (NULL, 'яндекс такси',  'transport.taxi',              'seed'),
    (NULL, 'ситимобил',     'transport.taxi',              'seed'),
    -- Латинские алиасы Яндекса: выписки иногда печатают «YANDEX.TAXI», что не
    -- матчится на кириллические правила выше (разные алфавиты, word_similarity=0).
    (NULL, 'yandex taxi',   'transport.taxi',              'seed'),
    (NULL, 'yandex eda',    'food.delivery',               'seed'),
    (NULL, 'yandex drive',  'transport.carshare',          'seed'),
    (NULL, 'yandex plus',   'entertainment.subscriptions', 'seed'),
    (NULL, 'детский мир',   'kids.clothes',                'seed'),
    (NULL, 'wildberries',   'shopping.clothes',            'seed'),
    (NULL, 'lamoda',        'shopping.clothes',            'seed'),
    (NULL, 'ozon',          'shopping.generic',            'seed'),
    (NULL, 'aliexpress',    'shopping.generic',            'seed'),
    (NULL, 'ikea',          'home.furniture',              'seed'),
    (NULL, 'hoff',          'home.furniture',              'seed'),
    (NULL, 'леруа мерлен',  'home.repair',                 'seed'),
    (NULL, 'петрович',      'home.repair',                 'seed'),
    (NULL, 'асна',          'health.pharmacy',             'seed'),
    (NULL, 'ригла',         'health.pharmacy',             'seed'),
    (NULL, 'инвитро',       'tax_ded.medical',             'seed'),
    (NULL, 'гемотест',      'tax_ded.medical',             'seed'),
    (NULL, 'netflix',       'entertainment.subscriptions', 'seed'),
    (NULL, 'spotify',       'entertainment.subscriptions', 'seed'),
    (NULL, 'ivi',           'entertainment.subscriptions', 'seed'),
    (NULL, 'okko',          'entertainment.subscriptions', 'seed'),
    (NULL, 'кинопоиск',     'entertainment.subscriptions', 'seed'),
    (NULL, 'wink',          'entertainment.subscriptions', 'seed'),
    (NULL, 'kion',          'entertainment.subscriptions', 'seed'),
    (NULL, 'premier',       'entertainment.subscriptions', 'seed'),
    (NULL, 'амедиатека',    'entertainment.subscriptions', 'seed'),
    (NULL, 'megogo',        'entertainment.subscriptions', 'seed'),
    (NULL, 'яндекс плюс',   'entertainment.subscriptions', 'seed'),
    (NULL, 'яндекс музыка', 'entertainment.subscriptions', 'seed'),
    (NULL, 'vk музыка',     'entertainment.subscriptions', 'seed'),
    (NULL, 'сберпрайм',     'entertainment.subscriptions', 'seed'),
    (NULL, 'зоомагазин',    'pets',                        'seed'),
    (NULL, 'ветмир',        'pets',                        'seed'),
    (NULL, 'делимобиль',    'transport.carshare',          'seed'),
    (NULL, 'яндекс драйв',  'transport.carshare',          'seed'),
    (NULL, 'belkacar',      'transport.carshare',          'seed'),
    (NULL, 'steam',         'entertainment.games',         'seed'),
    (NULL, 'playstation',   'entertainment.games',         'seed'),
    (NULL, 'skillbox',      'education.courses',           'seed'),
    (NULL, 'aviasales',     'travel.tickets',              'seed'),
    (NULL, 'ржд',           'travel.tickets',              'seed'),
    (NULL, 'букинг',        'travel.lodging',              'seed'),
    (NULL, 'островок',      'travel.lodging',              'seed'),
    (NULL, 'золотое яблоко','beauty.care',                 'seed'),
    (NULL, 'летуаль',       'beauty.care',                 'seed'),
    (NULL, 'росгосстрах',   'finance.insurance',           'seed'),
    (NULL, 'ингосстрах',    'finance.insurance',           'seed'),
    (NULL, 'сберстрахование','finance.insurance',          'seed'),
    (NULL, 'starbucks',     'food.coffee',                 'seed'),
    (NULL, 'cofix',         'food.coffee',                 'seed'),
    (NULL, 'кофе хауз',     'food.coffee',                 'seed'),
    (NULL, 'мтс',           'home.telecom',                'seed'),
    (NULL, 'билайн',        'home.telecom',                'seed'),
    (NULL, 'мегафон',       'home.telecom',                'seed'),
    (NULL, 'теле2',         'home.telecom',                'seed'),
    (NULL, 'ростелеком',    'home.telecom',                'seed'),
    (NULL, 'world class',   'health.fitness',              'seed'),
    (NULL, 'фитнес хаус',   'health.fitness',              'seed'),
    (NULL, 'госуслуги',     'government.fees',             'seed'),
    (NULL, 'гибдд',         'government.fees',             'seed'),
    (NULL, 'тинькофф инвестиции', 'finance.investment',    'seed'),
    (NULL, 'втб инвестиции','finance.investment',          'seed'),
    (NULL, 'русфонд',       'charity',                     'seed'),
    (NULL, 'нужна помощь',  'charity',                     'seed')
ON CONFLICT (family_id, merchant_norm) DO NOTHING;

-- QA-09: расширение словаря топ-РФ-ритейлерами (удешевляет каскад + поднимает recall).
-- Здоровье/образование вынесены в QA-08 (решение tax_ded там же).
INSERT INTO merchant_category_rule (family_id, merchant_norm, category_code, source) VALUES
    -- Продукты
    (NULL, 'ашан',          'food.groceries',              'seed'),
    (NULL, 'лента',         'food.groceries',              'seed'),
    (NULL, 'дикси',         'food.groceries',              'seed'),
    (NULL, 'окей',          'food.groceries',              'seed'),
    (NULL, 'метро',         'food.groceries',              'seed'),
    (NULL, 'спар',          'food.groceries',              'seed'),
    (NULL, 'азбука вкуса',  'food.groceries',              'seed'),
    (NULL, 'мираторг',      'food.groceries',              'seed'),
    (NULL, 'верный',        'food.groceries',              'seed'),
    (NULL, 'монетка',       'food.groceries',              'seed'),
    (NULL, 'глобус',        'food.groceries',              'seed'),
    (NULL, 'магнолия',      'food.groceries',              'seed'),
    -- Доставка еды
    (NULL, 'деливери клаб', 'food.delivery',               'seed'),
    (NULL, 'купер',         'food.delivery',               'seed'),
    (NULL, 'яндекс лавка',  'food.delivery',               'seed'),
    -- Рестораны / фастфуд
    (NULL, 'додо пицца',    'food.restaurant',             'seed'),
    (NULL, 'kfc',           'food.restaurant',             'seed'),
    (NULL, 'ростикс',       'food.restaurant',             'seed'),
    (NULL, 'бургер кинг',   'food.restaurant',             'seed'),
    (NULL, 'вкусно и точка','food.restaurant',             'seed'),
    (NULL, 'теремок',       'food.restaurant',             'seed'),
    (NULL, 'сабвей',        'food.restaurant',             'seed'),
    (NULL, 'papa johns',    'food.restaurant',             'seed'),
    -- Кофе
    (NULL, 'шоколадница',   'food.coffee',                 'seed'),
    (NULL, 'даблби',        'food.coffee',                 'seed'),
    (NULL, 'surf coffee',   'food.coffee',                 'seed'),
    (NULL, 'one price coffee','food.coffee',               'seed'),
    (NULL, 'правда кофе',   'food.coffee',                 'seed'),
    -- АЗС
    (NULL, 'татнефть',      'transport.fuel',              'seed'),
    (NULL, 'башнефть',      'transport.fuel',              'seed'),
    (NULL, 'газпромнефть',  'transport.fuel',              'seed'),
    (NULL, 'shell',         'transport.fuel',              'seed'),
    (NULL, 'трасса',        'transport.fuel',              'seed'),
    -- Городской транспорт
    (NULL, 'тройка',        'transport.public',            'seed'),
    (NULL, 'аэроэкспресс',  'transport.public',            'seed'),
    (NULL, 'метрополитен',  'transport.public',            'seed'),
    (NULL, 'мосгортранс',   'transport.public',            'seed'),
    -- Маркетплейсы / электроника
    (NULL, 'яндекс маркет', 'shopping.generic',            'seed'),
    (NULL, 'мегамаркет',    'shopping.generic',            'seed'),
    (NULL, 'dns',           'shopping.generic',            'seed'),
    (NULL, 'ситилинк',      'shopping.generic',            'seed'),
    (NULL, 'эльдорадо',     'shopping.generic',            'seed'),
    (NULL, 'м видео',       'shopping.generic',            'seed'),
    -- Одежда / обувь
    (NULL, 'zara',          'shopping.clothes',            'seed'),
    (NULL, 'h m',           'shopping.clothes',            'seed'),
    (NULL, 'uniqlo',        'shopping.clothes',            'seed'),
    (NULL, 'gloria jeans',  'shopping.clothes',            'seed'),
    (NULL, 'befree',        'shopping.clothes',            'seed'),
    (NULL, 'остин',         'shopping.clothes',            'seed'),
    (NULL, 'твое',          'shopping.clothes',            'seed'),
    (NULL, 'kari',          'shopping.clothes',            'seed'),
    -- Мебель
    (NULL, 'много мебели',  'home.furniture',              'seed'),
    (NULL, 'лазурит',       'home.furniture',              'seed'),
    (NULL, 'столплит',      'home.furniture',              'seed'),
    -- Ремонт / стройматериалы
    (NULL, 'obi',           'home.repair',                 'seed'),
    (NULL, 'касторама',     'home.repair',                 'seed'),
    (NULL, 'максидом',      'home.repair',                 'seed'),
    (NULL, 'всеинструменты','home.repair',                 'seed'),
    -- Дрогери / хозтовары
    (NULL, 'fix price',     'home.household',              'seed'),
    (NULL, 'улыбка радуги', 'home.household',              'seed'),
    (NULL, 'рубль бум',     'home.household',              'seed'),
    -- Связь
    (NULL, 'yota',          'home.telecom',                'seed'),
    (NULL, 'дом ру',        'home.telecom',                'seed'),
    -- Игры
    (NULL, 'xbox',          'entertainment.games',         'seed'),
    (NULL, 'epic games',    'entertainment.games',         'seed'),
    (NULL, 'nintendo',      'entertainment.games',         'seed'),
    -- Красота / косметика
    (NULL, 'рив гош',       'beauty.care',                 'seed'),
    (NULL, 'подружка',      'beauty.care',                 'seed'),
    (NULL, 'магнит косметик','beauty.care',                'seed'),
    -- Питомцы
    (NULL, 'четыре лапы',   'pets',                        'seed'),
    (NULL, 'бетховен',      'pets',                        'seed'),
    (NULL, 'petshop',       'pets',                        'seed'),
    -- События / кино
    (NULL, 'синема парк',   'entertainment.events',        'seed'),
    (NULL, 'формула кино',  'entertainment.events',        'seed'),
    (NULL, 'кассир ру',     'entertainment.events',        'seed'),
    -- Спорттовары / книги / хобби
    (NULL, 'спортмастер',   'entertainment.hobbies',       'seed'),
    (NULL, 'декатлон',      'entertainment.hobbies',       'seed'),
    (NULL, 'читай город',   'entertainment.hobbies',       'seed'),
    (NULL, 'лабиринт',      'entertainment.hobbies',       'seed'),
    (NULL, 'буквоед',       'entertainment.hobbies',       'seed')
ON CONFLICT (family_id, merchant_norm) DO NOTHING;

-- QA-08: вычитаемые траты (ст.219 НК РФ). Платная медицина (клиники/лаборатории/
-- стоматология) и лицензированное образование → tax_ded.*, чтобы tax_node их видел.
-- Аптеки НЕ сюда: лекарства вычитаемы только по рецепту → остаются health.pharmacy
-- (добавлены ниже для recall). Skillbox/Coursera — коммерческие, без лицензии →
-- остаются education.courses.
INSERT INTO merchant_category_rule (family_id, merchant_norm, category_code, source) VALUES
    -- Платная медицина с правом вычета
    (NULL, 'медси',         'tax_ded.medical',             'seed'),
    (NULL, 'кдл',           'tax_ded.medical',             'seed'),
    (NULL, 'ситилаб',       'tax_ded.medical',             'seed'),
    (NULL, 'смд',           'tax_ded.medical',             'seed'),
    (NULL, 'см клиника',    'tax_ded.medical',             'seed'),
    (NULL, 'мать и дитя',   'tax_ded.medical',             'seed'),
    (NULL, 'он клиник',     'tax_ded.medical',             'seed'),
    (NULL, 'ниармедик',     'tax_ded.medical',             'seed'),
    (NULL, 'доктор рядом',  'tax_ded.medical',             'seed'),
    (NULL, 'стоматология',  'tax_ded.medical',             'seed'),
    -- Латинские алиасы лабораторий/клиник (выписки печатают транслит, кириллица
    -- даёт word_similarity=0 — та же проблема, что у YANDEX.TAXI).
    (NULL, 'invitro',       'tax_ded.medical',             'seed'),
    (NULL, 'gemotest',      'tax_ded.medical',             'seed'),
    (NULL, 'medsi',         'tax_ded.medical',             'seed'),
    (NULL, 'kdl',           'tax_ded.medical',             'seed'),
    (NULL, 'citilab',       'tax_ded.medical',             'seed'),
    (NULL, 'stomatolog',    'tax_ded.medical',             'seed'),
    (NULL, 'dental',        'tax_ded.medical',             'seed'),
    (NULL, 'profi dent',    'tax_ded.medical',             'seed'),
    -- Лицензированное образование с правом вычета
    (NULL, 'автошкола',     'tax_ded.education',           'seed'),
    -- Аптеки (recall) — без вычета, остаются health.pharmacy
    (NULL, 'аптека 36 6',   'health.pharmacy',             'seed'),
    (NULL, 'горздрав',      'health.pharmacy',             'seed'),
    (NULL, 'столички',      'health.pharmacy',             'seed'),
    (NULL, 'апрель',        'health.pharmacy',             'seed'),
    (NULL, 'eapteka',       'health.pharmacy',             'seed'),
    (NULL, 'планета здоровья','health.pharmacy',           'seed'),
    (NULL, 'нео фарм',      'health.pharmacy',             'seed'),
    (NULL, 'фармленд',      'health.pharmacy',             'seed')
ON CONFLICT (family_id, merchant_norm) DO NOTHING;
