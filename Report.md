# Отчёт по лабораторной работе №3

Выполнили:

Студент (ФИО) | Роль в проекте   | Оценка
-------------|---------------------|------
Орусский Вячеслав Русланович | обучал Simple RNN с посимвольной и по-словной токенизацией | TBD
Петров Илья Олегович | Обучал однонаправленную однослойную и многослойную LSTM c посимвольной токенизацией и токенизацией по словам и на основе BPE | TBD
Карнаков Никита Дмитриевич | Отчет, обучал двунаправленную LSTM | TBD

В данном отчете будем приводить описание каждой из модели (процесс обучения, фрагменты кода и т.д.)

## LSTM с посимвольной токенизацией

### Предварительная обработка данных 

Из файла [data2.txt](data2.txt) извлекаются строки, игнорируются заголовки, такие как "Глава", и объединяются в один текстовый массив.

```python
raw = open('data2.txt', mode='r', encoding='utf-8').readlines()
data = []
for line in raw:
    if line != '\n' and 'Глава' not in line:
        data.append(' '.join(line.split()[1:]))
    
data = [line.replace('\n', ' ').replace('\xa0', ' ') for line in data]
text = ' '.join(data)
```
Пример:

```python
text[:100]
```

```plaintext
завет ЗАВЕТ Моисея  В начале сотворил Бог небо и землю. Земля же была безвидна и пуста, и тьма над б
```

В коде строится алфавит всех уникальных символов, присутствующих в тексте, и каждому символу присваивается индекс. Токенизация происходит на уровне символов, что позволяет модели работать с текстом по букве.

```python
def get_features_target(seq):
    features = seq[:-1]
    target = seq[1:]
    return features, target

BATCH_SIZE = 128

alphabet = np.array(sorted(set(text)))
sym_to_idx = {}
idx_to_sym = {}

for idx, sym in enumerate(alphabet):
    sym_to_idx[sym] = idx
    idx_to_sym[idx] = sym
    
text_idx = np.array([sym_to_idx[char] for char in text])
```

### Создание обучающих данных

Текст превращается в последовательность индексов символов. Эти последовательности подаются в нейронную сеть с помощью пакетов (батчей).

```python
sequences = Dataset.from_tensor_slices(text_idx).batch(BATCH_SIZE, drop_remainder=True)
dataset = sequences.map(get_features_target)
data = dataset.batch(BATCH_SIZE, drop_remainder=True).repeat()
data = data.prefetch(AUTOTUNE)
```
### Архитектура модели

Модель состоит из следующих слоёв:

* **Embedding слой**: Преобразует каждый символ в векторное представление;
* **LSTM слой**: Используется для обработки последовательностей и запоминания контекста;
* **Dense слой**: Выдаёт вероятности для каждого символа алфавита на выходе.

```python
model = keras.Sequential([
    keras.layers.Embedding(len(alphabet), 256),
    keras.layers.LSTM(512, return_sequences=True, stateful=True),
    keras.layers.Dense(len(alphabet))
])
```

### Компиляция и обучение модели

Модель компилируется с использованием стандартного оптимизатора `Adam` и функции потерь `SparseCategoricalCrossentropy`, подходящей для многоклассовой классификации.

```python
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(data, epochs=40, verbose=1, steps_per_epoch= len(sequences) // BATCH_SIZE)
```

### Генерация текста

Генерация текста происходит с помощью функции `predict_next`. Модель принимает начальную последовательность символов и предсказывает следующий символ, повторяя процесс до получения полной последовательности.

```python
def predict_next(sample, model, tokenizer, vocabulary, n_next, rnd_power, batch_size):
    sample_token = [tokenizer[char] for char in sample]
    predicted = sample_token

    sample_tensor = tf.expand_dims(sample_token, 0)
    sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)
    
    for _ in range(n_next):
        cur = model(sample_tensor)
        cur = cur[0].numpy() / rnd_power
        cur = tf.random.categorical(cur, num_samples=1)[-1, 0].numpy()
        predicted.append(cur)
        sample_tensor = predicted[-99:]
        sample_tensor = tf.expand_dims([cur], 0)
        sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)
    res = [vocabulary[i] for i in predicted]
    generated = ''.join(res)
    return generated
```

### Примеры предсказаний и анализ

**Пример 1 (с температурой 0.6)**

```python
print(predict_next(
    sample='б',
    model=model,
    tokenizer=sym_to_idx,
    vocabulary=idx_to_sym,
    n_next=200,
    temperature=0.6,
    batch_size=BATCH_SIZE
))
```
На выходе получаем следующее:

```plaintext
бом и селениями своими, ибо время согрешающего совершенного скота, о котором сказано: верен Бог их и умер, и освящающих нас, что делается всё, что значит им слово Божие, которое я имею тельцов и не дол
```

Этот пример показывает, что при температуре 0.6 текст получается разнообразным, но временами нарушается логика предложений. Модель способна генерировать осмысленные фразы, однако часть текста может не иметь связного смысла.

**Пример 2 (с температурой 0.2)**

```python
print(predict_next(
    sample='1',
    model=model,
    tokenizer=sym_to_idx,
    vocabulary=idx_to_sym,
    n_next=100,
    temperature=0.2,
    batch_size=BATCH_SIZE
))
```
На выходе получаем следующее:

```plaintext
1гая в сердцах своих и принятие в слове своём и сказали ему: вот, я возлюбленному своему и служению в
```

Здесь текст выглядит более предсказуемым, что объясняется низкой температурой (0.2). Однако начальный символ "1" явно выбивается из общего контекста. В то же время фразы более структурированы, хотя разнообразие предсказаний ниже.

### Выводы

Несмотря на работу с символами, модель способна генерировать текст, который напоминает исходные данные (религиозный текст). При этом, чем выше температура, появляются более разнообразные, но менее осмысленные фразы. При генерации длинных последовательностей иногда теряется связность между предложениями, что характерно для моделей, работающих на уровне символов.

## LSTM с пословной токенизацией

### Предварительная обработка данных

Модель работает с текстовыми данными, считанными из файла data2.txt. Текст предварительно очищается, фильтруются лишние символы (оставляются только кириллические символы, цифры и знаки препинания).

```python
raw = open('data2.txt', mode='r', encoding='utf-8').readlines()
data = []
for line in raw:
    if line != '\n' and 'Глава' not in line:
        data.append(' '.join(line.split()[1:]))
    
data = [line.replace('\n', ' ').replace('\xa0', ' ') for line in data]
text = ' '.join(data)
```

Здесь строки текста очищаются от переносов и пробелов, как и в первой модели. Однако есть дополнительный этап очистки текста с помощью регулярного выражения

```python
word_pattern = r'[^а-яА-ЯёЁ0-9 :,-]'
clear_text = re.sub(word_pattern, '', text)
```

Этот код удаляет все символы, которые не соответствуют указанному паттерну, оставляя только кириллические буквы, цифры, пробелы и знаки препинания.

### Токенизация слоев

В этой модели производится токенизация на уровне слов, а не символов, что позволяет модели лучше улавливать смысловые связи между словами.

```python
alphabet = np.array(sorted(set(clear_text.split(' '))))
clear_text = np.array(clear_text.split(' '))
clear_text = clear_text[clear_text != '']

word_to_idx = {}
idx_to_word = {}

for idx, word in enumerate(alphabet):
    word_to_idx[word] = idx
    idx_to_word[idx] = word
```

Здесь модель работает с последовательностями слов, что помогает лучше улавливать контекст по сравнению с символами.

### Создание обучающих данных

Данные преобразуются в последовательности индексов слов, которые подаются в модель через батчи с использованием библиотеки `TensorFlow Dataset`.

```python
text_idx = np.array([word_to_idx[word] for word in clear_text])
sequences = Dataset.from_tensor_slices(text_idx).batch(BATCH_SIZE, drop_remainder=True)
dataset = sequences.map(get_features_target)

data = dataset.batch(BATCH_SIZE, drop_remainder=True).repeat()
data = data.prefetch(AUTOTUNE)
```

### Архитектура модели

Архитектура модели включает три основных слоя:

* **Embedding слой**: Преобразует слова в векторное представление;
* **LSTM слой**: Используется для обработки последовательностей и запоминания контекста между словами;
* **Dense слой**: Выдаёт вероятности для каждого слова в словаре.

### Комплияция и обучение данных

Модель компилируется с оптимизатором **Adam** и функцией потерь `SparseCategoricalCrossentropy`. Она обучается на всех батчах данных в течение 20 эпох.

```python
model.compile(optimizer='adam', loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(data, epochs=20, verbose=1, steps_per_epoch=len(sequences) // BATCH_SIZE)
```

`SparseCategoricalCrossentropy` — функция потерь, которая подходит для задачи многоклассовой классификации, где каждый класс соответствует одному слову.

### Генерация текста

Текст генерируется с помощью функции `predict_next`, которая принимает начальную последовательность слов и предсказывает следующее слово на основе вероятностей, выданных моделью. Процесс повторяется, чтобы сгенерировать заданное количество слов.

```python
def predict_next(sample, model, tokenizer, vocabulary, n_next, rnd_power, batch_size):
    sample_token = [tokenizer[char] for char in sample.split()]
    predicted = sample_token

    sample_tensor = tf.expand_dims(sample_token, 0)
    sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)
    
    for _ in range(n_next):
        cur = model(sample_tensor)
        cur = cur[0].numpy() / rnd_power
        cur = tf.random.categorical(cur, num_samples=1)[-1, 0].numpy()
        predicted.append(cur)
        sample_tensor = predicted[-99:]
        sample_tensor = tf.expand_dims([cur], 0)
        sample_tensor = tf.repeat(sample_tensor, batch_size, axis=0)
    res = [vocabulary[i] for i in predicted]
    generated = ' '.join(res)
    return generated
```

### Примеры предсказаний и анализ

**Пример 1 с температурой 0.6**

```plaintext
бог его и хула из земли и преславного подчиниться останься в любезно, что длина их в Самом Каину а на земле
```

При генерации текстов с начальным словом "бог" видно, что модель может генерировать осмысленные предложения, но логика иногда сбивается (например, "подчиниться останься в любезно").

**Пример 2 с температурой 0.2**

Начало с цифры "1" также приводит к генерации относительно осмысленного текста, но с частичной утратой связности (например, "когда в потеряв который на него").

### Выводы

Модель генерирует текст на уровне слов. Это улучшает её способность создавать более осмысленные фразы по сравнению с моделью, работающей на уровне символов. Она лучше улавливает контексты между словами, что важно для смысловой целостности текста. Как и в первой модели, при повышении температуры генерации текст становится более разнообразным, но теряет связность. Низкие значения температуры делают текст более предсказуемым, но менее разнообразным. Благодаря токенизации на уровне слов, модель создаёт более логичные и связные предложения, хотя ещё встречаются некоторые ошибки и нарушения логики. Модель, хотя и улавливает общие связи между словами, не всегда правильно строит грамматически верные предложения.
