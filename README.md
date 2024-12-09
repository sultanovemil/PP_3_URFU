# Проектный практикум 3. Учебная задача


**Команда №1**
- Алексеева А.А.
- Осипов Н.В.
- Сокирка А.В.
- Султанов Э.М.
- Шибакова А.А.

**Краткое описание:**

Создать нейронную сеть, способную генерировать текстовые отзывы о различных местах на основе определенных входных параметров, таких как категория места, средний рейтинг и ключевые слова.

Данные: https://github.com/yandex/geo-reviews-dataset-2023


**Описание датасета**
- 500 000 уникальных отзывов
- Только отзывы на организации в России
- Доступны на Яндекс Картах
- Опубликованы с января по июль 2023 года
- Датасет не содержит коротких односложных отзывов
- Тексты очищены от персональных данных (номеров телефонов, адресов почты)

**Состав датасета**

Датасет в формате tskv содержит следующую информацию:

* Адрес организации (address)
* Название организации (name_ru)
* Список рубрик, к которым относится организация (rubrics)
* Оценка пользователя от 0 до 5 (rating)
* Текст отзыва (text)

Веб-приложение развернуто на Hugging Face Spaces и доступно по следующей [ссылке](https://huggingface.co/spaces/Emil25/PP3_Team_1)
![app](https://github.com/sultanovemil/PP_3_URFU/blob/main/img/app.png)
