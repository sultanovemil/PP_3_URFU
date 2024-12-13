import streamlit as st
import pandas as pd


from transformers import AutoModelForCausalLM, AutoTokenizer


# Настройка конфигурации страницы Streamlit
st.set_page_config(
    page_title="Generate reviews",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("Генератор отзывов на основе ИИ")
st.write("Создайте уникальные текстовые отзывы о различных местах на основе категорий, рейтинга и ключевых слов.")
st.sidebar.title("Параметры генерации")

# Загрузка модели и токенизатора
# @st.cache_data()
@st.cache_resource
def get_model():
    # Загрузка модели
    model = AutoModelForCausalLM.from_pretrained('model')
    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained('model')
    return model, tokenizer


# Генерация отзыва
def gen_review(input_text, model, tokenizer, params):  
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(
        input_ids,
        max_length=params['max_length'],
        num_return_sequences=params['num_return_sequences'],
        no_repeat_ngram_size=params['no_repeat_ngram_size'],
        do_sample=params['do_sample'],
        top_p=params['top_p'],
        top_k=params['top_k'],
        temperature=params['temperature'],
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def capitalize_and_punctuate(text):
    # Извлекаем часть текста после последнего двоеточия
    text = text.split(":")[-1].strip()
    
    # Разделяем текст на предложения по общим знакам препинания
    sentences = []
    current_sentence = []
    for char in text:
        current_sentence.append(char)
        # Если встречаем знак конца предложения, добавляем его в список предложений
        if char in '.!?':
            sentences.append(''.join(current_sentence).strip())
            current_sentence = []
    
    # Если остался текст, добавляем его как последнее предложение
    if current_sentence:
        sentences.append(''.join(current_sentence).strip())

    # Обрабатываем каждое предложение, чтобы сделать первую букву заглавной
    corrected_sentences = []
    for sentence in sentences:
        if sentence:
            # Делаем первую букву заглавной и добавляем точку в конце, если её нет
            corrected_sentence = sentence[0].upper() + sentence[1:]
            if not corrected_sentence.endswith('.'):
                corrected_sentence += '.'
            corrected_sentences.append(corrected_sentence)

    # Объединяем все исправленные предложения в финальный текст
    final_text = ' '.join(corrected_sentences)
    return final_text


# Главная функция
def main():
    # Загружаем модель и токенизатор
    model, tokenizer = get_model()
    if 'btn_predict' not in st.session_state:
        st.session_state['btn_predict'] = False

    # Параметры генерации
    params = {}
    params['max_length'] = st.sidebar.slider('Максимальная длина', min_value=50, max_value=300, value=200)
    params['num_return_sequences'] = st.sidebar.number_input('Количество ответов', min_value=1, max_value=10, value=1)
    params['no_repeat_ngram_size'] = st.sidebar.number_input('Размер n-грамм без повтора', min_value=1, max_value=20, value=2)
    params['do_sample'] = st.sidebar.checkbox('Использовать случайную выборку', value=True)
    params['top_p'] = st.sidebar.slider('Вероятность отбора (Top-p)', min_value=0.01, max_value=1.00, step=0.05, value=0.95)
    params['top_k'] = st.sidebar.number_input('Топ-k отбор', min_value=1, max_value=100, value=60)
    params['temperature'] = st.sidebar.slider('Температура', min_value=0.01, max_value=2.00, step=0.05, value=0.90)

    category = st.text_input("Категория:", value="Кондитерская")
    rating = st.slider("Рейтинг", 1, 5, 1)
    key_words = st.text_input("Ключевые слова", value="десерт, торт, цена")

    # Ввод новых параметров
    input_text = f"Категория: {category}; Рейтинг: {rating}; Ключевые слова: {key_words} -> Отзыв:"

    if st.button('Generate'):
        with st.spinner('Генерация отзыва...'):
            generated_text = gen_review(input_text, model, tokenizer, params)
            generated_text = capitalize_and_punctuate(generated_text)        
        st.success("Готово!")
        st.text(generated_text)


if __name__ == "__main__":  
    main()
