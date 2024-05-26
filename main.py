from openai import OpenAI
import math
import tiktoken
from config import OPENAI_API_KEY, MAIN_PROMPT, USER_DATA, ANSWER_PROMPT, AI_CHECKS, MAX_TOKENS, MAX_SYMBOLS

client = OpenAI()

# Функция для подсчета токенов


def count_tokens(prompt, model="gpt-4o"):
    # Получение кодировщика для указанной модели
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(prompt)
    return len(tokens)


# Функция для получения ответа от ChatGPT
def get_chatgpt_response(user_prompt: str, main_prompt: str) -> tuple:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": main_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content, response.usage.total_tokens


if __name__ == "__main__":

    spent_tokens = 0

    token_count = count_tokens(USER_DATA + ANSWER_PROMPT)
    print(
        f"В запросе: {USER_DATA + ANSWER_PROMPT} использовано {token_count} токенов ({round(token_count*100/30000,2)}% TPM tier1)")

    response_ready = False

    while not response_ready and spent_tokens < MAX_TOKENS:
        response, total_tokens = get_chatgpt_response(
            USER_DATA + ANSWER_PROMPT, MAIN_PROMPT)
        spent_tokens += total_tokens

        print("", "", "Ответ ChatGPT:", response, "", "",
              "Количество символов:", len(response), "", sep='\n')
        print(
            f"Общее количество использованных токенов (включая ответ):{spent_tokens} ({round(spent_tokens*100/30000,2)}% TPM tier1)", '', sep='\n')

        response_ready = True

        if len(response) > MAX_SYMBOLS + 50:
            response_ready = False
        else:
            for check_name, check_description in AI_CHECKS:
                check_response, total_tokens = get_chatgpt_response(
                    USER_DATA + "Совет пользователю: " + response + check_description, MAIN_PROMPT)
                spent_tokens += total_tokens
                if '0' in check_response:
                    print(
                        f"Проверка {check_name} – FAIL: {check_response}. Потрачено токенов {spent_tokens} ({round(spent_tokens*100/30000,2)}% TPM tier1)")
                    response_ready = False
                    break
                else:
                    print(
                        f"Проверка {check_name} – ОК: {check_response}. Потрачено токенов {spent_tokens} ({round(spent_tokens*100/30000,2)}% TPM tier1)")
