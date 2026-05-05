from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def ask_llm(question, model="gpt-4o-mini", temperature=0.0, logprobs=True):
    
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Answer the question clearly and concisely."},
            {"role": "user", "content": question}
        ],
        "temperature": temperature,
    }

    if logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 2

    response = client.chat.completions.create(**kwargs)

    choice = response.choices[0]
    answer = choice.message.content

    token_logprobs = []
    token_top_logprobs = []

    if choice.logprobs and choice.logprobs.content:
        for token_info in choice.logprobs.content:
            token_logprobs.append(token_info.logprob)
            token_top_logprobs.append(token_info.top_logprobs)

    usage = response.usage

    return {
        "answer": answer,
        "token_logprobs": token_logprobs,
        "token_top_logprobs": token_top_logprobs,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }