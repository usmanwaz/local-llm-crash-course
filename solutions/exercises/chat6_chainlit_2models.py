from ctransformers import AutoModelForCausalLM
from typing import List
import chainlit as cl


def get_prompt(instruction: str, history: List[str]) -> str:
    system = "You are an AI assistant that gives helpful answers. You answer the question in a short and concise way."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history so far: {''.join(history)} Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    # print(prompt)
    return prompt


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm


llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
)


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    global llm

    msg = cl.Message(content="")
    await msg.send()

    if message.content.lower() == "use llama2":
        await cl.Message(content="Switch to LLAMA2 successful!").send()

        llm = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
        )

    elif message.content.lower() == "use orca":
        await cl.Message(content="Switch to ORCA successful!").send()

        llm = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        )

    elif message.content.lower() == "forget":
        await cl.Message(content="Uh oh! I have forgotten everything").send()
        cl.user_session.set("message_history", [])

    else:
        prompt = get_prompt(message.content, message_history)
        response = ""

        for word in llm(prompt, stream=True):
            await msg.stream_token(word)
            response += word

            await msg.update()
        message_history.append(response)
