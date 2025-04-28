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


'''
#If using Orca    
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
'''
# If using Llama2 ...

llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
)


@cl.on_message
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")

    msg = cl.Message(content="")
    await msg.send()

    if message.content == "use llama2":
        await cl.Message(content="Switch to LLAMA2 successful!").send()

        llm2 = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GGUF", model_file="llama-2-7b-chat.Q5_K_M.gguf"
        )

    elif message.content == "use orca":
        await cl.Message(content="Switch to ORCA successful!").send()

        llm2 = AutoModelForCausalLM.from_pretrained(
            "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
        )

    else:
        prompt = get_prompt(message.content, message_history)
        response = ""

        for word in llm2(prompt, stream=True):
            await msg.stream_token(word)
            response += word

            await msg.update()
        message_history.append(response)
