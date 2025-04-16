from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)
'''
prompt = "My dog's name is"
print(llm(prompt))
'''

sentence = "Indonesia's capital is called "
prompt = "Please give a one word answer and then stop. The capital of Indonesia is"
print(sentence + llm(prompt))
