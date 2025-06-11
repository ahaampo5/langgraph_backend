import requests
import re
import json
import os

serving_id = 304
bearer_token = os.getenv("GENOS_BEARER_TOKEN")
genos_url = 'https://genos.mnc.ai:3443/'

url = f"{genos_url}/api/gateway/rep/serving/{serving_id}"
headers = dict(Authorization=f"Bearer {bearer_token}")
endpoint = f"{url}/v1/models"

# res = requests.get(endpoint, headers=headers)
# print(res.json())

body = {
    'model': 'qwen3-fp8',
    'messages': [
        {'role': 'system', 'content': '당신은 유용한 어시스턴트입니다.'},
        {'role': 'user', 'content': '안녕하세요, 잘 지내고 있나요?'}
    ]
}

endpoint = f"{url}/v1/chat/completions"
res = requests.post(endpoint, headers=headers, json=body)
print(res.json())

import wikienv, wrappers
env = wikienv.WikiEnv()
env = wrappers.HotPotQAWrapper(env, split="dev")
env = wrappers.LoggingWrapper(env)

def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1

def llm(prompt, stop=['\n']):
    body = {
        'model': 'qwen3-fp8',
        'messages': [{'role': 'user', 'content': prompt}],
        'stop': stop
    }
    endpoint = f"{url}/v1/chat/completions"
    res = requests.post(endpoint, headers=headers, json=body)
    if res.status_code != 200:
        raise Exception(f"LLM request failed: {res.text}")
    return res.json()['choices'][0]['message']['content']


import json
import sys

folder = './prompts/'
prompt_file = 'prompts_naive.json'
with open(folder + prompt_file, 'r') as f:
    prompt_dict = json.load(f)

webthink_examples = prompt_dict['webthink_simple6']
instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.
(2) Lookup[keyword], which returns the next sentence containing keyword in the current passage.
(3) Finish[answer], which returns the answer and finishes the task.
Here are some examples.
"""
webthink_prompt = instruction + webthink_examples

def webthink(idx=None, prompt=webthink_prompt, to_print=True):
    question = env.reset(idx=idx)
    if to_print:
        print(idx, question)
    prompt += question + "\n"
    n_calls, n_badcalls = 0, 0
    for i in range(1, 8):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            print('ohh...', thought_action)
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=[f"\n"]).strip()
        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')
        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str
        if to_print:
            print(step_str)
        if done:
            break
    if not done:
        obs, r, done, info = step(env, "finish[]")
    if to_print:
        print(info, '\n')
    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info

if __name__ == "__main__":
    import random
    import time
    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()
    for i in idxs[:500]:
        r, info = webthink(i, to_print=True)
        rs.append(info['em'])
        infos.append(info)
        print(sum(rs), len(rs), sum(rs) / len(rs), (time.time() - old_time) / len(rs))
        print('-----------')
        print()