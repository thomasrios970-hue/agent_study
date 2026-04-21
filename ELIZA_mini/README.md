## 构建基于规则的聊天机器人

通过一个具体的编程实践，来直观地感受基于规则的系统是如何工作的。将尝试复现人工智能历史上一个极具影响力的早期聊天机器人-ELIZA。

### 1 ELIZA的设计思想

通过一套预设的转化规则，将用户的陈述转化为一个开放式的提问。例如，当用户说“我为我的男朋友感到难过”时，ELIZA可能会识别出关键词“我为……感到难过”，并应用规则生成回应：“你为什么会为你的男朋友感到难过？”ELIZA作者想证明，通过一些简单的句式转换技巧，机器可以在完全不理解对话内容的情况下，营造出一种“智能”和共情的假象。

### 2 模式匹配与文本替换

算法流程可以分为以下四个步骤

#### （1）关键词识别与排序

在规则库中为每个关键词预设优先级，输入包含多个关键词时，选择优先级最高的规则处理

> 示例关键词：mother, dreamed, depressed

#### （2）分解规则

找到关键词后，程序用带通配符*的分解规则，捕获句子的其余部分

> 规则示例：* my *
> 输入："My mother is afraid of me"
> 捕获：["", "mother is afraid of me"]

#### （3）重组规则

从关联的重组规则中（可随机）选择一条生成回应，可选择性使用捕获的内容

> 规则示例： "Tell me more about your family."
>
> 生成输出： "Tell me more about your family."

#### （4）代词转换

在重组前进行代词转换（I → you，my→your），维持对话连贯性

> 示例："I feel sad" → "Why do you feel sad?"

整个流程用伪代码的思路如下所示：

```
FUNCTION generate_response(user_input):
    // 1. 将用户输入拆分成单词
    words = SPLIT(user_input)

    // 2. 寻找优先级最高的关键词规则
    best_rule = FIND_BEST_RULE(words)
    IF best_rule is NULL:
        RETURN a_generic_response() //	 例如:"Please go on."

    // 3. 使用规则分解用户输入
    decomposed_parts = DECOMPOSE(user_input, best_rule.decomposition_pattern)
    IF decomposition_failed:
        RETURN a_generic_response()

    // 4. 对分解出的部分进行代词转换
    transformed_parts = TRANSFORM_PRONOUNS(decomposed_parts)

    // 5. 使用重组规则生成回应
    response = REASSEMBLE(transformed_parts, best_rule.reassembly_patterns)

    RETURN response
```

### 3 python实现简单的ELIZA

```
import re
import random

# 定义规则库:模式(正则表达式) -> 响应模板列表
rules = {
    r'I need (.*)': [
        "Why do you need {0}?",
        "Would it really help you to get {0}?",
        "Are you sure you need {0}?"
    ],
    r'Why don\'t you (.*)\?': [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?"
    ],
    r'Why can\'t I (.*)\?': [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?"
    ],
    r'I am (.*)': [
        "Did you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you?"
    ],
    r'.*': [
        "Please tell me more.",
        "Let's change focus a bit... Tell me about your family.",
        "Can you elaborate on that?"
    ]
}

# 定义代词转换规则
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "yours": "mine",
    "mine": "yours"
}


def swap_pronouns(phrase):
    """
    对输入短语中的代词进行第一/第二人称转换
    """
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word) for word in words]
    return " ".join(swapped_words)


def respond(user_input):
    """
    根据规则库生成响应
    """
    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            # 捕获匹配到的部分
            captured_group = match.group(1) if match.groups() else ''
            # 进行代词转换
            swapped_group = swap_pronouns(captured_group)
            # 从模板中随机选择一个并格式化
            response = random.choice(responses).format(swapped_group)
            return response
    # 如果没有匹配任何特定规则，使用最后的通配符规则
    return random.choice(rules[r'.*'])


# 主聊天循环
if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")
```

局限性有缺乏语义理解，不理解语义，会机械的匹配规则生成不通的回应；无上下文记忆，每次回应仅基于单句，无法连贯多轮对话；

规则扩展性，增加规则导致规模爆炸、冲突管理复杂，系统难以维护。系统看似智能，实际完全依赖预设规则，面对真实世界语言的无限可能性，穷举方法注定不可拓展。

