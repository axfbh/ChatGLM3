from dis import opname
import json


f_out = open("./data/conversation_dataset.jsonl", "w", encoding="utf-8")

with open("./data/E-commerce dataset/dev.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        line = line.strip().replace(" ", "").split("\t")[1:]
        contents = []
        for ids, content in enumerate(line):
            if ids % 2 == 0:
                contents.append({"role": "user", "content": content})
            else:
                contents.append({"role": "assistant", "content": content})

        conversations = contents

        json_line = json.dumps(
            {"conversations": conversations},
            ensure_ascii=False
        )
        f_out.write(json_line+"\n")
f_out.close()
