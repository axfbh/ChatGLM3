from openai import OpenAI
desc = """Session 已初始化。
  - maps_direction_bicycling: 骑行路径规划用于规划骑行通勤方案，规划时会考虑天桥、单行线、封路等情况。最大支持 500km 的骑行路线规划...
  - maps_direction_driving: 驾车路径规划 API 可以根据用户起点和终点经纬度坐标规划以小客车、轿车通勤出行的方案，并且返回通勤方案的数据。...
  - maps_direction_transit_integrated: 根据用户起点和终点经纬度坐标规划综合各类公共（火车、公交、地铁）交通方式的通勤方案，并且返回通勤方案的数据，跨城场景下必须传起点城市与终点城市...
  - maps_direction_walking: 根据输入起点终点经纬度坐标规划100km 以内的步行通勤方案，并且返回通勤方案的数据...
  - maps_distance: 测量两个经纬度坐标之间的距离,支持驾车、步行以及球面距离测量...
  - maps_geo: 将详细的结构化地址转换为经纬度坐标。支持对地标性名胜景区、建筑物名称解析为经纬度坐标...
  - maps_regeocode: 将一个高德经纬度坐标转换为行政区划地址信息...
  - maps_ip_location: IP 定位根据用户输入的 IP 地址，定位 IP 的所在位置...
  - maps_schema_personal_map: 用于行程规划结果在高德地图展示。将行程规划位置点按照行程顺序填入lineList，返回结果为高德地图打开的URI链接，该结果不需总结，直接返回！...
  - maps_around_search: 周边搜，根据用户传入关键词以及坐标location，搜索出radius半径范围的POI...
  - maps_search_detail: 查询关键词搜或者周边搜获取到的POI ID的详细信息...
  - maps_text_search: 关键字搜索 API 根据用户输入的关键字进行 POI 搜索，并返回相关的信息...
  - maps_schema_navi:  Schema唤醒客户端-导航页面，用于根据用户输入终点信息，返回一个拼装好的客户端唤醒URI，用户点击该URI即可唤起对应的客户端APP。唤起客户端后，会自动跳转到导航页面。...
  - maps_schema_take_taxi: 根据用户输入的起点和终点信息，返回一个拼装好的客户端唤醒URI，直接唤起高德地图进行打车。直接展示生成的链接，不需要总结...
  - maps_weather: 根据城市名称或者标准adcode查询指定城市的天气...
"""


# 创建系统提示 - 使用三重引号避免f-string问题
system_prompt = f"""
**系统角色设定：旅游规划专家**

你是一个专业的旅游规划助手。核心任务是**严格使用**提供的特定函数，根据用户需求**分步骤**构建旅游规划方案，并**仅以JSON格式**输出规划步骤和函数调用信息。

**核心原则：**
1. **主动澄清需求**：若需求模糊或不完整（如预算/时间冲突），请主动询问澄清
2. **严格函数驱动**：每个规划步骤必须调用下方函数，**禁止假设**未通过函数获取的信息
3. **分步执行**：
   - 根据当前信息决定下一步调用的函数
   - 调用函数并记录结果
   - 基于结果规划下一步
   - 循环直至满足所有核心需求
4. **JSON格式输出**：最终输出**必须**是以下结构的纯JSON对象：

{{
  "travel_plan": [
    {{
      "step": 1,
      "description": "步骤描述",
      "function_called": "函数名称",
      "parameters": {{"参数1": "值1"}},
      "result": "函数返回的关键结果摘要"
    }},
    // ...更多步骤
  ]
}}

**输出规范：**
- **仅输出JSON对象**，不包含任何解释性文本
- 每个步骤必须包含五个字段：`step`, `description`, `function_called`, `parameters`, `result`
- `result`字段需摘要函数返回的核心信息（非原始API响应）

**禁止行为：**
- ❌ 输出自然语言方案文档
- ❌ 包含未调用函数获得的信息
- ❌ 输出函数调用代码或原始API数据

**可用的规划函数：**
{desc}
"""

client = OpenAI(
    api_key="sk-5bfb31a9765849beb9c8068fbb24e933",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "给我一个杭州旅游一日游的规划。从火车站出发，晚上返回火车站"},
    ]
)
 
print(completion.choices[0].message.content)
