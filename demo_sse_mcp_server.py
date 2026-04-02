from fastapi import FastAPI
from fastapi_mcp import FastApiMCP
from datetime import datetime

# 初始化FastAPI应用
app = FastAPI(title="AI计算服务", description="提供基础数学计算、时间查询和天气模拟功能")

# 数学运算端点
@app.get("/add", operation_id="add_numbers")
def add_numbers(a: int, b: int):
    """计算两个整数的和"""
    return {"result": a + b}

@app.get("/multiply", operation_id="multiply_numbers")
def multiply_numbers(a: int, b: int):
    """计算两个整数的乘积"""
    return {"result": a * b}

# 时间获取端点
@app.get("/datetime", operation_id="date_time")
def date_time():
    """获取当前系统时间"""
    return {"datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

@app.post("/weather", operation_id="get_weather")
def get_weather(city_name: str):
    """获取指定城市天气信息"""
    return {
        "city": city_name,
        "temperature": "23℃"
    }

# 初始化MCP服务器
mcp = FastApiMCP(
    app,
    name="Math & Weather API",
    description="提供数学运算和天气查询服务的MCP接口",
    #base_url="http://localhost:8000",
    include_operations=[
        "add_numbers",
        "multiply_numbers",
        "date_time",
        "get_weather"
    ]
)
mcp.mount()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
