from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.responses import JSONResponse, FileResponse
import os
import uvicorn
import logging
from pydantic import BaseModel
from typing import Dict, Any
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入现有的功能模块
# 从ai_case_chinese.py导入功能
try:
    from ai_case_chinese import read_yaml_file, GeminiClient, save_test_cases_to_file, \
        main as ai_case_main,ai_case_main_auto as ai_case_main_auto
    logger.info("成功导入ai_case_chinese模块")
except Exception as e:
    logger.error(f"导入ai_case_chinese模块失败: {e}")
    raise

# 从new_code.py导入功能
try:
    from new_code import (
        CodeGenerationConfig, CodeGenerationService, 
        load_test_cases, extract_class_name, save_generated_code, 
        get_config_from_env
    )
    logger.info("成功导入new_code模块")
except Exception as e:
    logger.error(f"导入new_code模块失败: {e}")
    raise

# 创建FastAPI应用
app = FastAPI(
    title="测试代码生成服务",
    description="整合Gemini和OpenAI的测试代码生成服务",
    version="1.0.0"
)

# 配置项
GEMINI_API_KEY = "AIzaSyDSLj4MAzc2NzewK-ScdbotULMREJ_8p2M"  # 从环境变量获取更好
OUTPUT_DIR = "case_generated"
JAVA_OUTPUT_DIR = "generated"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(JAVA_OUTPUT_DIR, exist_ok=True)

# 创建Gemini客户端实例（可以在请求时创建，但为了效率可以复用）
gemini_client = GeminiClient(api_key=GEMINI_API_KEY)

@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "message": "测试代码生成服务",
        "endpoints": {
            "/docs": "API文档",
            "/generate-test-cases":"生成测试用例",
            "/generate-java-code":"生成Java测试代码"
    }
    }



class GenerateTestCasesRequest(BaseModel):     #场景描述，用户信息，以及map data数据
    changjing: str
    user_info:str
    map_data: str

@app.post("/generate-test-cases")
async def generate_test_cases(req: GenerateTestCasesRequest):
    # return {
    #     "changjing": req.changjing,
    #     "map_data": req.map_data
    # }
    changjing=req.changjing
    map_data=req.map_data
    user_info=req.user_info
    print(req.user_info)
    try:
        logger.info(f"开始生成测试用例，场景: {changjing}")
        # 调用ai_case_chinese的main函数生成测试用例

        if "自动生成测试" in changjing:

            generated_test_cases, generated_test_cases_code = ai_case_main_auto(user_info,map_data)
        else:
            generated_test_cases, generated_test_cases_code = ai_case_main(changjing, user_info,map_data)

        # 构建响应结果
        results = []
        for interface_name, case_json in generated_test_cases.items():
            output_file = os.path.join(OUTPUT_DIR, f"{interface_name}.json")
            results.append({
                "interface": interface_name,
                "status": "success",
                "file": output_file
            })
        
        return JSONResponse({
            "status": "completed",
            "results": results,
            "total": len(results),
            "success": len(results),
            "test_cases": generated_test_cases,  # 返回完整的测试用例字典
            "generated_java_code": generated_test_cases_code  # 返回生成的Java代码字典
        })
        
    except Exception as e:
        logger.error(f"生成测试用例失败: {e}")
        raise HTTPException(status_code=500, detail=f"生成测试用例失败: {str(e)}")

if __name__ == "__main__":
    # 启动服务
    logger.info("启动FastAPI服务...")
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # 开发环境使用，生产环境应关闭
    )