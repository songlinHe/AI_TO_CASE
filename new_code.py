import json
import os
import logging
import time
import dataclasses
import concurrent.futures
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import yaml
from openai import OpenAI, OpenAIError, RateLimitError, APIConnectionError, APIError

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('code_generation.log', encoding='UTF-8')
    ]
)
logger = logging.getLogger(__name__)
GLOBAL_CONFIG = {
    "output_dir": None
}


@dataclasses.dataclass
class CodeGenerationConfig:
    """代码生成服务配置类"""
    api_key: str
    base_url: str
    model: str = "deepseek-chat"
    temperature: float = 0.6
    max_retries: int = 3
    retry_delay: int = 2
    max_concurrent_requests: int = 2  # 添加并发请求数量配置


class CodeGenerationService:
    """代码生成服务类，封装与OpenAI API交互的逻辑"""

    def __init__(self, config: CodeGenerationConfig):
        """
        初始化代码生成服务

        Args:
            config: 代码生成服务配置
        """
        self.load_config()
        try:
            self.client = OpenAI(
                base_url=config.base_url,
                api_key=config.api_key
            )
            self.config = config
            logger.info("OpenAI客户端初始化成功")
        except Exception as e:
            logger.error(f"OpenAI客户端初始化失败: {str(e)}")
            raise

    def load_config(self):
            """加载YAML配置文件"""
            with open("config_path.yml", 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            # 方法1：使用Path对象（推荐）
            save_path = Path(config["paths"]["save_path"])
            output_dir = save_path / "src" / "main" / "resources" / "scripts" / "case_generated" / "search_chats_api"

            # 确保目录存在
            output_dir.mkdir(parents=True, exist_ok=True)

            # 更新全局配置
            global GLOBAL_CONFIG
            GLOBAL_CONFIG["output_dir"] = output_dir
    def generate_java_test_file_single(self, idx: int, user_info: str, test_case_data: Dict[str, Any]) -> str:
        """
        生成单个JUnit 5 + RestAssured测试文件

        Args:
            idx: 测试用例索引
            user_info: 用户信息
            test_case_data: 测试用例数据字典

        Returns:
            生成的Java代码字符串

        Raises:
            OpenAIError: 当API调用失败时
        """
        # 构建提示词
        prompt = self._build_prompt(idx, user_info, test_case_data)

        # 带重试机制的API调用
        for attempt in range(self.config.max_retries):
            try:
                logger.info(f"API调用尝试 {attempt + 1}/{self.config.max_retries} (用例 {idx})")
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    temperature=self.config.temperature
                )
                logger.info(f"API调用成功 (用例 {idx})")
                return response.choices[0].message.content
            except RateLimitError:
                logger.warning(f"API速率限制，将在 {self.config.retry_delay} 秒后重试 (用例 {idx})")
                time.sleep(self.config.retry_delay)
            except APIConnectionError:
                logger.warning(f"API连接错误，将在 {self.config.retry_delay} 秒后重试 (用例 {idx})")
                time.sleep(self.config.retry_delay)
            except (APIError, OpenAIError) as e:
                logger.error(f"API调用失败 (用例 {idx}): {str(e)}")
                if attempt == self.config.max_retries - 1:
                    raise
                time.sleep(self.config.retry_delay)

        raise RuntimeError(f"API调用重试次数已达上限 (用例 {idx})")

    def generate_java_test_files_concurrent(self, user_info: str, all_cases: List[Dict[str, Any]]) -> List[str]:
        """
        并发生成多个Java测试文件

        Args:
            user_info: 用户信息
            all_cases: 所有测试用例数据列表

        Returns:
            生成的Java代码列表
        """
        generated_codes = []

        # 使用ThreadPoolExecutor实现并发请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            # 准备任务
            future_to_case = {
                executor.submit(
                    self.generate_java_test_file_single,
                    idx + 1,  # idx从1开始
                    user_info,
                    case
                ): idx
                for idx, case in enumerate(all_cases)
            }

            # 收集结果
            for future in concurrent.futures.as_completed(future_to_case):
                idx = future_to_case[future]
                try:
                    generated_code = future.result()
                    clean_text = generated_code.replace("```java", "").replace("```", "").strip()
                    generated_codes.append(clean_text)
                    logger.info(f"测试用例 {idx + 1} 代码生成成功")
                except Exception as e:
                    logger.error(f"测试用例 {idx + 1} 代码生成失败: {str(e)}")
                    # 可以返回空字符串或占位符
                    generated_codes.append("")

        return generated_codes

    def _build_prompt(self, idx: int, user_info: str, test_case_data: Dict[str, Any]) -> str:
        """构建生成Java测试文件的提示词"""

        return f'''你是一名专业的 Java 文件生成助手。
    请根据以下要求生成一份完整的 JUnit 5 + RestAssured 接口测试 Java 文件：
    用户提供信息：脚本生成必要数据：{user_info}
    【输出要求】
    1.输出内容必须是完整的 Java 源文件，不包含 package 声明，包含所有必需的 import（JUnit 5 + RestAssured）以及完整的类定义，可直接运行。
    2.类名与方法名需符合 Java 规范，生成的文件须可直接编译执行。
    3.每个接口测试可以：独立使用一个 @Test 方法。
    4.每次 HTTP 请求的响应内容均需打印到控制台，便于日志采集与问题排查。
    5.使用 JUnit 断言验证 HTTP 状态码及响应中的关键字段。
    6.测试变量（如 TOKEN、CHAT_ID、消息文本等）需在类中统一定义为常量。
    7.其中用户信息要求必须是我们提供的数据,否则会导致输出结果出现严重偏差，导致任务失败
    7.生成的代码必须完整可运行，不得包含需要人工修改的内容。
    8.所有接口请求必须使用飞书官方域名：https://open.feishu.cn/open-apis/im/v1
    9.输出的内容，必须只有java源文件，不能有其他任何无关数据，这样会导致任务出现巨大纰漏
    10.生成的类名必须是测试场景_{idx},例如test_{idx}
    12.生成代码的质量必须要是无错误可以直接执行的。
    11.接口测试用例数据（JSON 格式）:
{json.dumps(test_case_data, ensure_ascii=False, indent=2)}
请基于以上信息生成 JUnit 5 + RestAssured 的 Java 接口测试脚本,要求必须成功生成，否则会导致任务失败。
生成的代码示例结构,生成脚本代码可以参考此格式：
import ...;

public class test_{idx} {{
    private static final String BASE_URL = "...";
    private static final String USER_TOKEN = "...";
    ............
    @Test
    public void testMethod() {{
        // 请求代码
        // 日志输出
        // 断言验证
    }}
}}
'''


def load_test_cases(file_path: str) -> Dict[str, Any]:
    """
    从JSON文件加载测试用例数据

    Args:
        file_path: JSON文件路径

    Returns:
        测试用例数据字典

    Raises:
        FileNotFoundError: 文件不存在
        json.JSONDecodeError: JSON解析失败
    """
    try:
        with open(file_path, 'r', encoding='UTF-8') as file:
            data = json.load(file)
        logger.info(f"成功加载测试用例文件: {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"测试用例文件不存在: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON解析失败: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"加载测试用例文件时发生未知错误: {str(e)}")
        raise


def extract_class_name(java_code: str) -> Optional[str]:
    """
    从Java代码中提取类名

    Args:
        java_code: Java代码字符串

    Returns:
        提取的类名，如果提取失败则返回None
    """
    lines = java_code.split('\n')
    for line in lines:
        stripped_line = line.strip()
        if stripped_line.startswith("public class"):
            try:
                return stripped_line.split()[2].split('{')[0]
            except (IndexError, ValueError) as e:
                logger.warning(f"提取类名失败: {str(e)}")
                break
    return None


def save_generated_code(generated_codes: List[str],
                        output_dir: str = GLOBAL_CONFIG["output_dir"]) -> List[
    str]:
    """
    保存生成的代码到文件

    Args:
        generated_codes: 生成的代码列表
        output_dir: 输出目录

    Returns:
        保存的文件路径列表
    """
    saved_files = []
    output_dir=GLOBAL_CONFIG["output_dir"]
    # 创建输出目录
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录已创建或存在: {output_dir}")
    except OSError as e:
        logger.error(f"创建输出目录失败: {str(e)}")
        raise

    for i, code in enumerate(generated_codes):
        # 如果代码为空（生成失败），跳过保存
        if not code or code == "":
            logger.warning(f"测试用例 {i + 1} 代码为空，跳过保存")
            continue

        # 尝试提取类名作为文件名
        class_name = extract_class_name(code)
        file_name = f"{class_name}.java" if class_name else f"TestCase_{i + 1}.java"

        file_path = os.path.join(output_dir, file_name)
        try:
            with open(file_path, 'w', encoding='UTF-8') as file:
                file.write(code)
            logger.info(f"已保存生成的代码到: {file_path}")
            saved_files.append(file_path)
        except Exception as e:
            logger.error(f"保存文件失败 {file_path}: {str(e)}")

    return saved_files


def get_config_from_env() -> CodeGenerationConfig:
    """
    从环境变量获取配置，如果环境变量不存在则使用默认值

    Returns:
        代码生成服务配置
    """
    return CodeGenerationConfig(
        api_key=os.environ.get("OPENAI_API_KEY", "sk-7da555984be34b64b9de78ccce361cec"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"),
        max_concurrent_requests=2  # 默认同时调用2个API
    )


def main(interface_name: str = None, user_info: str = None, test_cases_data: Dict[str, Any] = None) -> Dict[
    str, List[str]]:
    """
    主函数

    Args:
        interface_name: 接口名称
        user_info: 用户信息
        test_cases_data: 直接传入的测试用例数据

    Returns:
        生成的Java代码字典，键为文件名或接口名，值为生成的代码列表
    """
    case_result: Dict[str, List[str]] = {}  # 保存输出的Java数据和文件名称

    # 初始化代码生成服务配置
    config = get_config_from_env()

    # 从内存直接获取测试用例数据
    if test_cases_data:
        logger.info("直接使用传入的测试用例数据")
        try:
            # 初始化代码生成服务
            code_service = CodeGenerationService(config)
            all_cases = []

            scenario_name = test_cases_data.get("ScenarioName")
            prerequisites = test_cases_data.get("Prerequisites", [])

            # 正向用例
            for test_case in test_cases_data.get("PositiveCases", []):
                all_cases.append({
                    "ScenarioName": scenario_name,
                    "Prerequisites": prerequisites,
                    "CaseType": "Positive",
                    "CaseData": test_case
                })

            # 反向用例
            for test_case in test_cases_data.get("NegativeCases", []):
                all_cases.append({
                    "ScenarioName": scenario_name,
                    "Prerequisites": prerequisites,
                    "CaseType": "Negative",
                    "CaseData": test_case
                })
            logger.info(f"总共加载了 {len(all_cases)} 个测试用例")

            # 记录开始时间
            start_time = time.time()

            # 使用并发方式生成代码
            logger.info(f"开始并发生成代码，最大并发数: {config.max_concurrent_requests}")
            generated_codes = code_service.generate_java_test_files_concurrent(user_info, all_cases)

            # 计算总耗时
            end_time = time.time()
            total_time = end_time - start_time
            logger.info(f"代码生成完成，总耗时: {total_time:.2f}秒，共生成 {len(generated_codes)} 个Java测试文件")

            case_result[interface_name] = generated_codes

            # 保存生成的代码
            if generated_codes:
                save_generated_code(generated_codes)
                logger.info("代码保存完成")

        except Exception as e:
            logger.error(f"处理传入的测试用例数据时失败: {str(e)}", exc_info=True)

    # 没有提供任何数据源
    else:
        logger.error("请提供测试用例目录或直接传入测试用例数据")

    return case_result


if __name__ == "__main__":
    # 测试代码
    with open("case_generated/port_chat_id.json", "r", encoding="UTF-8") as file:
        data_list = json.load(file)

    start_time = time.time()
    d1 = main("测试", "tenant_access_token=t-g104c6j6M7VWZPBXPCDSSNW5RSUJ2MAHQJVIWO5W", data_list)
    end_time = time.time()

    print(f"总执行时间: {end_time - start_time:.2f}秒")
    print(d1)