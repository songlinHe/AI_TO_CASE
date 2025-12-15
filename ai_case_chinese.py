import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import yaml
import sys
import os
from openai import OpenAI
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_changjing import UsernameFileFinder
# 导入new_code模块
import new_code
def read_yaml_file(file_path: str) -> Optional[Dict]:
    """读取YAML文件并返回解析后的字典"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"读取YAML文件失败 {file_path}: {e}")
        return None

GLOBAL_CONFIG = {
    "output_dir": None
}
class GeminiClient:
    """Gemini API客户端封装"""
    
    def __init__(self, api_key: str):
        """初始化Gemini客户端"""

        # 初始化客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com")
        self.load_config()
    def load_config(self):
        """加载YAML配置文件"""
        with open("config_path.yml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 方法1：使用Path对象（推荐）
        save_path = Path(config["paths"]["save_path"])
        GLOBAL_CONFIG["output_dir"] = save_path / "src" / "main" / "resources" / "case_generated"
    def generate_test_cases(self, api_docs, user_info,changjing,xiangsi):
        """
        调用Gemini API生成测试用例
        
        Args:
            api_docs: API文档字典
            relationship_data: 接口关系文档
            
        Returns:
            生成的测试用例字符串，失败时返回None
        """
        try:
            # 将字典转换为YAML格式文本
            api_doc_text = yaml.dump(api_docs, allow_unicode=True, sort_keys=False)
            # 构造提示词
            prompt = (
                    f"""
            请以“资深自动化测试架构师”的专业身份，基于以下 Feishu（飞书）OpenAPI 文档内容，按照指定业务场景设计一套高质量的接口级自动化测试用例集，并尽可能快速生成结果。

            【核心要求】
            - 每个接口的测试用例必须独立设计，不得出现跨接口的实际调用或依赖。
            - 所有测试设计必须基于指定场景进行推导，并确保与业务语义强相关。

            【场景描述】
            {changjing}

            【编写规则】
            - 独立性：每个接口均需作为单独测试对象进行用例设计。
            - 场景贴合：测试用例必须严格契合上述场景的业务逻辑与功能边界。

            【输出格式要求】
            - 必须输出 **纯 JSON（字典结构）**，否则任务判定失败。
            - 外层必须使用唯一的 ```json 代码块包裹全部内容。
            - 每条测试用例必须具备以下字段：
              - Prerequisites（前置条件）
              - Steps（步骤）
              - ExpectedResults（预期结果）
            - 正向用例中需包含 DataMapping 字段，仅用于说明：
              “接口返回的数据在业务链路中可能被下一接口如何消费（仅说明，不执行下一步）”
            - 强调：**每个接口必须形成独立的测试用例集合，严禁混合设计。**

            【PositiveCases（正向用例要求）】
            - 每个接口至少编写 1 条完整正向用例。
            - Steps 中必须包含：Method、Path、输入参数、断言字段等关键内容。

            【NegativeCases（反向/异常用例要求）】
            - 每个接口至少设计 1 条异常用例。
            - 异常用例必须包含：异常类型、预期状态码、预期错误码、错误提示关键字等。
            - 仍需保持接口级测试的独立性，不得触发跨接口链路。
            
            - **参考相似文档**
            分析相似文档中的接口命名、参数结构、输入输出、业务逻辑和异常处理模式，将有价值的信息用于新场景的用例设计。
            参考文档仅供参考，不代表实际内容，需要根据具体任务或具体文档进行分析：
            {xiangsi}
            以下为相关 OpenAPI 文档：
            {api_doc_text}
            
            - 请务必以 **纯 JSON** 格式输出，否则将导致任务失败。
            """
                    + """
            【可参考的 JSON 字段结构】（仅供参考）
            {
              "ScenarioName": "",
              "Prerequisites": [],
              "PositiveCases": [
                {
                  "CaseName": "",
                  "Steps": [
                    {
                      "StepID": 1,
                      "Description": "",
                      "Method": "",
                      "Path": "",
                      "Input": {},
                      "Expectations": [],
                      "DataMapping": {}
                    }
                  ],
                  "ExpectedResults": []
                }
              ],
              "NegativeCases": []
            }
            """
            )

            # 调用Gemini API
            # 创建聊天完成请求
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个有用的AI助手,你是一个资深的自动化测试架构师，要根据要求完成下面任务。"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ],
                temperature=0.6
            )

        except Exception as e:
            print(f"调用Gemini API出错: {e}")
            return None
        print(response.choices[0].message.content)

        return response.choices[0].message.content

    def auto_fenxi_port(self, api_doc_content):
        prompt = f"""
            你是一位经验丰富的测试工程师，请仔细且简短的分析下面这个接口文档，并思考这个接口在真实使用中可能会遇到的各种情况。
                1. 如果文档中包含多个接口，需要深入分析这些接口之间的业务关联和数据流转关系
                2. 如果文档中只有单接口，不需要分析接口之间的关系，只需要分析场景及功能点
                3. 基于分析结果，设计完整的测试场景
                4. 列出所有可能的测试场景，包括正常流程和异常情况
接口文档内容：
        {api_doc_content}""" + """请严格按照以下JSON格式输出：
        {{
      "scenarios": [
        {{
          "category": "正常场景",
          "priority": "P0",
          "description": "场景的简要描述",
          "test_points": [
            "具体的测试点1",
            "具体的测试点2"
          ],
          "preconditions": "前置条件",
          "expected_results": "预期结果"
        }},
        {{
          "category": "边界场景",
          "priority": "P1",
          "description": "边界条件测试",
          "test_points": [
            "具体的测试点"
          ],
          "preconditions": "前置条件",
          "expected_results": "预期结果"
        }}
      ]
    }}
    【分类说明】
    1. 正常场景（P0优先级）：主流程、核心功能
    2. 边界场景（P1优先级）：参数边界、数据边界
    3. 异常场景（P1优先级）：错误参数、系统异常
    """
        response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个有用的AI助手,你是一个资深的自动化测试架构师，要根据要求完成下面任务。"
                    },
                    {
                        "role": "user",
                        "content": f"{prompt}"
                    }
                ],
                temperature=0.6
            )

        print(response.choices[0].message.content)

        return response.choices[0].message.content
    def generate_test_cases_auto(self, api_docs, fenxiport):
            """
            调用Gemini API生成测试用例

            Args:
                api_docs: API文档字典
                relationship_data: 接口关系文档

            Returns:
                生成的测试用例字符串，失败时返回None
            """
            try:
                # 将字典转换为YAML格式文本
                api_doc_text = yaml.dump(api_docs, allow_unicode=True, sort_keys=False)

                # 构造提示词
                prompt = """
                请以资深自动化测试架构师的专业视角，基于以下提供的飞书（Feishu）OpenAPI 规范文档，系统性分析各接口的功能边界、输入输出约束及潜在异常路径，并为每个接口分别生成一套完备的接口级自动化测试用例集。要求保证设计的严谨性、覆盖度以及可执行性。
                【总体设计原则】
                1. 场景适配性：所有测试用例需准确贴合接口本身的业务语义、数据结构及参数契约。
                2. 可验证性：所有断言必须基于接口返回的明确字段、HTTP 状态码以及错误码规范进行验证。
                3. 如果输入的内容有多个接口信息，需要根据给出的接口信息，自动分析多个接口之间的入参和出参之间的关联。
                4. 如果有多个接口，需要模型自动分析接口之间所涉及的场景。
                5. 如果场景内的用例写到一个场景下，生成测试用例时不能太过于繁琐，需要严格保证json格式的输出。
                【测试用例设计要求】

                一、正向用例（PositiveCases）
                - 每个接口至少设计 1 条完整正向流程用例。
                - Steps 中必须明确包含：
                  - Method：HTTP 请求方法（如 GET、POST、PUT 等）
                  - Path：接口路径
                  - Input：请求参数，包括 Query、Body、Header 等
                  - Expectations：针对响应的校验点，包括但不限于：
                    * HTTP 状态码
                    * 关键字段值与类型校验
                    * 响应结构完整性校验

                二、异常用例（NegativeCases）
                - 每个接口至少设计 1 条异常/反向用例，用于验证接口的健壮性与容错能力。
                - 异常用例必须包含：
                  - 异常类型描述（如参数缺失、格式错误、非法值、权限不足等）
                  - 期望 HTTP 状态码
                  - 期望错误码（如接口有规范）
                  - 期望错误提示关键字或核心内容

                三、前置条件（Prerequisites）
                - 需说明执行本接口用例前所必须具备的条件，包括但不限于：
                  - 鉴权方式（如 AccessToken）
                  - 环境要求（如测试环境说明）
                  - 固定基础数据（如必要的业务实体 ID）

                【输出格式要求】
                - 最终输出必须严格为 **纯 JSON（字典）结构**，否则视为任务失败。
                - 整体输出必须包含在唯一一个 ```json 代码块中。
                - JSON 顶层结构必须符合以下格式：
                {
                  "ScenarioName": "",
                  "Prerequisites": [],
                  "PositiveCases": [
                    {
                      "CaseName": "",
                      "Steps": [
                        {
                          "StepID": 1,
                          "Description": "",
                          "Method": "",
                          "Path": "",
                          "Input": {},
                          "Expectations": []
                        }
                      ],
                      "ExpectedResults": []
                    }
                  ],
                  "NegativeCases": []
                }"""+f"""【输入的 OpenAPI 文档如下】：
                {api_doc_text}
                场景分析数据，生成用例时请进行参考
                {fenxiport}
                请严格遵循上述要求，生成结构规范、覆盖充分、可直接用于接口自动化测试的 JSON 输出."""

                # 调用Gemini API
                # 创建聊天完成请求
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个有用的AI助手,你是一个资深的自动化测试架构师，要根据要求完成下面任务。"
                        },
                        {
                            "role": "user",
                            "content": f"{prompt}"
                        }
                    ],
                    temperature=0.6
                )

            except Exception as e:
                print(f"调用Gemini API出错: {e}")
                return None
            print(response.choices[0].message.content)

            return response.choices[0].message.content

def save_test_cases_to_file(test_cases, output_file):
    """
    保存测试用例到JSON文件
    
    Args:
        test_cases: 测试用例字符串
        output_file: 输出文件路径
        
    Returns:
        保存成功返回True，失败返回False
    """
    try:
        # 移除代码块标记并清理空白字符
        clean_text = test_cases.replace("```json", "").replace("```", "").strip()
        
        # 尝试将字符串解析为JSON对象，然后再格式化输出，这样可以保证正确换行
        try:
            # 如果是有效的JSON字符串，解析后再格式化输出
            json_obj = json.loads(clean_text)
            with open(output_file, 'w', encoding='UTF-8') as file:
                json.dump(json_obj, file, ensure_ascii=False, indent=4)
        except json.JSONDecodeError:
            # 如果不是有效的JSON字符串，直接写入文本
            with open(output_file, 'w', encoding='UTF-8') as file:
                file.write(clean_text)
        
        return True
    except Exception as e:
        print(f"保存测试用例失败: {e}")
        return False


def read_output_dir(output_dir: str) -> Optional[List[str]]:
    """
    读取指定目录中的文件列表
    
    Args:
        output_dir: 要读取的目录路径
        
    Returns:
        文件路径列表，如果目录不存在或发生错误则返回None
    """
    try:
        # 检查目录是否存在
        if not os.path.exists(output_dir):
            print(f"目录不存在: {output_dir}")
            return None
        
        # 获取目录中的所有文件（不包括子目录）
        files = []
        for file_name in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file_name)
            if os.path.isfile(file_path):
                files.append(file_path)
        
        print(f"成功读取目录 {output_dir}，找到 {len(files)} 个文件")
        return files
        
    except Exception as e:
        print(f"读取目录 {output_dir} 时出错: {e}")
        return None
def fenxiport(map_data):
    gemini_client = GeminiClient(api_key="sk-7da555984be34b64b9de78ccce361cec")
    case_json = gemini_client.auto_fenxi_port(
            map_data,  # API文档内容
        )
    return case_json

def ai_case_main_auto(user_info,map_data):
    """主函数"""
    print("正在生成测试用例...")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    changjing = f"auto_generate_{timestamp}"    # 确保输出目录存在
    os.makedirs(GLOBAL_CONFIG["output_dir"], exist_ok=True)

    # 创建一个字典来保存所有生成的测试用例
    generated_test_cases = {}

    # 创建一个字典来保存生成的Java代码
    generated_test_cases_code = {}

    # 初始化Gemini客户端（只创建一次）
    gemini_client = GeminiClient(api_key="sk-7da555984be34b64b9de78ccce361cec")

    # 假设map_data中的键是关系文件名，值是关系文件内容
    # 例如：{"relationship.yml": {"...": "..."}}

    # 提取接口名称（从文件名中）

    try:
        # 获取API文档内容（假设从port_message.yml读取）
        api_doc_content = map_data

        if not api_doc_content:
            print(f"无法读取API文档，跳过 {changjing}")
        fenxiport_data=fenxiport(map_data)
        # 调用Gemini生成测试用例，传入API文档和关系文档内容
        case_json = gemini_client.generate_test_cases_auto(
            api_doc_content,  # API文档内容
            fenxiport_data,#端口关系，以及可能的场景
        )

        if case_json:
            # 保存测试用例到文件
            output_file = os.path.join(GLOBAL_CONFIG["output_dir"], f"{changjing}.json")
            if save_test_cases_to_file(case_json, output_file):
                print(f"✓ 测试用例已保存到: {output_file}")
            else:
                print(f"✗ 保存测试用例失败: {changjing}")

            # 解析生成的测试用例JSON并保存到字典中
            try:
                # 移除代码块标记并清理空白字符
                clean_text = case_json.replace("```json", "").replace("```", "").strip()
                # 尝试解析为JSON
                case_data = json.loads(clean_text)
                generated_test_cases[changjing] = case_data
                print(f"✓ 测试用例已添加到字典: {changjing}")
            except json.JSONDecodeError as e:
                print(f"✗ 解析测试用例JSON失败 {changjing}: {e}")
        else:
            print(f"✗ 生成测试用例失败: {changjing}")

    except Exception as e:
        print(f"✗ 处理 {changjing} 时出错: {e}")

    # 将生成的测试用例传递给new_code的main函数进行下一步处理
    if generated_test_cases:
        print(f"\n共生成 {len(generated_test_cases)} 个接口的测试用例，准备调用new_code进行下一步处理...")

        # 创建输出目录
        output_root = "generated_java"
        os.makedirs(output_root, exist_ok=True)

        try:
            for interface_name, test_data in generated_test_cases.items():
                print(f"\n开始处理接口 {interface_name} 的Java测试代码生成...")

                # new_code.main 返回的是多个 Java 文件内容
                generated_codes = new_code.main(interface_name=interface_name, user_info=user_info,
                                                test_cases_data=test_data)

                interface_dir = os.path.join(output_root, interface_name)
                os.makedirs(interface_dir, exist_ok=True)

                # 保存每一个生成的 Java 文件
                for idx, code_text in enumerate(generated_codes, start=1):

                    # 自动从 class 名提取文件名（如果找不到 class，就用默认）
                    file_name = None
                    for line in code_text.splitlines():
                        if line.strip().startswith("public class"):
                            file_name = line.split()[2].split("{")[0].strip() + ".java"
                            break

                    if not file_name:
                        file_name = f"TestCase_{idx}.java"

                print(f"\n接口 {interface_name} 处理完成:")
                print(f"  - 生成了 {len(generated_codes)} 个Java测试文件")

                # 将生成的Java代码保存到字典中
                generated_test_cases_code[interface_name] = generated_codes

                # 打印每个生成的类名
                for class_name in generated_codes:
                    print(f"    ✓ 生成了类: {class_name}")

        except Exception as e:
            print(f"✗ 调用new_code.main时出错: {e}")
        else:
            print("\n没有生成任何测试用例，跳过调用new_code")

    # 返回生成的测试用例字典和Java代码字典，以便fastapi可以接收
    return generated_test_cases, generated_test_cases_code
def main(changjing: str, user_info: str,map_data:str) -> Tuple[Dict[str, Any], Dict[str, Dict[str, str]]]:
    """主函数"""
    print("正在生成测试用例...")

    # 确保输出目录存在
    os.makedirs(GLOBAL_CONFIG["output_dir"], exist_ok=True)
    finder_instance = UsernameFileFinder(GLOBAL_CONFIG["output_dir"])
    xiangsi = finder_instance.main(GLOBAL_CONFIG["output_dir"], changjing)
    # 创建一个字典来保存所有生成的测试用例
    generated_test_cases = {}

    # 创建一个字典来保存生成的Java代码
    generated_test_cases_code = {}

    # 初始化Gemini客户端（只创建一次）
    gemini_client = GeminiClient(api_key="sk-7da555984be34b64b9de78ccce361cec")

    # 假设map_data中的键是关系文件名，值是关系文件内容
    # 例如：{"relationship.yml": {"...": "..."}}

    # 提取接口名称（从文件名中）


    try:
        # 获取API文档内容（假设从port_message.yml读取）
        api_doc_content = map_data

        if not api_doc_content:
            print(f"无法读取API文档，跳过 {changjing}")

        # 调用Gemini生成测试用例，传入API文档和关系文档内容
        case_json = gemini_client.generate_test_cases(
            api_doc_content,  # API文档内容
            user_info,  # 关系文档内容（来自map_data）
            changjing,  # 测试场景
            xiangsi,
        )

        if case_json:
            # 保存测试用例到文件
            output_file = os.path.join(GLOBAL_CONFIG["output_dir"], f"{changjing}.json")
            if save_test_cases_to_file(case_json, output_file):
                print(f"✓ 测试用例已保存到: {output_file}")
            else:
                print(f"✗ 保存测试用例失败: {changjing}")

            # 解析生成的测试用例JSON并保存到字典中
            try:
                # 移除代码块标记并清理空白字符
                clean_text = case_json.replace("```json", "").replace("```", "").strip()
                # 尝试解析为JSON
                case_data = json.loads(clean_text)
                generated_test_cases[changjing] = case_data
                print(f"✓ 测试用例已添加到字典: {changjing}")
            except json.JSONDecodeError as e:
                print(f"✗ 解析测试用例JSON失败 {changjing}: {e}")
        else:
            print(f"✗ 生成测试用例失败: {changjing}")

    except Exception as e:
        print(f"✗ 处理 {changjing} 时出错: {e}")
    
    # 将生成的测试用例传递给new_code的main函数进行下一步处理
    if generated_test_cases:
        print(f"\n共生成 {len(generated_test_cases)} 个接口的测试用例，准备调用new_code进行下一步处理...")

        # 创建输出目录
        output_root = "generated_java"
        os.makedirs(output_root, exist_ok=True)

        try:
            for interface_name, test_data in generated_test_cases.items():
                print(f"\n开始处理接口 {interface_name} 的Java测试代码生成...")

                # new_code.main 返回的是多个 Java 文件内容
                generated_codes = new_code.main(interface_name=interface_name,user_info=user_info,test_cases_data=test_data)

                interface_dir = os.path.join(output_root, interface_name)
                os.makedirs(interface_dir, exist_ok=True)

                # 保存每一个生成的 Java 文件
                for idx, code_text in enumerate(generated_codes, start=1):

                    # 自动从 class 名提取文件名（如果找不到 class，就用默认）
                    file_name = None
                    for line in code_text.splitlines():
                        if line.strip().startswith("public class"):
                            file_name = line.split()[2].split("{")[0].strip() + ".java"
                            break

                    if not file_name:
                        file_name = f"TestCase_{idx}.java"

                print(f"\n接口 {interface_name} 处理完成:")
                print(f"  - 生成了 {len(generated_codes)} 个Java测试文件")
                
                # 将生成的Java代码保存到字典中
                generated_test_cases_code[interface_name] = generated_codes
                
                # 打印每个生成的类名
                for class_name in generated_codes:
                    print(f"    ✓ 生成了类: {class_name}")
                    
        except Exception as e:
            print(f"✗ 调用new_code.main时出错: {e}")
        else:
            print("\n没有生成任何测试用例，跳过调用new_code")
    
    # 返回生成的测试用例字典和Java代码字典，以便fastapi可以接收
    return generated_test_cases, generated_test_cases_code
