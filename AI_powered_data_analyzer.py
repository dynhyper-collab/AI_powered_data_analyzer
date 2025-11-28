import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
import seaborn as sns
import io  # 显式导入io模块
import re
import ast
import traceback
import requests
from io import BytesIO

# 自定义安全异常类
class SecurityError(Exception):
    """自定义安全异常，用于代码安全检查"""
    pass

# 确保中文显示正常
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 使用硅基流动接口的大语言模型客户端
class SiliconFlowLLM:
    def __init__(self, base_url="https://api.siliconflow.cn", model="Qwen/Qwen3-Coder-30B-A3B-Instruct"):
        self.base_url = base_url
        self.model = model
        self.headers = {"Content-Type": "application/json"}
        
    def _get_system_prompt(self, df):
        columns = df.columns.tolist()
        dtypes = {col: str(df[col].dtype) for col in columns}
        sample_data = df.head(3).to_dict(orient="records")
        
        # 重点修改：明确告知如何使用BytesIO，不需要通过io模块
        system_prompt = """你是一个专业的数据分析代码生成助手，需要根据用户的数据分析需求，生成可在Python中执行的代码。

以下是需要分析的数据信息：
- 列名及数据类型：{dtypes}
- 数据样例：{sample_data}

请遵循以下要求生成代码：
1. 代码必须是完整的、可运行的Python代码，包含一个名为analyze的函数，该函数接收一个pandas DataFrame(df)作为参数
2. 函数需要返回一个字典，包含分析结果，其中必须包含'plot'键，对应的值是一个BytesIO对象（存储可视化图表）
3. 可视化必须使用matplotlib或seaborn，确保中文能正常显示
4. 只生成代码，不要添加任何解释说明
5. 代码中只能使用已导入的模块和对象：
   - pandas (as pd)
   - numpy (as np)
   - matplotlib.pyplot (as plt)
   - seaborn (as sns)
   - BytesIO (直接使用，无需通过io模块)
6. 不需要导入任何模块，直接使用上述已提供的模块和对象
7. 生成图表后，使用以下方式保存到BytesIO：
   buf = BytesIO()
   plt.savefig(buf, format='png', bbox_inches='tight')
   buf.seek(0)
8. 不要读取或写入本地文件，不要进行网络请求

示例代码结构：
def analyze(df):
    # 分析逻辑
    stats = df.describe()
    
    # 生成可视化
    plt.figure(figsize=(10, 6))
    sns.histplot(df[df.columns[0]])
    plt.title('数据分布')
    
    # 保存图表 - 注意这里直接使用BytesIO，不需要io.前缀
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    
    return {{
        'plot': buf,
        'statistics': stats
    }}
""".format(dtypes=dtypes, sample_data=sample_data)
        
        return system_prompt
    
    def generate_code(self, user_query, df):
        try:
            system_prompt = self._get_system_prompt(df)
            
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=self.headers,
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 1500
                },
                timeout=60
            )
            
            if response.status_code != 200:
                return f"API调用失败，状态码：{response.status_code}\n响应内容：{response.text}"
            
            result = response.json()
            code = result["choices"][0]["message"]["content"]
            
            code_match = re.search(r'```python(.*?)```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()
            
            # 过滤导入语句
            lines = code.split('\n')
            filtered_lines = [line for line in lines if not line.strip().startswith(('import ', 'from '))]
            code = '\n'.join(filtered_lines)
            
            # 替换可能的io.BytesIO为直接使用BytesIO
            code = code.replace('io.BytesIO', 'BytesIO')
            
            return code
            
        except Exception as e:
            return f"生成代码时出错：{str(e)}\n{traceback.format_exc()}"

# 安全代码沙箱执行环境
class SecureCodeSandbox:
    def __init__(self):
        # 确保BytesIO和io模块都能被访问到
        self.allowed_modules = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'BytesIO': BytesIO,
            'io': io  # 显式添加io模块
        }
        
        self.banned_patterns = {
            'os', 'sys', 'subprocess', 'eval', 'exec', 'open', 
            'compile', 'globals', 'locals', 'importlib', 'pickle',
            'socket', 'requests', 'urllib', 'shutil', 'pathlib'
        }
    
    def _check_safe_code(self, code):
        for pattern in self.banned_patterns:
            if re.search(r'\b' + re.escape(pattern) + r'\b', code):
                raise SecurityError(f"代码中包含禁止使用的模块或函数: {pattern}")
        
        if re.search(r'^(import |from )', code, re.MULTILINE):
            raise SecurityError("代码中不允许包含导入语句")
        
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    raise SecurityError("代码中不允许包含导入语句")
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in self.banned_patterns:
                        raise SecurityError(f"禁止调用危险函数: {node.func.id}")
        except SyntaxError as e:
            raise SecurityError(f"代码存在语法错误: {str(e)}")
        
        return True
    
    def execute(self, code, df):
        try:
            self._check_safe_code(code)
            
            safe_globals = {
                '__builtins__': {
                    'abs': abs, 'all': all, 'any': any, 'bool': bool, 'float': float,
                    'int': int, 'len': len, 'list': list, 'dict': dict, 'tuple': tuple,
                    'str': str, 'sum': sum, 'min': min, 'max': max, 'range': range
                }
            }
            
            # 添加允许的模块和数据框
            safe_globals.update(self.allowed_modules)
            safe_globals['df'] = df
            
            # 执行代码
            exec(code, safe_globals)
            
            if 'analyze' in safe_globals and callable(safe_globals['analyze']):
                result = safe_globals['analyze'](df)
                return result
            else:
                return {'error': '生成的代码中没有定义有效的analyze函数'}
                
        except SecurityError as e:
            return {'error': f'代码安全检查失败: {str(e)}'}
        except Exception as e:
            return {'error': f'代码执行错误: {str(e)}\n{traceback.format_exc()}'}

# 主应用
def main():
    st.title("AI-powered Data Analyzer")
    st.write("上传数据文件，用自然语言提出分析需求，系统将使用AI生成代码并展示结果")
    
    with st.sidebar:
        st.header("模型服务配置")
        siliconflow_url = st.text_input("模型接口地址", "https://api.siliconflow.cn")
        model_name = st.text_input("使用的模型名称", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
        api_key = st.text_input("API密钥（如需要）", type="password")
        st.info("请确保已开通大语言模型服务并配置好模型")
        
        if api_key:
            st.session_state.api_key = api_key
    
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'generated_code' not in st.session_state:
        st.session_state.generated_code = ""
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    
    uploaded_file = st.file_uploader("上传数据文件 (CSV或Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            
            st.success("数据上传成功！")
            
            with st.expander("查看数据预览"):
                st.dataframe(st.session_state.df.head())
            
            with st.expander("查看数据基本信息"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"形状: {st.session_state.df.shape}")
                    st.write("列名:")
                    st.write(list(st.session_state.df.columns))
                with col2:
                    st.write("数据类型:")
                    st.write(st.session_state.df.dtypes.astype(str).to_dict())
        
        except Exception as e:
            st.error(f"读取数据失败: {str(e)}")
            st.session_state.df = None
    
    if st.session_state.df is not None:
        user_query = st.text_area("请输入您的数据分析需求", 
                                 placeholder="例如: 汇总数据基本信息、分析各变量相关性...")
        
        if st.button("生成并执行分析代码") and user_query.strip() != "":
            with st.spinner("正在调用AI生成分析代码..."):
                llm = SiliconFlowLLM(base_url=siliconflow_url, model=model_name)
                if 'api_key' in st.session_state and st.session_state.api_key:
                    llm.headers["Authorization"] = f"Bearer {st.session_state.api_key}"
                
                st.session_state.generated_code = llm.generate_code(user_query, st.session_state.df)
            
            with st.expander("查看生成的代码", expanded=True):
                st.code(st.session_state.generated_code, language="python")
            
            if "API调用失败" in st.session_state.generated_code or "生成代码时出错" in st.session_state.generated_code:
                st.error(st.session_state.generated_code)
            else:
                with st.spinner("正在执行分析..."):
                    sandbox = SecureCodeSandbox()
                    st.session_state.analysis_result = sandbox.execute(
                        st.session_state.generated_code, 
                        st.session_state.df
                    )
                
                if st.session_state.analysis_result:
                    if 'error' in st.session_state.analysis_result:
                        st.error(st.session_state.analysis_result['error'])
                    else:
                        st.success("分析完成！")
                        
                        if 'plot' in st.session_state.analysis_result:
                            st.image(st.session_state.analysis_result['plot'])
                        
                        for key, value in st.session_state.analysis_result.items():
                            if key != 'plot':
                                with st.expander(f"查看{key}结果"):
                                    st.write(value)

if __name__ == "__main__":
    main()
    