import json
import uuid
import websocket
import threading
import time
import requests
from typing import Dict, List, Optional, Any, Tuple

class JupyterNotebookManager:
    def __init__(self, server: str, username: str, token: str, notebook_path: str = "mfrt_research_tools.ipynb"):
        self.server = server
        self.username = username
        self.token = token
        self.notebook_path = notebook_path
        self.ws = None
        self.kernel_id = None
        self.debug = False
        self.execution_results = []
        self._setup_websocket_lock = threading.Lock()
        self._execution_completed = threading.Event()
        self._msg_id = None
        
    def get_notebook_content(self) -> Dict:
        """获取notebook的当前内容"""
        url = f"http://{self.server}/user/{self.username}/api/contents/{self.notebook_path}?content=1&token={self.token}"
        print(url)
        response = requests.get(url)
        # print(response)
        if response.status_code != 200:
            raise Exception(f"Failed to get notebook content: {response.text}")
        return response.json()
    
    def append_code(self, code: str, metadata: Dict = None) -> bool:
        """追加代码到notebook"""
        try:
            notebook = self.get_notebook_content()
            
            if metadata is None:
                metadata = {"trusted": True}
            
            new_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": metadata,
                "outputs": [],
                "source": code
            }
            
            notebook['content']['cells'].append(new_cell)
            
            url = f"http://{self.server}/user/{self.username}/api/contents/{self.notebook_path}?token={self.token}"
            response = requests.put(url, json={
                "content": notebook['content'],
                "type": "notebook"
            })
            
            return response.status_code == 200
        except Exception as e:
            if self.debug:
                print(f"Error appending code: {e}")
            return False
    
    def _create_message(self, msg_type: str, content: Dict) -> Dict:
        """创建Jupyter消息"""
        return {
            'header': {
                'msg_id': str(uuid.uuid4()),
                'username': self.username,
                'session': str(uuid.uuid4()),
                'msg_type': msg_type,
                'version': '5.3'
            },
            'parent_header': {},
            'metadata': {},
            'content': content,
            'channel': 'shell'
        }
    
    def _on_message(self, ws, message):
        """处理WebSocket消息"""
        msg = json.loads(message)
        msg_type = msg.get('header', {}).get('msg_type', '')
        parent_msg_id = msg.get('parent_header', {}).get('msg_id', '')
        content = msg.get('content', {})
        
        if parent_msg_id != self._msg_id and self._msg_id is not None:
            return
            
        if msg_type == 'stream':
            result = content.get('text', '').strip()
            if result:
                self.execution_results.append(('output', result))
        elif msg_type == 'error':
            error_msg = f"{content.get('ename', '')}: {content.get('evalue', '')}"
            self.execution_results.append(('error', error_msg))
            if self.debug and content.get('traceback'):
                self.execution_results.append(('traceback', content['traceback']))
            self._execution_completed.set()
        elif msg_type == 'execute_reply':
            self._execution_completed.set()
            if content.get('status') == 'error' and not any(r[0] == 'error' for r in self.execution_results):
                self.execution_results.append(('error', str(content)))
    
    def setup_kernel(self) -> Optional[str]:
        """设置并启动一个新的kernel"""
        try:
            response = requests.post(
                f"http://{self.server}/user/{self.username}/api/kernels?token={self.token}",
                headers={"Content-Type": "application/json"}
            )
            kernel_info = response.json()
            return kernel_info['id']
        except Exception as e:
            if self.debug:
                print(f"Error setting up kernel: {e}")
            return None
    
    def connect_websocket(self, kernel_id: str = None) -> bool:
        """连接到kernel的WebSocket"""
        with self._setup_websocket_lock:
            if kernel_id is None:
                kernel_id = self.setup_kernel()
                if not kernel_id:
                    return False
            
            self.kernel_id = kernel_id
            ws_url = f"ws://{self.server}/user/{self.username}/api/kernels/{kernel_id}/channels?token={self.token}"
            
            websocket.enableTrace(self.debug)
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_message=self._on_message
            )
            
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            time.sleep(1)  # 等待连接建立
            return True
    
    def execute_code(self, code: str, store_result: bool = True, timeout: float = 30.0) -> List[tuple]:
        """执行代码并返回结果"""
        if store_result:
            self.execution_results = []
        
        if not self.ws:
            if not self.connect_websocket():
                return [('error', 'Failed to connect to kernel')]
        
        self._execution_completed.clear()
        msg = self._create_message('execute_request', {
            'code': code,
            'silent': False,
            'store_history': True,
            'user_expressions': {},
            'allow_stdin': False
        })
        self._msg_id = msg['header']['msg_id']
        
        self.ws.send(json.dumps(msg))
        
        if not self._execution_completed.wait(timeout):
            self.execution_results.append(('error', f'Execution timeout after {timeout} seconds'))
        
        self._msg_id = None
        return self.execution_results
    
    def execute_all_cells(self, new_code: str = None, timeout: float = 30.0) -> Tuple[bool, List[tuple]]:
        """按顺序执行所有代码单元格，包括新代码"""
        print("Executing all cells...")
        if not self.ws:
            if not self.connect_websocket():
                # print("Failed to connect to kernel")
                return False, [('error', 'Failed to connect to kernel')]
        # print("heheheh")
        try:
            notebook = self.get_notebook_content()
            # print("That's ok.")
            # print(f"Notebook: {notebook['content']['name']}")
            all_results = []
            
            # 执行现有的代码单元格
            cnt_codeblk = 1
            for idx, cell in enumerate(notebook['content']['cells'], 1):
                print(idx)
                if cell['cell_type'] == 'code':
                    code = cell['source']
                    if isinstance(code, list):
                        code = ''.join(code)
                    print(f"\n[Cell {cnt_codeblk}] Code:")
                    print("```python")
                    print(code.strip())
                    print("```")
                    
                    results = self.execute_code(code, store_result=True, timeout=timeout)
                    
                    print("\nOutput:")
                    has_error = False
                    if not results:
                        print("<no output>")
                    else:
                        for result_type, content in results:
                            if result_type == 'error':
                                has_error = True
                                print(f"Error: {content}")
                            else:
                                print(content)
                    print("\n" + "-"*50)
                    
                    all_results.extend(results)
                    if has_error:
                        print("\nExecution stopped due to error in previous cell")
                        return False, all_results
                    cnt_codeblk += 1
            
            # 处理新代码
            if new_code:
                if not self.append_code(new_code):
                    return False, all_results + [('error', 'Failed to append new code')]
                
                print(f"\n[New Cell] Code:")
                print("```python")
                print(new_code.strip())
                print("```")
                
                results = self.execute_code(new_code, store_result=True, timeout=timeout)
                
                print("\nOutput:")
                has_error = False
                if not results:
                    print("<no output>")
                else:
                    for result_type, content in results:
                        if result_type == 'error':
                            has_error = True
                            print(f"Error: {content}")
                        else:
                            print(content)
                print("\n" + "-"*50)
                
                all_results.extend(results)
                if has_error:
                    return False, all_results
            
            return True, all_results
        except Exception as e:
            return False, [('error', f'Failed to execute all cells: {str(e)}')]
    
    def close(self):
        """清理资源"""
        if self.ws:
            self.ws.close()
        if self.kernel_id:
            try:
                requests.delete(
                    f"http://{self.server}/user/{self.username}/api/kernels/{self.kernel_id}?token={self.token}"
                )
            except:
                pass

def main():
    """使用示例"""
    # 配置
    SERVER = "192.168.1.55:8999"
    USERNAME = "melon"
    TOKEN = "22f6c7e9325f448696646d88e376b58b"
    
    # 创建管理器实例
    notebook = JupyterNotebookManager(SERVER, USERNAME, TOKEN)
    # print("Notebook manager created.")
    notebook.debug = False# 设置为True可以看到更多调试信息
    # 查看是否联通，如果没有联通，输出
    try:
        # 示例代码
        new_code = '''
import os
import pandas as pd

def factor_cal_custom(data_path):
    # 加载所需数据
    net_profit_incl_min_int_inc = pd.read_parquet(os.path.join(data_path, "net_profit_incl_min_int_inc.h5"))
    inc_tax = pd.read_parquet(os.path.join(data_path, "inc_tax.h5"))
    
    # 计算净利润和税收的差额
    net_profit_after_tax = net_profit_incl_min_int_inc - inc_tax
    
    # 对差值进行排名，并按百分比形式表示
    factor = net_profit_after_tax.rank(axis=1, pct=True)
    
    # 移除全为空值的行
    factor = factor.dropna(how='all', axis=0)
    
    return factor
'''
        # new_code = """print("Hello, World!")"""
        # 执行所有代码单元格，然后执行新代码
        success, results = notebook.execute_all_cells(new_code, timeout=60.0)
        
        if success:
            print("\nAll cells executed successfully!")
        else:
            print("\nExecution failed! Check the error messages above.")
            
    finally:
        notebook.close()

if __name__ == "__main__":
    main()
