import json
import re

# full_input_text = """
#     Original Problem:
#     Prove: For every integer a, b, c ∈ ℤ such that a + b + c = 0, find all functions f: ℤ → ℤ satisfying:
#     f(a)^2 + f(b)^2 + f(c)^2 = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a)
#     Proof Tree Structure:
#     Proposition1. Main Proposition
#     ├── Auxiliary Condition1.1. Definition of Function f
#     │   └── Define f: ℤ → ℤ
#     ├── Auxiliary Condition1.2. Functional Equation
#     │   └── For all a, b, c ∈ ℤ with a + b + c = 0:
#     │       f(a)² + f(b)² + f(c)² = 2f(a)f(b) + 2f(b)f(c) + 2f(c)f(a)
#     ├── Case1. There exists r ≥ 1 such that f(r) = 0
#     │   └── Sub-Case1.1. r = 1
#     │       └── Proposition1.1.1. f is the constant zero function
#     └── Case2. f(1) = k ≠ 0
#         ├── Sub-Case2.1. f(2) = 0
#         │   └── Proposition2.1.1. f is a period 2 function
#         └── Sub-Case2.2. f(2) = 4k
#             ├── Sub-Sub-Case2.2.1. f(4) = 0
#             │   └── Proposition2.2.1. f is a period 4 function
#             └── Sub-Sub-Case2.2.2. f(4) = 16k
#                 └── Proposition2.2.2. f(x) = kx² for all x ∈ ℤ
#     """
full_input_text = """
Proposition1. Main Proposition\n├── Auxiliary Condition1.1. Known Lemma on Matrix Maxima\n├── Case1. Case 1: n is even\n│   ├── Sub-Case1.1. Construct matrix for even n\n│   └── Proposition1.1.1. Verify m = 1 + n/2 satisfies conditions\n└── Case2. Case 2: n is odd\n    ├── Sub-Case2.1. Construct matrix for odd n\n    └── Proposition2.1.1. Verify m = 1 + ⌊n/2⌋ + 1 satisfies conditions
"""



class TreeNode:
    def __init__(self, node_id, node_type, content):
        self.id = node_id
        self.type = node_type
        self.content = content
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "content": self.content,
            "dependencies": [self.parent.id] if self.parent else []
        }
        
    def print_detailed_tree(self, prefix="", is_last=True):
        # 打印当前节点
        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}[{self.type}] {self.id}")
        print(f"{prefix}{'    ' if is_last else '│   '}Content: {self.content}")
        
        # 打印子节点
        child_prefix = prefix + ("    " if is_last else "│   ")
        for i, child in enumerate(self.children):
            child.print_detailed_tree(child_prefix, i == len(self.children) - 1)

def build_tree(tree_lines, main_proposition_content):
    """构建树状数据结构"""
    root = TreeNode("Proposition1", "Proposition", main_proposition_content)
    nodes_dict = {"Proposition1": root}  # 使用字典存储所有节点，便于查找
    last_node = None  # 跟踪最后一个创建的节点
    
    node_pattern = re.compile(
        r'^(Auxiliary Condition|Proposition|Case|Sub-Case|Sub-Sub-Case)(\d+(?:\.\d+)*)(?:\.\s*)(.*)'
    )
    
    # 跳过第一行
    for line in tree_lines[1:]: 
        indent = len(line) - len(line.lstrip())
        original_line = line
        content_line = re.sub(r'^[\s│├└─]+', '', line).strip()
        
        if not content_line:
            continue

        # 检查是否是以└─开始的非节点类型行
        if '└──' in original_line and not any(type_name in content_line for type_name in [
            'Auxiliary Condition', 'Proposition', 'Case', 'Sub-Case', 'Sub-Sub-Case']):
            if last_node:
                # 将内容追加到上一个节点
                last_node.content = (last_node.content + " " + content_line).strip()
            continue

        match = node_pattern.match(content_line)
        if match:
            node_type = match.group(1)
            node_id_number = match.group(2)
            content = match.group(3).strip()
            node_id = node_type.replace(' ', '') + node_id_number

            # 创建新节点
            new_node = TreeNode(node_id, node_type, content)
            nodes_dict[node_id] = new_node
            last_node = new_node  # 更新最后一个节点
            
            # 对于Proposition类型的节点，查找完全匹配的父节点
            if node_type == 'Proposition' and node_id != 'Proposition1':
                parent_types = ['Sub-Sub-Case', 'Sub-Case', 'Case']
                for parent_type in parent_types:
                    potential_parent_id = parent_type.replace(' ', '') + node_id_number
                    if potential_parent_id in nodes_dict:
                        nodes_dict[potential_parent_id].add_child(new_node)
                        break
            else:
                # 对于其他类型的节点，使用ID序号找到父节点
                id_parts = node_id_number.split('.')
                if len(id_parts) > 1:
                    parent_id_parts = id_parts[:-1]
                    possible_parent_types = {
                        'Sub-Sub-Case': ['Sub-Case'],
                        'Sub-Case': ['Case'],
                        'Case': ['Proposition'],
                        'AuxiliaryCondition': ['Proposition']
                    }
                    
                    parent_number = '.'.join(parent_id_parts)
                    parent_type_list = possible_parent_types.get(node_type, [])
                    
                    for parent_type in parent_type_list:
                        potential_parent_id = parent_type.replace(' ', '') + parent_number
                        if potential_parent_id in nodes_dict:
                            nodes_dict[potential_parent_id].add_child(new_node)
                            break
                    else:
                        root.add_child(new_node)
                else:
                    root.add_child(new_node)
        elif last_node:
            # 处理其他类型的行，追加到最后一个节点
            last_node.content = (last_node.content + " " + content_line).strip()

    return root

def level_order_traversal(root):
    """层序遍历树结构"""
    result = []
    if not root:
        return result
        
    # 使用队列来进行层序遍历
    queue = [root]
    
    while queue:
        # 获取当前层的节点数
        level_size = len(queue)
        
        # 处理当前层的所有节点
        for _ in range(level_size):
            node = queue.pop(0)  # 取出队首节点
            result.append(node.to_dict())  # 将节点加入结果
            
            # 将子节点加入队列
            for child in node.children:
                queue.append(child)
    
    return result

def parse_tree(full_input_text=full_input_text):
    # 提取Prove内容
    prove_pattern = r"Prove:(.*?)(?=Proof Tree Structure:)"
    prove_match = re.search(prove_pattern, full_input_text, re.DOTALL)
    if prove_match:
        main_proposition_content = prove_match.group(1).strip().replace("\n", " ")
    else:
        main_proposition_content = ""

    # 提取树结构
    _, proof_structure = full_input_text.split("Proof Tree Structure:", 1)
    tree_lines = [line for line in proof_structure.strip().split('\n') if line.strip()]

    # 构建树结构
    root = build_tree(tree_lines, main_proposition_content)
    
    # 打印详细的树结构
    print("\nDetailed Tree Structure:")
    root.print_detailed_tree()

    # 通过层序遍历获取节点列表
    nodes = level_order_traversal(root)

    # 保存和输出结果
    with open('/lustre/fast/fast/txiao/zly/lean/blocktree/utils/tree.json', 'w', encoding='utf-8') as f:
        json.dump(nodes, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    import pdb
    parse_tree()