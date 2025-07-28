# -*- coding: utf-8 -*-
from neo4j import GraphDatabase, exceptions
import logging

# ==============================================================================
#  知识图谱生成器 (修正版 v2.0)
#  该脚本使用Python的neo4j驱动，连接到Neo4j数据库，
#  并根据预定义的数据，自动创建节点、关系和约束，
#  从而生成一个考研知识图谱的原型。
#
#  运行前置条件:
#  1. 确保你已经安装并启动了Neo4j数据库。
#  2. 安装Python的Neo4j驱动: pip install neo4j
#  3. 根据你的数据库设置，修改下面的 NEO4J_CONFIG。
# ==============================================================================

# --- Neo4j数据库连接配置 ---
# 请根据您的Neo4j数据库实例修改这里的URI, 用户名和密码
NEO4J_CONFIG = {
    "uri": "bolt://localhost:7687",  # 默认的Bolt协议URI
    "auth": ("neo4j", "password"),   # 默认用户名是'neo4j'，'password'是您设置的密码
    "database": "neo4j"              # 要操作的数据库名称
}

class KnowledgeGraphGenerator:
    """
    一个用于在Neo4j中创建知识图谱原型的类。
    """

    def __init__(self, uri, user, password, db):
        """
        初始化数据库连接。
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            self.database = db
            logging.info("成功连接到Neo4j数据库。")
        except exceptions.AuthError as e:
            logging.error(f"数据库认证失败，请检查用户名和密码: {e}")
            self.driver = None
        except Exception as e:
            logging.error(f"数据库连接失败，请检查URI和数据库是否正在运行: {e}")
            self.driver = None

    def close(self):
        """
        关闭数据库连接。
        """
        if self.driver:
            self.driver.close()
            logging.info("数据库连接已关闭。")

    def run_query(self, query, parameters=None, **kwargs):
        """
        一个通用的函数，用于在事务中执行Cypher查询。
        """
        if not self.driver:
            logging.error("驱动未初始化，无法执行查询。")
            return
        
        with self.driver.session(database=self.database) as session:
            try:
                # 使用事务来保证操作的原子性
                session.execute_write(
                    self._execute_query, query, parameters, **kwargs
                )
                logging.info(f"成功执行查询: {query[:80].strip()}...")
            except Exception as e:
                logging.error(f"执行查询失败: {e}\n查询语句: {query}\n参数: {parameters}")

    @staticmethod
    def _execute_query(tx, query, parameters=None, **kwargs):
        """私有方法，用于在事务内部执行查询"""
        tx.run(query, parameters, **kwargs)

    def clear_database(self):
        """
        清除数据库中的所有节点和关系，以便重新开始。
        这是一个危险操作，请谨慎使用。
        """
        logging.warning("正在清除数据库中的所有数据...")
        query = "MATCH (n) DETACH DELETE n"
        self.run_query(query)
        logging.info("数据库已清空。")

    def create_constraints(self):
        """
        为不同类型的节点创建唯一性约束，保证数据质量和查询性能。
        """
        logging.info("正在创建唯一性约束...")
        queries = [
            "CREATE CONSTRAINT unique_teacher_name IF NOT EXISTS FOR (n:Teacher) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT unique_kp_name IF NOT EXISTS FOR (n:KnowledgePoint) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT unique_question_id IF NOT EXISTS FOR (n:ExamQuestion) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT unique_technique_name IF NOT EXISTS FOR (n:SolvingTechnique) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT unique_difficulty_level IF NOT EXISTS FOR (n:Difficulty) REQUIRE n.level IS UNIQUE",
            "CREATE CONSTRAINT unique_chapter_name IF NOT EXISTS FOR (n:SyllabusChapter) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT unique_person_name IF NOT EXISTS FOR (n:Person) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT unique_theory_name IF NOT EXISTS FOR (n:Theory) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT unique_work_name IF NOT EXISTS FOR (n:Work) REQUIRE n.name IS UNIQUE",
        ]
        for query in queries:
            self.run_query(query)
        logging.info("唯一性约束创建完成。")

    def create_nodes_from_data(self, nodes_data):
        """
        根据标签对节点数据进行分组，并使用UNWIND批量创建，效率更高且不依赖APOC。
        """
        logging.info("正在批量创建节点...")
        # 按标签对节点数据进行分组
        nodes_by_label = {}
        for node in nodes_data:
            label = node['label']
            if label not in nodes_by_label:
                nodes_by_label[label] = []
            nodes_by_label[label].append(node['properties'])

        # 为每种标签执行一次批量创建
        for label, props_list in nodes_by_label.items():
            query = f"UNWIND $props_list as props CREATE (n:{label}) SET n = props"
            self.run_query(query, parameters={'props_list': props_list})
        
        logging.info(f"节点创建流程完成，共处理 {len(nodes_data)} 个节点定义。")


    def create_relationships_from_data(self, relationships_data):
        """
        遍历关系数据列表，为每条关系创建连接。
        这种方法不依赖APOC，通用性更强。
        """
        logging.info("正在批量创建关系...")
        for head, rel_type, tail, props in relationships_data:
            head_label, head_props = head
            tail_label, tail_props = tail
            
            # 使用节点的唯一标识符来匹配，而不是第一个属性
            # 假设'name'是大多数节点的唯一标识符，'id'是ExamQuestion的唯一标识符
            head_match_key = 'id' if 'id' in head_props else 'name'
            tail_match_key = 'id' if 'id' in tail_props else 'name'

            head_match_value = head_props[head_match_key]
            tail_match_value = tail_props[tail_match_key]

            # 构造Cypher查询语句
            cypher_query = (
                f"MATCH (a:{head_label} {{{head_match_key}: $head_val}}), "
                f"(b:{tail_label} {{{tail_match_key}: $tail_val}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) "
                f"SET r = $props"
            )
            self.run_query(cypher_query, parameters={
                'head_val': head_match_value, 
                'tail_val': tail_match_value, 
                'props': props
            })
        logging.info(f"关系创建流程完成，共处理 {len(relationships_data)} 条关系定义。")


def main():
    """
    主执行函数
    """
    # 配置日志记录
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- 1. 定义图谱数据 ---
    nodes_data = [
        {"label": "Teacher", "properties": {"name": "张宇"}},
        {"label": "Teacher", "properties": {"name": "徐涛"}},
        {"label": "KnowledgePoint", "properties": {"name": "函数极限"}},
        {"label": "KnowledgePoint", "properties": {"name": "导数定义"}},
        {"label": "KnowledgePoint", "properties": {"name": "洛必达法则"}},
        {"label": "KnowledgePoint", "properties": {"name": "泰勒公式"}},
        {"label": "KnowledgePoint", "properties": {"name": "商品二因素"}},
        {"label": "KnowledgePoint", "properties": {"name": "社会基本矛盾"}},
        {"label": "ExamQuestion", "properties": {"id": "2023-Math1-Q5", "name": "2023年数学一第5题"}},
        {"label": "SolvingTechnique", "properties": {"name": "等价无穷小替换"}},
        {"label": "Difficulty", "properties": {"level": "中等"}},
        {"label": "SyllabusChapter", "properties": {"name": "高等数学-第一章"}},
        {"label": "Person", "properties": {"name": "马克思"}},
        {"label": "Theory", "properties": {"name": "剩余价值理论"}},
        {"label": "Theory", "properties": {"name": "唯物史观"}},
        {"label": "Work", "properties": {"name": "《资本论》"}},
        {"label": "Work", "properties": {"name": "《德意志意识形态》"}}
    ]

    relationships_data = [
        # (头节点(标签, 属性), 关系类型, 尾节点(标签, 属性), 关系属性)
        (("Teacher", {"name": "张宇"}), "EXPLAINS", ("KnowledgePoint", {"name": "洛必达法则"}), {}),
        (("ExamQuestion", {"id": "2023-Math1-Q5"}), "TESTS", ("KnowledgePoint", {"name": "洛必达法则"}), {"source": "2023真题"}),
        (("ExamQuestion", {"id": "2023-Math1-Q5"}), "TESTS", ("KnowledgePoint", {"name": "导数定义"}), {"source": "2023真题"}),
        (("ExamQuestion", {"id": "2023-Math1-Q5"}), "USES_TECHNIQUE", ("SolvingTechnique", {"name": "等价无穷小替换"}), {}),
        (("ExamQuestion", {"id": "2023-Math1-Q5"}), "HAS_DIFFICULTY", ("Difficulty", {"level": "中等"}), {}),
        (("KnowledgePoint", {"name": "函数极限"}), "IS_PREREQUISITE_FOR", ("KnowledgePoint", {"name": "导数定义"}), {}),
        (("KnowledgePoint", {"name": "导数定义"}), "IS_PREREQUISITE_FOR", ("KnowledgePoint", {"name": "洛必达法则"}), {}),
        (("KnowledgePoint", {"name": "洛必达法则"}), "IS_PREREQUISITE_FOR", ("KnowledgePoint", {"name": "泰勒公式"}), {}),
        (("KnowledgePoint", {"name": "洛必达法则"}), "BELONGS_TO", ("SyllabusChapter", {"name": "高等数学-第一章"}), {}),
        (("Person", {"name": "马克思"}), "PROPOSED", ("Theory", {"name": "剩余价值理论"}), {}),
        (("Person", {"name": "马克思"}), "PROPOSED", ("Theory", {"name": "唯物史观"}), {}),
        (("Theory", {"name": "剩余价值理论"}), "DISCUSSED_IN", ("Work", {"name": "《资本论》"}), {}),
        (("Theory", {"name": "唯物史观"}), "DISCUSSED_IN", ("Work", {"name": "《德意志意识形态》"}), {}),
        (("Teacher", {"name": "徐涛"}), "EXPLAINS", ("KnowledgePoint", {"name": "商品二因素"}), {}),
    ]

    # --- 2. 创建图谱生成器实例并执行 ---
    graph_generator = KnowledgeGraphGenerator(
        NEO4J_CONFIG["uri"],
        NEO4J_CONFIG["auth"][0],
        NEO4J_CONFIG["auth"][1],
        NEO4J_CONFIG["database"]
    )

    if graph_generator.driver:
        # 按照 清空 -> 创建约束 -> 创建节点 -> 创建关系 的顺序执行
        graph_generator.clear_database()
        graph_generator.create_constraints()
        graph_generator.create_nodes_from_data(nodes_data)
        graph_generator.create_relationships_from_data(relationships_data)
        
        # 关闭连接
        graph_generator.close()

if __name__ == "__main__":
    main()

