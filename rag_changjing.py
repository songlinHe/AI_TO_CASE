import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class UsernameFileFinder:
    def __init__(self, data_dir: str, threshold: float = 0.5):
        self.data_dir = Path(data_dir)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = threshold
        self.embedding_cache = {}

    def get_all_files(self) -> list:
        return [str(p) for p in self.data_dir.rglob('*') if p.is_file()]

    def get_embedding(self, text: str) -> np.ndarray:
        """带缓存的 embedding"""
        if text not in self.embedding_cache:
            self.embedding_cache[text] = self.model.encode(text, convert_to_numpy=True)
        return self.embedding_cache[text]

    def read_file(self, file_path: str) -> str:
        """读取文件内容"""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except:
            with open(file_path, "r", encoding="gbk", errors="ignore") as f:
                return f.read()

    def find_best_matched_file(self, target_usernames: list):
        """
        找出最匹配文件并读取内容。
        相似度 < 阈值（默认0.5）时直接返回 None。
        """

        if not target_usernames:
            return None

        # 目标 embedding（平均向量）
        target_embeddings = [
            self.get_embedding(u) for u in target_usernames if u.strip()
        ]

        if not target_embeddings:
            return None

        target_embedding = np.mean(target_embeddings, axis=0).reshape(1, -1)

        best_match = None
        best_score = -1

        # 遍历所有文件
        for file_path in self.get_all_files():

            usernames = [Path(file_path).stem]
            if not usernames:
                continue

            file_embeddings = [self.get_embedding(u) for u in usernames]
            file_embedding = np.mean(file_embeddings, axis=0).reshape(1, -1)

            similarity = cosine_similarity(target_embedding, file_embedding)[0][0]

            # 保留最高分
            if similarity > best_score:
                best_score = similarity
                best_match = file_path

        # 阈值过滤：小于0.5认为不相关
        if best_score < self.threshold:
            return None

        # 读取文件内容
        file_content = self.read_file(best_match)

        return {
            "file_path": best_match,
            "similarity": float(best_score),
            "content": file_content
        }

    def main(self,data_dir,target_username):
        # 你的目标目录（你需要检索的文件所在位置）
        # 初始化
        finder = UsernameFileFinder(data_dir=data_dir, threshold=0.5)
        name=[]
        name.append(target_username)
        # 示例：你可以传入想匹配的“用户名关键词”

        print("🔍 正在查找最匹配的文件...")

        result = finder.find_best_matched_file(name)
        
        print("🔍 最匹配的文件...",result)

        return result
