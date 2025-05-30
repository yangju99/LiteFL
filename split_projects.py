import os
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm 

DATA_PATH = "./chunks/all.pkl"
OUTPUT_DIR= "./chunks"

PROJECTS = ["Lang", "Chart", "Closure", "Math", "Time"]
    

# 1. 전체 chunks 불러오기
with open(DATA_PATH, 'rb') as f:
    chunks = pickle.load(f)

# 2. 프로젝트별로 분류할 딕셔너리 초기화
project_chunks = {proj: [] for proj in PROJECTS}


# 3. 각 chunk를 해당 프로젝트로 분류
for chunk in chunks:
    project_id = chunk.get('project_id', '')  # 예: "Closure_47"
    for proj in PROJECTS:
        if project_id.startswith(proj):
            project_chunks[proj].append(chunk)
            break  # 한 번 매칭되면 종료

# 4. 각 프로젝트별 chunk 개수 출력
for project_name in project_chunks:
    print(project_name +  ": ", str(len(project_chunks[project_name])))

# 5. 분류된 데이터를 개별 pkl 파일로 저장
for proj, proj_chunks in project_chunks.items():
    out_path = os.path.join(OUTPUT_DIR, f"{proj}.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump(proj_chunks, f)
    print(f"[✔] 저장 완료: {out_path} ({len(proj_chunks)}개)")
