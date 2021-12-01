# About NSML

- 연구에 불필요한 작업들을 제거하고, GPU 자원의 효율적인 사용을 위해 개발된 MLaaS (Machine Learning as a Service), 클라우드 플랫폼
- 단순히 CLI 및 Web interface 만으로 복잡한 설정 없이 AI 학습을 진행 가능
  - CLI 명령어를 통해서 Train 및 Model Load, Save, 그리고 Submit 진행
    - Load 및 Checkpoint 이어서 훈련하는 것이 쉬웠던 점은 좋았다.
  - Web Interface 통해서 Train Log, LeaderBoard 확인
  - 단순화를 많이 시키다 보니 구조상 코드 형태 및 사용 가능한 모델에 제약이 많아 아쉬웠다.
- Session 개념이 존재
  - 각 Session 마다 Docker Container 를 `setup.py` 참고해서 올림
- Dataset 및 그 경로는 직접 접근이 불가능
  - `nsml` package 로부터 import: `from nsml import DATASET_PATH`

# How to use

See [documents](https://n-clair.github.io/ai-docs/_build/html/ko_KR/index.html) for detailed information.

```bash
# 명칭이 'nia_dm'인 데이터셋을 사용해 세션 실행하기
$ nsml run -d nia_dm
# 메인 파일명이 'main.py'가 아닌 경우('-e' 옵션으로 entry point 지정)
# 예: nsml run -d nia_dm -e main_lightgbm.py
$ nsml run -d nia_dm -e [파일명]

# 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml logs -f [세션명]

# 세션 종료 후 모델 목록 및 제출하고자 하는 모델의 checkpoint 번호 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml model ls [세션명]

# 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
$ nsml submit -t [세션명] [모델_checkpoint_번호]

# 모델 제출하기
# 제출 후 리더보드에서 점수 확인 가능
$ nsml submit [세션명] [모델_checkpoint_번호]

# session 멈추기
$ nsml stop [세션명]
```