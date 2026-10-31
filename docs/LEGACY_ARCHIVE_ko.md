# 이전 연구 보관 안내

[English](LEGACY_ARCHIVE.md) | **한국어**

이전의 물체 인식·TripoSplat 연구는 현재 연구 checkout 밖에 보관되어 있다.

`/Users/jongbinkim/Documents/ChatGPT/disco-geo-legacy-20261028-7cb6a52`

결과 커밋: `78649360cb01b9d80056db4f0785c8e98b33bedd` (`result commit with readme`). 2026-10-31 전까지는 보관 경로에 들어 있는 해시인 `7cb6a52ba4517a9d549464d2be6ccc9e84cbbab6`였다. 이 이력을 처음 push하기 전에, 데이터 파일 하나가 GitHub의 100 MiB 한도를 넘어서 push하지 않은 커밋들에서 `runs/`와 `datasets/`를 지웠다. 원래 커밋은 로컬 브랜치 `server-backup-before-strip`에 남아 있고, 데이터 자체는 보관본에 있다.

보관본에는 이전 소스, 문서, 데이터셋, GPU checkpoint, 결과, 그리고 복사한 Linux `.venv.cuda`가 들어 있다. `MANIFEST.json`은 옮긴 경로와 복사한 메타데이터의 checksum을 정확히 기록한다. 소스 경로는 버리지 않고 같은 파일시스템 안에서 이름만 바꿨다. `.git` 이력과 사용 가능한 macOS `.venv`는 그대로 두었다. 이후 위에 적은 대로 데이터 폴더를 빼기 위해서만 이력을 재작성했다.

현재 작업 트리는 Memory Maze를 이용한 제한된 reward의 공간 능력 평가 연구로 다시 구성되었다. 이전 결과 수치는 이 새 연구를 설명하지 않는다.
