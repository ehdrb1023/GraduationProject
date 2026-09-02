# Colab CLI 조사 — VS Code에서 Colab GPU로 실험 돌리기

> 조사일: 2026-09-02

## 결론 먼저

**된다. 그리고 이 프로젝트에는 특히 잘 맞는다.**

2026년 6월 Google이 공식 `google-colab-cli`를 출시했다.
로컬 터미널에서 `colab run train.py` 하면 Colab의 원격 GPU에서 실행되고,
출력이 셸로 스트리밍되며, 산출물을 다시 받아온 뒤 VM이 자동으로 정리된다.
브라우저도 Jupyter 커널 관리도 필요 없다.

이 프로젝트에 맞는 이유:

1. **로컬에 GPU가 없다** (WSL2, `nvidia-smi` 없음) → 원격 GPU가 유일한 선택지
2. **다음 할 일이 "seed 3개 반복"** (`PROGRESS.md` §6) → 브라우저에서 셀을 3번 누르는 것보다
   `for s in 42 1337 7; do colab run train.py --seed $s; done` 이 압도적으로 낫다
3. **run 하나가 30~60분** → keep-alive 데몬이 idle 종료를 막아준다.
   브라우저 탭 세션이 끊겨 학습이 날아가는 문제가 사라진다
4. **에이전트 통합이 공식 지원** — 저장소에 `COLAB_SKILL.md`가 동봉되어 있어
   Claude Code 같은 터미널 에이전트가 직접 CLI를 몰 수 있다

---

## 설치 — ✅ 완료됨 (2026-09-02, 인증·GPU 동작까지 검증)

이 머신에 이미 설치·인증해 두었다.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv 0.12.9 → ~/.local/bin
uv tool install google-colab-cli --with "jupyter-kernel-client==0.9.0"
```

### ⚠️ `--with` 핀이 필수다 (안 하면 모든 exec가 죽는다)

colab-cli 0.6.0은 `jupyter-kernel-client`를 **버전 제약 없이** 의존성으로 건다.
그런데 그 패키지가 1.0.0(2026-07-26)에서 `KernelClient` → `JupyterKernelClient`로
심볼 이름을 바꿨다. 그냥 `uv tool install google-colab-cli` 하면 최신 1.0.2가 깔리고,
`colab new`/`status`는 되는데 **`colab exec`만 아래로 죽는다**:

```
AttributeError: module 'jupyter_kernel_client' has no attribute 'KernelClient'
```

colab-cli 0.6.0은 2026-06-16 릴리즈라 당시 최신이던 **0.9.0**이 맞는 짝이다.
이미 깨진 상태라면 아래로 복구:

```bash
uv tool install --force google-colab-cli --with "jupyter-kernel-client==0.9.0"
```

`--with`로 박아두면 나중에 `uv tool upgrade`를 해도 핀이 유지된다.
(`uv pip install`로 그 자리에서 내리는 것도 되지만 upgrade 한 번에 되돌아간다.)

- PEP 668(externally-managed) 때문에 `pip3 install`은 막혀 있어 `uv tool`로 설치했다.
- `~/.local/bin`은 `~/.bashrc:118`에 이미 PATH로 잡혀 있어 **새 셸에서 `colab`이 바로 잡힌다.**
- 플랫폼: Linux/macOS만 지원(Windows 네이티브 미지원). WSL2는 Linux로 취급되므로 문제없음.
- `requires_python >= 3.12` / 로컬 Python 3.12.3 → 충족.

### 검증 결과 (2026-09-02 실측)

```
colab new -s gputest --gpu T4     →  Session READY
colab status -s gputest           →  Hardware: T4 | Variant: GPU | Status: IDLE
colab exec -s gputest -f x.py     →  Tesla T4 / VRAM 14.6 GB
                                     torch 2.11.0+cu128, torchvision 0.26.0, python 3.13.15
                                     fp32 matmul 3.36 TFLOPS
colab stop -s gputest             →  Session terminated
```

**Kaggle에서 쓰던 것과 같은 T4다.** 기존 노트북의 에폭당 시간(ResNet-18 22.7s,
SAM 43.0s)이 그대로 적용된다고 보면 된다. torch 2.11 / torchvision 0.26이 이미 있어
별도 패키지 설치도 필요 없다.

### 인증 — ✅ 완료됨

`colab sessions`를 처음 실행하면 OAuth URL이 출력된다.
**localhost 리다이렉트가 아니라 코드 붙여넣기(out-of-band) 방식**이라
헤드리스 WSL에서도 문제없고, **gcloud 설치도 필요 없다.**
브라우저에서 승인 → 화면의 인증 코드를 터미널에 붙여넣으면 끝.

`~/.config/colab-cli/token.json`에 `refresh_token`과 함께 저장되므로
**재인증은 필요 없다.** 스코프는 `openid`, `userinfo.profile`, `userinfo.email`,
`cloud-platform`, `colaboratory`, `drive.file` 6종이 모두 부여됐다
(`colaboratory`가 있어야 keep-alive가 동작한다).

> 인증 직후 `[colab] No active sessions found on server.`가 뜨는데 **이건 에러가 아니다.**
> "아직 만든 VM이 없다"는 정상 응답이고, 이게 보이면 인증에 성공한 것이다.

에이전트/헤드리스 자동화를 원하면 ADC(`--auth=adc`) 방식도 있지만 `gcloud`가 필요하고
현재 이 머신에는 설치돼 있지 않다. 기본 oauth2로 충분하다.

---

## 명령어 레퍼런스

> 아래는 설치된 **v0.6.0의 `--help` 실측** 기준. 블로그/README와 다른 부분은 ⚠️로 표시했다.

### ⚠️ 먼저 알아야 할 함정 3가지

1. **`--timeout` 기본값이 30초다.** `colab exec`/`colab run` 둘 다.
   이 프로젝트의 run은 30~60분이므로 **반드시 명시해야 한다**:
   ```bash
   colab exec -s wsd -f GraduationProject/5_phases/1_wsd_grid.ipynb --timeout 25000
   ```
   빠뜨리면 30초 만에 잘린다. 가장 흔하게 밟을 지뢰.
2. **`--high-mem` 플래그는 v0.6.0에 없다.** README에는 언급돼 있지만 실제 CLI에는 미구현.
3. **인식 못 하는 `--gpu` 값은 조용히 A100으로 폴백**한다(그리고 대개 다음 단계에서 실패).
   오타에 주의. `--gpu`에 accelerator를 주고 400이 나면 그 계정에 해당 GPU 할당량이 없다는 뜻이니
   `--gpu T4`로 내리거나 생략(CPU)할 것.

### 세션 관리
| 명령 | 설명 |
|---|---|
| `colab new [-s NAME] [--gpu GPU] [--tpu TPU]` | VM 런타임 할당 |
| `colab sessions` | 활성 세션 목록 |
| `colab status [-s NAME]` | 하드웨어·세션 메타데이터 |
| `colab stop [-s NAME]` | 세션 종료 |
| `colab restart-kernel [-s NAME]` | 커널 재시작 |
| `colab url [-s NAME] [--open]` | 브라우저 URL 출력/열기 |

### 코드 실행
| 명령 | 설명 |
|---|---|
| `colab exec [-s N] [-f FILE] [--output-image P] [--timeout S]` | Python 코드 / `.py` / `.ipynb` 실행 |
| `colab run [--gpu G] [--keep] [-s N] [--timeout S] SCRIPT [ARGS...]` | **새 VM 생성 → 실행 → 자동 종료** |
| `colab repl [-s NAME]` | 대화형 Python REPL |
| `colab console [-s NAME]` | 원격 VM의 raw TTY 셸 (tmux) |

`exec` 세부:
- `-f nb.ipynb`를 주면 각 코드 셀을 실행하고 결과를 `<basename>_output.ipynb`로 옆에 쓴다.
- **커널 상태는 같은 세션의 `exec` 호출 사이에 유지된다.** import·변수·함수가 살아 있으므로
  매번 전부 다시 import할 필요가 없다. 초기화하려면 `colab restart-kernel` 또는 `colab stop`.
- 기본 작업 디렉터리는 `/content`. 파일은 절대경로(`/content/...`)로 다루는 게 안전하다.

`run` 세부:
- **종료 코드가 전파된다** — 스크립트의 예외/`sys.exit(N)`이 그대로 `colab run`의 exit code가 된다.
  시드 스윕 루프에서 실패를 감지할 수 있다.
- **스트림이 분리된다** — CLI의 `[colab] ...` 로그는 stderr, 스크립트 출력은 stdout.
  `colab run job.py > out.txt` 하면 스크립트 출력만 깔끔하게 잡힌다.
- 존재하지 않는 스크립트 경로면 **VM 할당 전에** 실패한다(컴퓨트 낭비 없음).
- shebang으로도 쓸 수 있다: `#!/usr/bin/env -S colab run --gpu T4 --timeout 7200`

### 파일
| 명령 | 설명 |
|---|---|
| `colab upload [-s NAME] LOCAL REMOTE` | 업로드 |
| `colab download [-s NAME] REMOTE LOCAL` | 다운로드 |
| `colab ls / rm / edit` | 원격 파일 목록 / 삭제 / 로컬 편집 |

### 유틸
| 명령 | 설명 |
|---|---|
| `colab auth [-s NAME]` | GCP 서비스 인증 |
| `colab drivemount [-s NAME] [PATH]` | Google Drive 마운트 |
| `colab install [-s NAME] [-r FILE \| PKG...]` | `uv`로 패키지 설치 |
| `colab log [-s NAME] [-n N] [-o FILE]` | 실행 히스토리 조회/내보내기 (ipynb·md·json) |

**가속기**: GPU `T4` `L4` `G4` `H100` `A100` / TPU `v5e1` `v6e1`
가속기 가용성은 **구독 티어로 게이팅**된다. 상위 GPU는 Colab Pro/Pro+가 필요하고,
계정에 따라 CPU만 할당될 수도 있다 — GPU가 붙을 거라고 가정하지 말 것.

세션이 하나뿐이면 `-s` 생략 가능하지만, `colab new`에 이름을 안 주면 랜덤 6자리 hex가
자동 생성되어 이후 명령이 모호해진다. **항상 `-s <이름>`을 붙이는 편이 낫다.**

---

## 이 프로젝트에 적용하는 법

### 두 가지 사용 패턴

**패턴 A — 지속 세션 (데이터를 재사용할 때, 권장)**

지금 노트북들은 매번 CIFAR-10(129 MB) / Tiny-ImageNet(248 MB)을 새로 받는다.
`colab run`은 매번 VM을 버리므로 그때마다 다시 받게 된다. 세션을 살려두는 편이 낫다.

```bash
colab new  -s wsd --gpu T4
colab exec -s wsd -f GraduationProject/5_phases/1_wsd_grid.ipynb --timeout 25000
colab download -s wsd /content/out/phase1_results.tar.gz ./results/
colab stop -s wsd        # ★ 반드시. 안 끄면 컴퓨트 유닛이 계속 소모된다
```

`colab exec -f` 는 `.ipynb` 를 셀 단위로 실행하고 결과를 `<이름>_output.ipynb` 로 남긴다.
따라서 **노트북 하나로 웹 실행과 CLI 실행을 모두 커버**한다.
데이터 다운로드와 Stage 1 체크포인트가 한 세션 안에서 재사용되므로,
arm 마다 VM 을 새로 띄우는 `colab run` 보다 이쪽이 유리하다.

설정(시드·rho·에폭)은 노트북 9번 셀 하나에만 있다. 바꾸고 다시 `colab exec` 하면
이미 끝난 arm 은 건너뛰고 이어서 진행된다.

### 마이그레이션 — ✅ 완료 (`5_phases/*.ipynb`)

아카이브 노트북을 헤드리스로 돌리려면 세 가지가 걸렸는데 전부 정리했다.

| 문제 | 처리 |
|---|---|
| 경로 하드코딩 (`/kaggle/working`, `/content/drive/...`) | `DATA_DIR` / `OUT_DIR` 환경변수 (기본 `/content/...`) |
| "셀을 두 번 실행하세요" 패턴 | `stage2(seed, rho, ...)` 함수 호출로 대체 |
| `plt.show()` 의존 | 결과는 arm 별 JSON, 그림은 파일로 저장 |

덤으로 아카이브 노트북의 실제 결함 세 가지도 같이 고쳤다
(`docs/PROGRESS.md` §6). 가장 중요한 건 **Stage 2 에 시드가 없어서
두 arm 이 다른 데이터 순서를 봤던 것**이다.

---

## 주의할 점

- **`--timeout`을 빠뜨리지 말 것.** 기본 30초. 이 프로젝트 최대 지뢰. (위 함정 §1)
- **세션은 자동으로 안 꺼진다.** keep-alive 데몬이 24시간 상한까지 살려두므로,
  `colab stop -s <이름>`을 안 하면 **컴퓨트 유닛이 계속 소모된다.**
  `colab run`은 스크립트가 에러로 죽어도 스스로 정리한다(`--keep` 없을 때).
  현재 세션 확인은 `colab sessions`.
- **VM은 언제나 임시(ephemeral)다.** 체크포인트(`stage1_*.pt`)는 `colab download`로
  로컬에 회수하거나 `colab drivemount`로 Drive에 저장해야 한다.
  현재 분기 프로토콜이 Stage 1 체크포인트에 의존하므로 이 부분이 특히 중요하다.
- **무료 티어는 T4까지**, 상위 GPU는 Pro/Pro+ 필요. 계정 상태에 따라 CPU만 나올 수도 있다.
  현재 실험이 T4에서 30~60분이므로 무료로도 충분하지만, Tiny-ImageNet 사전학습(35.8분)을
  자주 반복할 거면 유료를 고려할 만하다. (`colab pay`로 컴퓨트 유닛 관리 페이지가 열린다.)
- **패키지 설치는 대개 불필요.** Colab VM에 torch/torchvision/matplotlib이 이미 있다.
  필요하면 `colab install -s <이름> <패키지>` (내부적으로 `uv pip install --system`).
- **`colab repl` / `console` / `auth` / `drivemount`는 TTY가 필요하다** — 스크립트나
  에이전트에서 호출하면 멈춘다. 사람이 터미널에서 직접 실행할 것.
- 실패 시 진단은 `colab log -s <이름>`. `colab log -s <이름> -o run.ipynb`로
  세션 전체를 노트북으로 내보낼 수 있어 **실험 기록 보존용으로 유용하다.**
- Kaggle을 계속 쓸 이유가 남아 있다면 (주당 30시간 GPU 무료 등) 병행해도 된다.
  다만 **경로 분기 로직을 스크립트 상단 한 곳으로 모아두는 게** 유지보수에 낫다.

### 참고 파일
- `COLAB_SKILL.md` — CLI에 동봉된 공식 에이전트용 가이드 (`colab skill` 출력을 저장해 둔 것).
  Claude Code가 이 CLI를 몰 때 참조한다.

---

## 출처

- [Introducing the Google Colab CLI — Google Developers Blog](https://developers.googleblog.com/introducing-the-google-colab-cli/)
- [googlecolab/google-colab-cli — GitHub](https://github.com/googlecolab/google-colab-cli)
- [google-colab-cli — PyPI](https://pypi.org/project/google-colab-cli/)
- [Google Colab CLI opens runtimes to Claude Code and Codex — Help Net Security](https://www.helpnetsecurity.com/2026/06/08/google-colab-command-line-interface-cli/)
- [Google Colab CLI fuses local terminals with cloud GPUs — Developer Tech](https://www.developer-tech.com/news/google-colab-cli-local-terminal-cloud-gpus/)
- [Colab local runtimes (Docker 방식, 로컬 GPU가 있을 때의 대안)](https://research.google.com/colaboratory/local-runtimes.html)
