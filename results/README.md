# 실험 산출물

Kaggle `/kaggle/working`에서 회수한 원본 로그.
노트북 출력에는 `grad_norm` / `ascent_gap`이 찍히지 않으므로 **이 JSON들이 유일한 출처**다.

| 파일 | 출처 |
|---|---|
| `arm_lion_sam_{off,on}_*.json` | WSD x SAM Overlay (ResNet-18) |
| `arm_pre_sam_{off,on}_*.json`  | Tiny-ImageNet WSD x SAM (ViT-Tiny) |
| `stage1_lion_ep40.pt`          | ResNet Stage 1 체크포인트 (v1/v2 공통 분기점) |
| `pretrain_vit_tinyimagenet.pt` | ViT 사전학습 backbone — Phase 2 재현에 필수 |

`.pt`는 `.gitignore` 처리되어 커밋되지 않는다. 로컬 보관 또는 Drive 백업할 것.
