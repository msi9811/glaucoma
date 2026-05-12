```markdown
# XAI-Based Glaucoma Diagnosis via Multi-Input DNN with SERS Spectra

This repository contains the official implementation of the multi-input deep learning framework for glaucoma diagnosis using surface-enhanced Raman scattering (SERS) spectra of aqueous humor (AH). This project integrates Explainable AI (XAI) to optimize model architecture and ensure biochemical interpretability.

## 📌 Project Overview
Glaucoma diagnosis traditionally relies on structural imaging, which often captures irreversible damage. Our approach utilizes the biochemical signatures within the intraocular microenvironment (AH) to establish a label-free, highly accurate (97.1% accuracy, AUC 0.986) diagnostic platform.

### Key Features
- **XAI-Driven Feature Extraction**: Utilizes first-layer weight analysis to identify core spectral regions (e.g., glutamine and acetylcarnitine-related peaks).
- **Multi-Input DNN Architecture**: Parallelly processes the whole spectrum and specific core feature regions to maximize diagnostic performance.
- **Biochemical Validation**: Mathematical evidence is cross-referenced with standard Raman peaks of glaucoma-related metabolites.

## 🛠 Installation
```bash
git clone [https://github.com/your-username/XAI-Glaucoma-SERS.git](https://github.com/your-username/XAI-Glaucoma-SERS.git)
cd XAI-Glaucoma-SERS
pip install -r requirements.txt

```

## 🚀 Usage

### 1. Data Preprocessing

Prepare your SERS spectral data in `.csv` or `.npy` format.

```bash
python preprocess.py --input ./raw_data --output ./processed_data

```

### 2. Model Training & XAI Analysis

To train the multi-input DNN and perform first-layer weight analysis:

```bash
python train.py --model multi_input --epochs 100 --batch_size 32

```

## 📊 Results

Our model achieves exceptional performance across independent datasets:

* **Accuracy**: 97.1%
* **Sensitivity**: 97.9%
* **AUC**: 0.986

The XAI analysis identified critical diagnostic regions at **399.25–600.50 cm⁻¹** and **1,998.75–2,401.25 cm⁻¹**.

## 📄 Citation

If you find this code useful for your research, please cite:

```text
(저널 게재 후 정식 인용 문구 삽입 예정)
Mun, S. I., et al. "Development of a Deep Learning Model for Glaucoma Diagnosis Using Aqueous Humor Raman Spectra: An XAI-Based Region-Specific Input." (2026).

```

## 📫 Contact

**Yun Hak Kim** - yunhak10510@pusan.ac.kr

**Seong Ik Mun** - (사용자 이메일 주소)

```

---

### 2. .gitignore 및 License 설정 추천

두 설정 모두 **"On (사용)"**을 강력하게 추천합니다. 그 이유는 다음과 같습니다.

#### **① .gitignore: On (필수)**
딥러닝 프로젝트에는 깃허브에 올라가면 안 되는 파일들이 많습니다.
* **추천 이유:** * **대용량 데이터 보호:** 수백 메가바이트가 넘는 RAW 데이터나 학습된 모델 파라미터(`.pth`, `.h5`) 파일이 올라가 저장소가 무거워지는 것을 방지합니다.
    * **환경 깨짐 방지:** `__pycache__`나 로컬 설정 파일(`.env`)이 꼬이는 것을 막아줍니다.
* **설정 방법:** Repository 생성 시 `Add .gitignore`에서 **"Python"** 템플릿을 선택하세요. 그 후 파일 안에 `/data/`, `/checkpoints/`, `*.log` 등을 추가로 적어주시면 안전합니다.

#### **② License: On (강력 권장)**
학술적 코드를 공유할 때는 라이선스가 있어야 연구 성과를 보호받을 수 있습니다. 라이선스가 없으면 다른 사람들이 이 코드를 써도 되는지 법적으로 알 수 없어 오히려 인용을 꺼리게 됩니다.
* **추천 라이선스:**
    * **MIT License:** 가장 대중적입니다. 누구나 자유롭게 사용하고 수정할 수 있지만, 반드시 원저작자(교수님과 연구팀)를 명시(인용)해야 합니다.
    * **Apache License 2.0:** MIT와 비슷하지만, 특허권에 대한 보호 조항이 더 강력합니다. 대학 연구소나 기업 협업 프로젝트에서 선호합니다.
* **설정 방법:** 가장 무난한 것은 **MIT License**입니다. 생성 시 `Choose a license`에서 선택하시면 됩니다.

**요약하자면:**
1.  **README:** 연구의 '신뢰성(XAI)'과 '성능(97.1%)'을 강조하는 영문 버전으로 작성하세요.
2.  **.gitignore:** **Python** 템플릿으로 **On** 하세요.
3.  **License:** **MIT** 또는 **Apache 2.0**으로 **On** 하여 연구 인용을 공식화하세요.

이 설정대로 깃허브를 관리하시면 저널 리뷰어들에게도 "연구 프로세스가 투명하고 체계적이다"라는 인상을 줄 수 있습니다.

```
