# GET 324 Group ME11: Fresh versus Rotten Orange Classifier

This project completes Laboratory Exercise 10 by training, evaluating and
deploying a binary image-classification model. The application accepts an
orange photograph and predicts either:

- `fresh_orange` (class 0)
- `rotten_orange` (class 1)

## Tools

- Google Colab
- Python
- TensorFlow and Keras
- Scikit-learn
- Git and GitHub
- Streamlit Community Cloud

## Dataset

The Colab notebook downloads the
[Fruits Fresh and Rotten for Classification dataset](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification).
The dataset contains apple, banana and orange images. Only the fresh-orange and
rotten-orange folders are retained. The notebook verifies each image, removes
duplicate files and creates stratified training, validation and test sets.

## Repository structure

Upload the contents of this folder directly to the GitHub repository root:

```text
get324-me11-fresh-rotten-orange/
├── .streamlit/
│   └── config.toml
├── tests/
│   └── test_app_helpers.py
├── app.py
├── inference_utils.py
├── ME11_Fresh_vs_Rotten_Orange_Colab.ipynb
├── fresh_rotten_orange_model.keras
├── model_info.json
├── CONTRIBUTORS.md
├── PROJECT_REPORT.md
├── README.md
└── requirements.txt
```

Do not upload an outer folder with spaces or a name ending in `(1)`. Streamlit
must see `app.py` and `requirements.txt` directly on the repository's first
page.

## Procedure 1: Train in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **File > Upload notebook**.
3. Upload `ME11_Fresh_vs_Rotten_Orange_Colab.ipynb`.
4. Select **Runtime > Change runtime type > T4 GPU > Save**.
5. Run every cell from top to bottom.
6. If Kaggle requests authentication, create a Kaggle API token and follow the
   notebook prompt. Never upload the API token to GitHub.
7. Inspect the image counts and sample images before training.
8. Record the training curves, test accuracy, precision, recall, F1 score and
   confusion matrix.
9. Download:
   - `fresh_rotten_orange_model.keras`
   - `model_info.json`
10. Replace the placeholder `model_info.json` in this project with the Colab
    version.
11. Put `fresh_rotten_orange_model.keras` beside `app.py`.

The notebook uses 70% of the verified images for training, 15% for validation
and 15% for testing. The validation set controls early stopping and threshold
selection. The test set is used only for final evaluation.

## Procedure 2: Test the application

Python 3.11 is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Google Colab alternative

The model training is completed in Colab. Local Streamlit testing is useful but
not compulsory if the computer cannot install TensorFlow. After uploading all
files to GitHub, Streamlit Community Cloud can perform the deployment test.

## Procedure 3: Create the GitHub repository

1. Sign in to GitHub.
2. Create a public repository named:

   ```text
   get324-me11-fresh-rotten-orange
   ```

3. Open this project folder on the computer.
4. Select all files inside it, not the outer folder.
5. Drag the selected contents to **Add file > Upload files** on GitHub.
6. Commit the upload to the `main` branch.
7. Confirm that `app.py`, `requirements.txt`, `model_info.json` and
   `fresh_rotten_orange_model.keras` appear at the repository root.

## Procedure 4: Deploy with Streamlit

1. Open [Streamlit Community Cloud](https://share.streamlit.io/).
2. Sign in through the GitHub account that owns the repository.
3. Select the workspace matching that GitHub username.
4. Click **Create app**.
5. Select the new repository.
6. Set:

   ```text
   Branch: main
   Main file path: app.py
   Python version: 3.11
   ```

7. Deploy and watch the installation log.
8. Test the public URL.
9. Add the URL to `PROJECT_REPORT.md`.

## Testing checklist

Use images that were not used for training:

| Image | Expected response |
|---|---|
| Clear fresh orange | Fresh Orange |
| Clear rotten orange | Rotten Orange |
| Early decay | Result with confidence shown |
| Blurry orange | Uncertain warning where appropriate |
| Apple or banana | Must be recorded as an unsupported test |
| Person, vehicle or building | Must be recorded as an unsupported test |

The application rejects low-confidence results. However, a binary classifier
can still assign high confidence to an unrelated image. Therefore, the user
interface clearly limits the application to orange photographs. A future
version should use a separate orange-versus-not-orange validator before the
fresh-versus-rotten classifier.

## Evidence to retain

Save screenshots showing:

- Dataset download and valid image counts
- Fresh and rotten sample images
- Training and validation curves
- Selected decision threshold
- Test metrics and confusion matrix
- Correct and incorrect test predictions
- GitHub repository contents
- Streamlit application and public URL

## Deliverables

- Complete `app.py`
- Saved Keras model
- `model_info.json`
- GitHub repository
- Deployed Streamlit URL
- 100 to 150 word report
- Group members' names, registration numbers and GitHub usernames
- Observations, challenges and solutions

## Limitation

This remains a binary fresh-orange versus rotten-orange classifier, as required
by the exercise. The confidence score describes preference between those two
classes; it does not prove that the uploaded object is an orange.
