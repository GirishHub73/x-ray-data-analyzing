# Chest X-ray Mass/Nodule Image Classification
An educational project that trains image-classification models to identify chest X-rays carrying the project’s Mass/Nodule target label. It includes two workflows: TensorFlow/Keras and PyTorch with CUDA.
> **Important:** This is a learning and research project, not a medical device. Its predictions must not be used to diagnose or guide patient care.
## Project purpose
The project explores how image classification can be applied to chest X-rays and compares two popular Python deep-learning frameworks. It may help students understand how computer vision could support future research on organizing or prioritizing medical images. Any real-world use would require much stronger testing and clinical validation.
## Dataset
- **Source:** NIH Clinical Center ChestX-ray14 dataset: [official NIH download page](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Images used in this project:** 95,756 total
  - Training: 79,006 images
  - Validation: 16,750 images
- **Training labels:** 71,253 negative and 7,753 positive images. The positive class is the project’s Mass/Nodule target.
- Images are not stored in this code repository. Download the dataset from its source and point the training scripts to the local or Colab data directory.
The classes are imbalanced: about 9.8% of training images are positive. For that reason, accuracy alone does not describe model quality; precision, recall, and ROC-AUC are also reported.
## How the project works
1. Load labeled chest X-ray images from the configured data directory.
2. Resize and normalize the images for the selected model. The PyTorch transfer-learning workflow uses ImageNet normalization values.
3. Train a binary classifier for the project’s Mass/Nodule target.
4. Evaluate on the validation split and save model weights and training metrics.
The PyTorch implementation uses **EfficientNet-B0 transfer learning** with an ImageNet-pretrained model. Its ImageNet channel normalization uses means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`. It uses a positive-class loss weight of `9.1904` to account for the training class imbalance.
## Technology
- Python
- TensorFlow/Keras
- PyTorch and torchvision
- NVIDIA CUDA for the PyTorch GPU workflow
- GitHub for code, PyCharm for local development, and Google Colab as an optional hosted runtime
## Run locally on Windows
Run commands from the repository root. Create and activate a virtual environment, then install the project dependencies. If the repository includes `requirements.txt`, use:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Make sure the dataset directory configured in the training script matches where the images and labels are stored.
### TensorFlow/Keras workflow
```powershell
python -m src.train
```
The reported TensorFlow run completed 8 initial epochs followed by 3 additional epochs. It saved the model as `models/mass_nodule_model.keras`.
**Windows GPU note:** The TensorFlow run shown for this project used the CPU on native Windows. TensorFlow 2.11 and later do not provide CUDA GPU support on native Windows; use Google Colab or WSL2 for a TensorFlow GPU run. See the [TensorFlow Windows guidance](https://www.tensorflow.org/install/source_windows).
### PyTorch CUDA workflow
First check that PyTorch can see the NVIDIA GPU:
```powershell
python src/cuda_check.py
```
Then train with the full training split:
```powershell
python -m src.train_pytorch_GPU --max-images 0
```
`--max-images 0` means use all 79,006 training images. The reported run used batch size 32, learning rate `0.0001`, and the NVIDIA RTX PRO 2000 Blackwell Generation Laptop GPU. The log confirmed that the model, images, and labels were on `cuda:0`.
The PyTorch model is saved to `models/mass_nodule_model.pth`; training metrics are saved to `results/training_history.csv`. Image reading and preparation still use the CPU, while model training runs on CUDA.
## Run in Google Colab
1. Open a notebook and select **Runtime → Change runtime type → GPU**.
2. Clone the GitHub repository and install its dependencies.
3. Make the dataset available in the Colab session and update the configured data path.
4. Run either workflow from the repository root:
```python
!python -m src.train
!python -m src.train_pytorch_GPU --max-images 0
```
Run one command at a time. In Colab, check the notebook’s GPU runtime status before starting training.
## Reported validation results
| Workflow | Validation accuracy | Precision | Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| TensorFlow/Keras, final reported fine-tuning epoch | 57.54% | 15.23% | 66.42% | 66.11% |
| PyTorch/CUDA, epoch 8 | 68.18% | 17.31% | 53.55% | 66.81% |
These runs used different framework implementations and training settings, so the numbers are not a controlled head-to-head comparison. The TensorFlow run had higher recall; the PyTorch run had higher precision and ROC-AUC in the reported epochs. Both runs had low precision, meaning many positive predictions were false alarms. A separate untouched test-set result was not included in the logs.
### Metric meanings
- **Accuracy:** fraction of all images classified correctly; it can be misleading when classes are imbalanced.
- **Precision:** among images predicted positive, fraction that were labeled positive.
- **Recall (sensitivity):** among positive images, fraction the model detected. Higher recall means fewer false negatives, usually at the cost of more false positives.
- **ROC-AUC:** how well the model ranks positive images above negative images across thresholds.
## Limitations and next steps
- The positive class is much smaller than the negative class.
- The reported precision and ROC-AUC show that model performance needs improvement.
- Tune the decision threshold using validation data and report the precision/recall tradeoff.
- Verify that images from the same patient do not appear in both training and validation. Evaluate the final model on a separate, untouched test set.
- Test on data from other sources before drawing conclusions about performance beyond this dataset.
## References
- [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [TensorFlow GPU and Windows guidance](https://www.tensorflow.org/install/source_windows)
- [PyTorch local installation selector](https://pytorch.org/get-started/locally/)
