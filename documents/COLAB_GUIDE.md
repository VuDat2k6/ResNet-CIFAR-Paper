# Guide for Running Knowledge Distillation Experiments on Google Colab

To train the **Student ResNet-20** model using **Knowledge Distillation (KD)** for 200 epochs at an extremely fast speed, you should leverage the free GPU (such as T4 GPU) provided by Google Colab. Here is the step-by-step guide:

---

## Step 1: Prepare Repository on GitHub
Since you have already configured Git in your project (`VuDat2k6/ResNet-CIFAR-Paper`), save your latest changes (including the refactored directory structure and code files) and push them to GitHub:

```bash
git add .
git commit -m "Refactor project structure and prepare for Colab run"
git push
```

---

## Step 2: Initialize Google Colab
1. Access [Google Colab](https://colab.research.google.com/).
2. Select **New Notebook**.
3. Rename the notebook to `ResNet20_KD_CIFAR10.ipynb`.
4. Go to **Runtime** $\to$ **Change runtime type** $\to$ Select **T4 GPU** (or L4/A100 if you have Colab Pro) $\to$ Click **Save**.

---

## Step 3: Setup Environment and Clone Code on Colab
Create a new code cell in Colab and run the following commands to clone the repository and prepare the working directory:

```python
# 1. Clone code from your GitHub repository
!git clone https://github.com/VuDat2k6/ResNet-CIFAR-Paper.git
%cd ResNet-CIFAR-Paper

# 2. Verify CUDA is active on Colab's GPU
import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

---

## Step 4: Run Knowledge Distillation Training
Your repository already contains the pre-trained **Teacher** model checkpoint (`outputs/cifar10_seresnet/best_model.pth`). When running `scripts/train_kd.py`, the program will automatically load this Teacher model to guide the Student.

Create a new code cell and run the training command for **200 epochs**:

```python
!python scripts/train_kd.py --epochs 200 --batch_size 128 --lr 0.1 --alpha 0.6 --temp 4.0
```

> [!NOTE]
> *   On Colab's T4 GPU, each epoch takes only about **5 to 8 seconds** (compared to 108 seconds on a local GTX 1650).
> *   The total training time for 200 epochs on Colab will only take about **15 to 25 minutes**!

---

## Step 5: View Results and Plot Comparison
Once the training is complete, the results and training curves will be saved in the `outputs/cifar10_kd/` directory.
You can view the comparison curves directly inside Colab by executing:

```python
from IPython.display import Image, display
display(Image("outputs/cifar10_kd/training_curves.png"))
```

After completion, you can download the entire results directory `outputs/cifar10_kd/` to your computer, or commit and push it back to GitHub to update your academic reports.
