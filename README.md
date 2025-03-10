# Capstone_Project_Industrial_OCR

# Automated Inspection of Laser-Etched Serial Numbers on Copper Surfaces Using Deep Learning Techniques  

## 📝 Overview  
This project focuses on improving the readability of laser-etched serial numbers on **contaminated copper surfaces** using **deep learning techniques**. The challenge arises due to:  
- **Low contrast and contamination**, making serial numbers difficult to read.  
- **Limited dataset (<200 samples)**, restricting fine-tuning performance.  

To overcome these issues, we:  
1. **Enhanced image readability** using **industrial cameras**.  
2. Developed a **data augmentation pipeline** leveraging **image processing** and **Generative Adversarial Networks (GANs)** to synthetically expand the dataset.  
3. Fine-tuned the **TrOCR text recognition model**, achieving an **average accuracy of 0.89**.  

---

## 🎯 Pain Points & Objectives  

| Pain Points | Solutions |
|------------|----------|
| **Hard-to-read serial numbers on metal surfaces due to contamination** | Used **industrial cameras** to enhance readability. |
| **Limited dataset (<200 samples)** | Developed a **data augmentation pipeline** using **image processing + GANs** to generate high-fidelity synthetic samples. |
| **Fine-tuning TrOCR with a small dataset might not be effective** | Leveraged synthetic data to improve model generalization, leading to **0.89 accuracy**. |

---

## 📊 **Presentation Slides**  

| Slide Topic | Link |
|------------|------|
| 📌 Project Introduction | [View Slides](#) |
| 🔍 Challenges & Solutions | [View Slides](#) |
| 🏗️ Data Augmentation Pipeline | [View Slides](#) |
| 🤖 Model Fine-Tuning & Results | [View Slides](#) |

---

## 🎥 **Project Demonstration**  

Here’s a quick overview of our **automated serial number inspection** system:  

![Project Overview](images/project_demo.gif)  

---

## 🚀 **Key Takeaways**  
✅ Improved text readability from low-quality industrial images.  
✅ Successfully generated synthetic data mimicking real-world conditions.  
✅ Achieved **0.89 accuracy** in recognizing serial numbers on copper surfaces.  

---

📌 **Author:** *Theerapat Niamhom*  
📌 **Keywords:** *Deep Learning, OCR, GAN, Image Processing, TrOCR, Industrial AI*  
