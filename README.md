# Automated Inspection of Laser-Etched Serial Numbers on Copper Surfaces Using Deep Learning Techniques

![Project Overview](slideshow/capstone.gif)

## 📝 Overview

This project focuses on improving the readability of laser-etched serial numbers on **contaminated copper surfaces** using **deep learning techniques**. The project is my **4th-year capstone project** in **Computer Science** at **King Mongkut’s University of Technology Thonburi (KMUTT)**. The project showcases my **research and development skills** in the filed of \*_computer vision_.

---

## 🎯 Pain Points

- **Low contrast and contamination**, making serial numbers difficult to read.
- **Limited dataset (<200 samples)**, restricting fine-tuning performance.

---

## 🎯 Objectives

| Challenges                                                        | Solutions                                                                                                                               |
| ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Hard-to-read, Low contrast, contamination**                     | - Used **industrial cameras** to enhance readability. <br> - Developed a **robust text recognition model** to improve reading accuracy. |
| **Limited dataset (<200 samples)**                                | Developed a **data augmentation pipeline** using **image processing + GANs** to generate high-fidelity synthetic samples.               |
| **Fine-tuning TrOCR with a small dataset might not be effective** | Leveraged synthetic data to improve model generalization, leading to **0.89 accuracy**.                                                 |

---

## 📊 **Presentation Slides**

| Slide Topic                    | Link             |
| ------------------------------ | ---------------- |
| 📌 Project Introduction        | [View Slides](#) |
| 🔍 Challenges & Solutions      | [View Slides](#) |
| 🏗️ Data Augmentation Pipeline  | [View Slides](#) |
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

📌 **Author:** _Theerapat Niamhom_  
📌 **Keywords:** _Deep Learning, OCR, GAN, Image Processing, TrOCR, Industrial AI_
