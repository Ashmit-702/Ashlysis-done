"""Generates synthetic Mumbai-University-style Machine Learning papers (NOT real exam papers) for testing."""
import os, shutil, pymupdf
from PIL import Image, ImageDraw

def paper_lines(period, items):
    L = ["University of Mumbai", f"B.E. Semester VII Examination {period}", "Machine Learning", "Subject Code: CSC701",
         "Time: 3 Hours   Total Marks: 80", "N.B.: Question No. 1 is compulsory. Attempt any three from the remaining."]
    for q in items:
        L.append(q)
    return L

BASE = {
 '2022 May': ["Q1 Attempt any four (5 marks each)",
  "a) Explain bias and variance tradeoff with example. [5]", "b) Define overfitting and describe regularization techniques. [5]",
  "c) Explain the K-means clustering algorithm briefly. [5]", "d) What is gradient descent? Explain with a diagram. [5]",
  "Q2 (A) Explain Support Vector Machine with kernel trick. (10 marks)", "(B) Discuss the decision tree algorithm using entropy and information gain. (10 marks)",
  "Q3 (A) Explain Principal Component Analysis for dimensionality reduction. (10 marks)", "(B) Describe the Naive Bayes classifier with an example. (10 marks)"],
 '2022 Dec': ["Q1 Attempt any four (5 marks each)",
  "a) Differentiate between supervised and unsupervised learning. [5]", "b) Explain overfitting and underfitting and how to avoid them. [5]",
  "c) Describe evaluation metrics: precision, recall and F1 score. [5]", "d) Explain the bias variance trade off. [5]",
  "Q2 (A) Describe SVM and the role of the kernel function. (10 marks)", "(B) Explain K-means algorithm with a numerical example. (10 marks)",
  "Q3 (A) Explain the decision tree induction algorithm ID3. (10 marks)", "(B) Explain the back propagation algorithm in neural networks. (10 marks)"],
 '2023 May': ["Q1 Attempt any four (5 marks each)",
  "a) Explain the bias-variance tradeoff with a suitable example. [5]", "b) What is regularization? Explain L1 and L2. [5]",
  "c) Explain the working of the k nearest neighbor algorithm. [5]", "d) Explain PCA and its applications. [5]",
  "Q2 (A) Explain the support vector machine algorithm with a neat diagram. (10 marks)", "(B) Explain the random forest ensemble method. (10 marks)",
  "Q3 (A) Describe the K means clustering algorithm and its limitations. (10 marks)", "(B) Explain the Naive Bayes classifier. (10 marks)"],
 '2023 Dec': ["Q1 Attempt any four (5 marks each)",
  "a) Explain overfitting and the techniques used to reduce it. [5]", "b) Explain gradient descent and its variants. [5]",
  "c) Explain confusion matrix and accuracy measures. [5]", "d) Define bias and variance in machine learning. [5]",
  "Q2 (A) Explain SVM with kernel trick for non linear data. (10 marks)", "(B) Explain Principal Component Analysis step by step. (10 marks)",
  "Q3 (A) Explain the decision tree algorithm with information gain. (10 marks)", "(B) Explain reinforcement learning with an example. (10 marks)"],
 '2024 May': ["Q1 Attempt any four (5 marks each)",
  "a) Explain bias and variance with a suitable diagram. [5]", "b) Explain regularization to prevent overfitting. [5]",
  "c) Explain K-means clustering. [5]", "d) Explain the Naive Bayes classifier. [5]",
  "Q2 (A) Explain Support Vector Machine and kernel functions. (10 marks)", "(B) Explain the decision tree algorithm and entropy. (10 marks)",
  "Q3 (A) Explain PCA for dimensionality reduction. (10 marks)", "(B) Explain gradient boosting and ensemble learning. (10 marks)"],
}

def text_pdf(path, lines):
    d = pymupdf.open(); p = d.new_page(); y = 70
    for l in lines:
        if y > 790: p = d.new_page(); y = 70
        p.insert_text((45, y), l, fontsize=10); y += 22
    d.save(path)

def scanned_pdf(path, lines):
    img = Image.new('RGB', (1240, 1754), 'white'); dr = ImageDraw.Draw(img); y = 100
    for l in lines: dr.text((60, y), l, fill='black'); y += 36
    png = path + '.png'; img.save(png)
    d = pymupdf.open(); p = d.new_page(); p.insert_image(p.rect, filename=png); d.save(path); os.remove(png)

def build(outdir, scanned='2023 Dec'):
    os.makedirs(outdir, exist_ok=True); out = {}
    for period, items in BASE.items():
        fn = f"ML_{period.replace(' ', '_').replace('May','May').replace('Dec','Dec')}.pdf"
        path = os.path.join(outdir, fn)
        (scanned_pdf if period == scanned else text_pdf)(path, paper_lines(period, items))
        out[period] = path
    dup = os.path.join(outdir, 'ML_May_2024_copy.pdf'); shutil.copy(out['2024 May'], dup); out['dup'] = dup
    return out
