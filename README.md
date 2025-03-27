# 🧠 Transfer Learning - Image Classification

L'objectif est d'entraîner un modèle pour résoudre des problèmes de classification d'images.

Projet réalisé par **Saidane Adam**

---

##  Technologies utilisées

- Python 3.12
- PyTorch
- torchvision
- matplotlib
- PIL (Pillow)

---

##  Étapes du projet

1. **Prétraitement des données** :
   - Découpage automatique des images en `train` (80%) et `val` (20%) via `separation_data.py`

2. **Modèle utilisé** :
   - Réseau **ResNet18** pré-entraîné sur **ImageNet**
   - Fine-tuning sur **6 classes** :
     `elephant`, `gorilla`, `leopard`, `lion`, `panda`, `rhinoceros`

3. **Résultats** :
   - Accuracy validation atteinte : `100%`
   - Modèle sauvegardé dans `data_model/final_model_ft.pth`

4. **Test de performance** :
   - Prédictions fiables sur des images de test jamais vues (dans le dossier `image_test/`)

---

##  Pour commencer

### 1. Cloner le projet et activer l’environnement virtuel

```bash
git clone https://github.com/admm7/Transfer-Learning-Image-Classification.git
cd Transfer-Learning-Image-Classification

python -m venv env
env\Scripts\activate     # Pour Windows
```

### 2. Installer les dépendances

```bash
pip install torch torchvision matplotlib pillow
```

> (Tu peux aussi utiliser `pip install -r requirements.txt` si le fichier est fourni.)

---

##  Utilisation du projet

### 🔹 Étape 1 : Séparer les données

```bash
python main_code/separation_data.py
```

### 🔹 Étape 2 : Entraîner le modèle

```bash
python main_code/transfer_learning_test.py
```

### 🔹 Étape 3 : Tester le modèle

```bash
python main_code/test_model.py
```

---

##  Exemple de sortie

```
image_test/gorilla.jpg -> Predicted: gorilla
image_test/leopard.jpg -> Predicted: leopard
image_test/lion.jpg -> Predicted: lion
...
```

---

## 📁 Arborescence du projet

```
Transfer-Learning-Image-Classification/
├── data_model/
│   └── final_model_ft.pth
├── image_test/
├── main_code/
│   ├── separation_data.py
│   ├── transfer_learning_test.py
│   └── test_model.py
├── model_donnee/
├── separation_donnee/
│   ├── train/
│   └── val/
├── LICENSE
└── README.md
```

---

> ✨ _Projet personnel réalisé pour se former à l'utilisation de PyTorch et du Transfer Learning._

