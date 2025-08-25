import os
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import joblib

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load Dataset
data_path = r"C:\Alex The Great\Project\ecg_dataset\ecg_data.npy"
labels_path = r"C:\Alex The Great\Project\ecg_dataset\ecg_labels.npy"

X = np.load(data_path).astype('float32')
y = np.load(labels_path)
y_cat = tf.keras.utils.to_categorical(y, num_classes=5)

# Normalize per sample
X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

# Split dataset
X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = train_test_split(
    X, y, y_cat, test_size=0.2, stratify=y, random_state=42
)

# Data Augmentation
def augment_ecg(batch):
    noise = np.random.normal(0, 0.01, batch.shape)
    shift = np.random.uniform(0.9, 1.1, (batch.shape[0], 1, 1))
    return batch * shift + noise

def data_generator(X, y, batch_size=32, augment=True):
    idx = np.arange(len(X))
    while True:
        np.random.shuffle(idx)
        for i in range(0, len(X), batch_size):
            batch_idx = idx[i:i+batch_size]
            X_batch = X[batch_idx]
            y_batch = y[batch_idx]
            if augment:
                X_batch = augment_ecg(X_batch)
            yield X_batch, y_batch

# Model 1: CNN + LSTM
def create_cnn_lstm(input_shape=(2500,1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(128, return_sequences=False)(x)
    outputs = layers.Dense(5, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

cnn_lstm_model = create_cnn_lstm()
cnn_lstm_model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for Model 1
cb1 = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    callbacks.ModelCheckpoint("cnn_lstm_best.keras", save_best_only=True, monitor='val_accuracy', mode='max')
]

# Training Model 1
batch_size = 32
steps_per_epoch = len(X_train)//batch_size
validation_steps = len(X_test)//batch_size

print("Training CNN+LSTM Model...")
cnn_lstm_model.fit(
    data_generator(X_train, y_train_cat, batch_size=batch_size, augment=True),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(X_test, y_test_cat, batch_size=batch_size, augment=False),
    validation_steps=validation_steps,
    epochs=100,
    callbacks=cb1,
    verbose=2
)

cnn_lstm_model.save("cnn_lstm_final.keras")

# Evaluate Model 1
loss1, acc1 = cnn_lstm_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"CNN+LSTM Test Accuracy: {acc1:.4f} | Loss: {loss1:.4f}")

# Model 2: DenseNet1D-like
def create_densenet1d(input_shape=(2500,1)):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 7, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    for filters in [64,128,256]:
        shortcut = x
        x = layers.Conv1D(filters, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, 3, activation=None, padding='same')(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv1D(filters, 1, padding='same')(shortcut)
        x = layers.add([shortcut, x])
        x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(5, activation='softmax')(x)
    model = models.Model(inputs, outputs)
    return model

densenet1d_model = create_densenet1d()
densenet1d_model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for Model 2
cb2 = [
    callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
    callbacks.ModelCheckpoint("densenet1d_best.keras", save_best_only=True, monitor='val_accuracy', mode='max')
]

print("Training DenseNet1D-like Model...")
densenet1d_model.fit(
    data_generator(X_train, y_train_cat, batch_size=batch_size, augment=True),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_generator(X_test, y_test_cat, batch_size=batch_size, augment=False),
    validation_steps=validation_steps,
    epochs=100,
    callbacks=cb2,
    verbose=2
)

densenet1d_model.save("densenet1d_final.keras")

loss2, acc2 = densenet1d_model.evaluate(X_test, y_test_cat, verbose=0)
print(f"DenseNet1D Test Accuracy: {acc2:.4f} | Loss: {loss2:.4f}")

# Feature Extraction
feat_cnn_train = cnn_lstm_model.predict(X_train)
feat_dn_train = densenet1d_model.predict(X_train)
X_meta_train = np.concatenate([feat_cnn_train, feat_dn_train], axis=1)

feat_cnn_test = cnn_lstm_model.predict(X_test)
feat_dn_test = densenet1d_model.predict(X_test)
X_meta_test = np.concatenate([feat_cnn_test, feat_dn_test], axis=1)

# Meta-Learner
meta_learner = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3)
meta_learner.fit(X_meta_train, y_train)

y_pred_meta_prob = meta_learner.predict_proba(X_meta_test)
y_pred_meta = np.argmax(y_pred_meta_prob, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_meta)
print("Confusion Matrix:\n", cm)

plt.figure(figsize=(6,6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix (Meta-Learner)')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# ROC-AUC
roc_auc = roc_auc_score(y_test_cat, y_pred_meta_prob, multi_class='ovr')
print("Meta-Learner ROC-AUC:", roc_auc)

plt.figure(figsize=(10,8))
for i in range(5):
    fpr, tpr, _ = roc_curve(y_test_cat[:, i], y_pred_meta_prob[:, i])
    auc_score = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'Class {i} (AUC={auc_score:.2f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve (Meta-Learner)')
plt.legend()
plt.show()

# Save Models
joblib.dump(meta_learner, "meta_learner_ecg.pkl")
print("All models saved successfully.")