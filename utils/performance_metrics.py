"""
TODO: add descriptions
"""
import matplotlib.pyplot as plt
import pickle
import cv2
from scipy.signal import savgol_filter
import numpy as np


def extract_losses(file_path):
    """
    Extract loss values from a training output file.

    :param file_path: Path to the text file containing training output
    :return: A dictionary containing arrays of loss values for each loss type
    """
    losses = {
        'total_loss': [], 'cls_loss': [], 'box_loss': [], 'model_loss': [],
        'val_total_loss': [], 'val_cls_loss': [], 'val_box_loss': [], 'val_model_loss': []
    }

    with open(file_path, 'r') as file:
        for line in file:
            if 'total_loss:' in line:
                parts = line.split(' - ')
                for part in parts:
                    for loss_type in losses.keys():
                        if part.strip().startswith(loss_type + ':'): # TODO: this condition can be improved
                            value = float(part.split(': ')[1])
                            losses[loss_type].append(value)

    # Print debug information
    for loss_type, values in losses.items():
        print(f"Number of {loss_type} values: {len(values)}")

    return losses


def convergence_analysis(epochs, train_total_loss, val_total_loss, smoothing=True, log_scale=False):
    """
    Analyze and visualize the convergence of training and validation losses.

    Parameters:
    epochs (int): Total number of epochs
    train_total_loss (list): Training loss values
    val_total_loss (list): Validation loss values
    smoothing (bool): Apply smoothing to the loss curves (default: True)
    log_scale (bool): Use logarithmic scale for y-axis (default: False)

    Returns:
    None (displays the plot)
    """
    epochs_range = list(range(1, int(epochs) + 1))

    # Ensure all data have the same length
    min_length = min(len(epochs_range), len(train_total_loss), len(val_total_loss))
    epochs_range = epochs_range[:min_length]
    train_total_loss = train_total_loss[:min_length]
    val_total_loss = val_total_loss[:min_length]

    # FIXME

    # Apply smoothing if requested
    if smoothing:
        window_length = min(51, len(train_total_loss) - 2)  # Must be odd and less than data length
        if window_length % 2 == 0:
            window_length -= 1
        train_total_loss_smooth = savgol_filter(train_total_loss, window_length, 3)
        val_total_loss_smooth = savgol_filter(val_total_loss, window_length, 3)

    # Create the plot
    plt.figure(figsize=(12, 7))
    plt.plot(epochs_range, train_total_loss, label='Training Loss', color='#1f77b4', alpha=0.3)
    plt.plot(epochs_range, val_total_loss, label='Validation Loss', color='#ff7f0e', alpha=0.3)

    if smoothing:
        plt.plot(epochs_range, train_total_loss_smooth, label='Training Loss (Smoothed)', color='#1f77b4', linewidth=2)
        plt.plot(epochs_range, val_total_loss_smooth, label='Validation Loss (Smoothed)', color='#ff7f0e', linewidth=2)

    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Total Loss', fontsize=12)
    plt.title('Convergence Analysis', fontsize=16)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    if log_scale:
        plt.yscale('log')

    # Add annotations for minimum loss values
    min_train_epoch = np.argmin(train_total_loss) + 1
    min_train_loss = min(train_total_loss)
    min_val_epoch = np.argmin(val_total_loss) + 1
    min_val_loss = min(val_total_loss)

    plt.annotate(f'Min Train Loss: {min_train_loss:.4f} (Epoch {min_train_epoch})',
                 xy=(min_train_epoch, min_train_loss), xytext=(5, 5),
                 textcoords='offset points', ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.annotate(f'Min Val Loss: {min_val_loss:.4f} (Epoch {min_val_epoch})',
                 xy=(min_val_epoch, min_val_loss), xytext=(5, 5),
                 textcoords='offset points', ha='left', va='bottom',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    plt.tight_layout()
    plt.show()

    print(f"Training Loss - Minimum: {min_train_loss:.4f} at Epoch {min_train_epoch}")
    print(f"Validation Loss - Minimum: {min_val_loss:.4f} at Epoch {min_val_epoch}")


# Evaluation Metrics (COCO Metrics)
def evaluation_metrics():

    # Carica le metriche
    with open('path_to_your_file.pkl', 'rb') as f:
        coco_metrics = pickle.load(f)

    # Stampa le metriche
    print(coco_metrics)

    # Supponiamo che tu abbia caricato le metriche come un dizionario
    metrics = {
        'AP': 0.573,
        'AP50': 0.869,
        'AP75': 0.637,
        'APs': -1.0,
        'APm': 0.205,
        'APl': 0.577,
        'ARmax1': 0.626,
        'ARmax10': 0.659,
        'ARmax100': 0.659,
        'ARs': -1.0,
        'ARm': 0.212,
        'ARl': 0.663
    }

    # Rimuovi le metriche con valori non validi
    metrics = {k: v for k, v in metrics.items() if v >= 0}

    # Grafico a barre
    plt.figure(figsize=(12, 6))
    plt.bar(metrics.keys(), metrics.values(), color='skyblue')
    plt.xlabel('Metrics')
    plt.ylabel('Value')
    plt.title('COCO Evaluation Metrics')
    plt.xticks(rotation=45)
    plt.show()


# Comparison of Training and Validation Losses
def compare_losses():

    # Script simile al precedente ma con tutte le componenti della loss

    train_cls_loss = [0.2498, 0.2457, 0.2471, 0.2424, 0.2395, 0.2386, 0.2393, 0.2412]
    val_cls_loss = [0.3195, 0.3185, 0.3176, 0.3167, 0.3164, 0.3163, 0.3153, 0.3147]

    # Tracciare le componenti della loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_cls_loss, label='Training Classification Loss')
    plt.plot(epochs, val_cls_loss, label='Validation Classification Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Classification Loss')
    plt.title('Classification Loss Analysis')
    plt.legend()
    plt.grid(True)
    plt.show()


# Detailed Class-wise Performance
def detailed_performance():

    # Esempio ipotetico di metriche per classe
    classes = ['Class A', 'Class B', 'Class C']
    AP_class = [0.55, 0.65, 0.75]

    # Grafico a barre per le metriche per classe
    plt.figure(figsize=(10, 6))
    plt.bar(classes, AP_class, color='lightgreen')
    plt.xlabel('Class')
    plt.ylabel('AP')
    plt.title('AP per Class')
    plt.show()


# IoU Distribution (Distribuzione di IoU)
def iou_distribution():
    import numpy as np

    # Esempio ipotetico di valori IoU
    iou_values = np.random.uniform(0.5, 1.0, 1000)  # Sostituisci con i tuoi valori IoU

    # Istogramma della distribuzione di IoU
    plt.figure(figsize=(10, 6))
    plt.hist(iou_values, bins=20, color='purple', alpha=0.7)
    plt.xlabel('IoU')
    plt.ylabel('Frequency')
    plt.title('IoU Distribution')
    plt.show()


# Supponendo di avere le immagini e le predizioni
def plot_image_with_boxes(image, boxes, labels, scores, color=(0, 255, 0)):

    # Disegna le bounding boxes
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f'{label}: {score:.2f}'
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostra l'immagine
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# Qualitative Analysis
def qualitative_analysis():

    # Carica un'immagine e le relative annotazioni
    #image = cv2.imread('path_to_image.jpg')
    #annotations = [(100, 100, 200, 200), (300, 300, 100, 150)]  # Esempio di bounding boxes

    # Disegna le annotazioni sull'immagine
    #for bbox in annotations:
    #    x, y, w, h = bbox
    #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Visualizza l'immagine con le annotazioni
    #plt.figure(figsize=(8, 8))
    #plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #plt.axis('off')
    #plt.show()


    # Esempio d'uso (sostituisci con le tue immagini e predizioni)
    image = cv2.imread('path_to_image.jpg')
    boxes = [[50, 50, 200, 200]]  # Esempio di box
    labels = ['Label1']
    scores = [0.9]
    plot_image_with_boxes(image, boxes, labels, scores)


def main():
    """

    :return:
    """

    file_path = 'losses.txt'
    loss_data = extract_losses(file_path)

    convergence_analysis(120, loss_data['total_loss'], loss_data['val_total_loss'], smoothing=False, log_scale=True)
    #evaluation_metrics()
    #compare_losses()
    #detailed_performance()
    #iou_distribution()
    #qualitative_analysis()


if __name__ == '__main__':
    main()
