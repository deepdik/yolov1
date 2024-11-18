import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as T
import json
import os
from PIL import ImageDraw
import random
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Set constants for training and model configuration
DATA_PATH = 'data'          # Path to the dataset
BATCH_SIZE = 64             # Number of samples per batch
EPOCHS = 135                # Number of epochs to train
WARMUP_EPOCHS = 0           # Number of warmup epochs (not used here)
LEARNING_RATE = 1E-4        # Learning rate for the optimizer

EPSILON = 1E-6              # Small value to prevent division by zero
IMAGE_SIZE = (448, 448)     # Size to which input images will be resized

S = 7       # Number of grid cells along one dimension (SxS grid)
B = 2       # Number of bounding boxes predicted per grid cell
C = 20      # Number of classes in the dataset

# Define the class mapping for Pascal VOC
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
# Create mappings from class names to indices and vice versa
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(VOC_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for idx, cls_name in enumerate(VOC_CLASSES)}

def get_iou(p, a):
    """
    Calculate the Intersection over Union (IoU) between predicted and actual bounding boxes.
    Args:
        p: Predicted bounding boxes tensor of shape (batch, S, S, B*5+C)
        a: Actual bounding boxes tensor of shape (batch, S, S, B*5+C)
    Returns:
        IoU tensor of shape (batch, S, S, B, B)
    """
    # Convert bounding boxes from center coordinates to corner coordinates
    p_tl, p_br = bbox_to_coords(p)          # Predicted top-left and bottom-right corners
    a_tl, a_br = bbox_to_coords(a)          # Actual top-left and bottom-right corners

    # Prepare tensors for broadcasting
    coords_join_size = (-1, -1, -1, B, B, 2)

    # Calculate the intersection coordinates
    tl = torch.max(
        p_tl.unsqueeze(4).expand(coords_join_size),   # Expand predicted corners for broadcasting
        a_tl.unsqueeze(3).expand(coords_join_size)    # Expand actual corners for broadcasting
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size)
    )

    # Calculate intersection area
    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = intersection_sides[..., 0] * intersection_sides[..., 1]   # Intersection area

    # Calculate areas of predicted and actual boxes
    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)      # Predicted box areas
    p_area = p_area.unsqueeze(4).expand_as(intersection)
    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)      # Actual box areas
    a_area = a_area.unsqueeze(3).expand_as(intersection)

    # Calculate union area
    union = p_area + a_area - intersection

    # Handle division by zero
    zero_unions = (union == 0.0)
    union[zero_unions] = EPSILON
    intersection[zero_unions] = 0.0

    # Compute IoU
    return intersection / union

def bbox_to_coords(t):
    """
    Convert bounding boxes from center coordinates to corner coordinates.
    Args:
        t: Tensor containing bounding boxes in [x, y, width, height] format
    Returns:
        Tuple of tensors: (top-left corners, bottom-right corners)
    """
    # Extract width and height
    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    # Calculate x1 and x2
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    # Extract height and y-coordinate
    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    # Calculate y1 and y2
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)

def scheduler_lambda(epoch):
    """
    Learning rate scheduler function.
    Args:
        epoch: Current epoch number
    Returns:
        Scaling factor for the learning rate
    """
    if epoch < WARMUP_EPOCHS + 75:
        return 1
    elif epoch < WARMUP_EPOCHS + 105:
        return 0.1
    else:
        return 0.01

def bbox_attr(data, i):
    """
    Returns the i-th attribute of each bounding box in data.
    Args:
        data: Tensor containing bounding box data
        i: Index of the attribute to extract (0: x, 1: y, 2: width, 3: height, 4: confidence)
    Returns:
        Tensor containing the specified attribute
    """
    attr_start = C + i
    return data[..., attr_start::5]

def scale_bbox_coord(coord, center, scale):
    """
    Scale a coordinate around a center point.
    Args:
        coord: Original coordinate
        center: Center point to scale around
        scale: Scaling factor
    Returns:
        Scaled coordinate
    """
    return ((coord - center) * scale) + center

def get_overlap(a, b):
    """
    Returns the proportion of overlap between two boxes.
    Args:
        a: First bounding box (tl, width, height, confidence, class)
        b: Second bounding box (tl, width, height, confidence, class)
    Returns:
        Overlap proportion (float)
    """
    a_tl, a_width, a_height, _, _ = a
    b_tl, b_width, b_height, _, _ = b

    # Calculate intersection coordinates
    i_tl = (
        max(a_tl[0], b_tl[0]),
        max(a_tl[1], b_tl[1])
    )
    i_br = (
        min(a_tl[0] + a_width, b_tl[0] + b_width),
        min(a_tl[1] + a_height, b_tl[1] + b_height),
    )

    # Calculate intersection area
    intersection = max(0, i_br[0] - i_tl[0]) * max(0, i_br[1] - i_tl[1])

    # Calculate areas
    a_area = a_width * a_height
    b_area = b_width * b_height

    # Handle division by zero
    a_intersection = intersection if a_area > 0 else 0
    b_intersection = intersection if b_area > 0 else 0
    a_area = a_area if a_area > 0 else EPSILON
    b_area = b_area if b_area > 0 else EPSILON

    # Return maximum overlap proportion
    return max(a_intersection / a_area, b_intersection / b_area)

def plot_boxes(data, labels, classes, color='orange', min_confidence=0.2, max_overlap=0.5, file=None):
    """
    Plots bounding boxes on the given image.
    Args:
        data: Original image tensor
        labels: Predicted labels tensor
        classes: List of class names
        color: Color of bounding boxes
        min_confidence: Minimum confidence threshold for displaying boxes
        max_overlap: Maximum allowed overlap for non-maximum suppression
        file: File path to save the image (if None, displays the image)
    """
    # Calculate grid cell sizes
    grid_size_x = data.size(dim=2) / S
    grid_size_y = data.size(dim=1) / S
    m = labels.size(dim=0)  # Rows
    n = labels.size(dim=1)  # Columns

    bboxes = []
    for i in range(m):
        for j in range(n):
            # Apply softmax to class probabilities
            class_probs = F.softmax(labels[i, j, :C], dim=0)
            class_index = torch.argmax(class_probs).item()
            for k in range((labels.size(dim=2) - C) // 5):
                bbox_start = 5 * k + C
                bbox_end = 5 * (k + 1) + C
                bbox = labels[i, j, bbox_start:bbox_end]
                confidence = class_probs[class_index].item() * bbox[4].item()  # pr(c) * confidence
                if confidence > min_confidence:
                    # Calculate bounding box dimensions
                    width = bbox[2] * IMAGE_SIZE[0]
                    height = bbox[3] * IMAGE_SIZE[1]
                    tl = (
                        bbox[0] * IMAGE_SIZE[0] + j * grid_size_x - width / 2,
                        bbox[1] * IMAGE_SIZE[1] + i * grid_size_y - height / 2
                    )
                    # Clamp coordinates to ensure they are within image bounds
                    tl = (max(0, tl[0]), max(0, tl[1]))
                    br = (min(IMAGE_SIZE[0], tl[0] + width), min(IMAGE_SIZE[1], tl[1] + height))

                    # Only add the bounding box if the coordinates are valid
                    if tl[0] < br[0] and tl[1] < br[1]:
                        bboxes.append([tl, width, height, confidence, class_index])

    # Sort bounding boxes by confidence score in descending order
    bboxes = sorted(bboxes, key=lambda x: x[3], reverse=True)

    # Calculate IoUs between each pair of boxes for non-maximum suppression
    num_boxes = len(bboxes)
    iou = [[0 for _ in range(num_boxes)] for _ in range(num_boxes)]
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou[i][j] = get_overlap(bboxes[i], bboxes[j])

    # Non-maximum suppression and render image
    image = T.ToPILImage()(data)  # Convert tensor to PIL image
    draw = ImageDraw.Draw(image)
    discarded = set()
    for i in range(num_boxes):
        if i not in discarded:
            tl, width, height, confidence, class_index = bboxes[i]

            # Suppress overlapping boxes of the same class
            for j in range(num_boxes):
                other_class = bboxes[j][4]
                if j != i and other_class == class_index and iou[i][j] > max_overlap:
                    discarded.add(j)

            # Annotate image with bounding box and class label
            br = (tl[0] + width, tl[1] + height)
            draw.rectangle((tl, br), outline=color)
            text_pos = (max(0, tl[0]), max(0, tl[1] - 11))
            text = f'{classes[class_index]} {round(confidence * 100, 1)}%'
            text_bbox = draw.textbbox(text_pos, text)
            draw.rectangle(text_bbox, fill=color)
            draw.text(text_pos, text)
    if file is None:
        image.show()
    else:
        # Save the annotated image to the specified file
        output_dir = os.path.dirname(file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not file.endswith('.png'):
            file += '.png'
        image.save(file)

class YoloPascalVocDataset(Dataset):
    """
    Custom Dataset class for loading Pascal VOC data suitable for YOLOv1 training.
    """
    def __init__(self, set_type, normalize=False, augment=False):
        """
        Initialize the dataset.
        Args:
            set_type: 'train' or 'test' to specify the dataset split
            normalize: Whether to normalize the images
            augment: Whether to apply data augmentation
        """
        assert set_type in {'train', 'test'}
        self.dataset = VOCDetection(
            root=DATA_PATH,
            year='2007',
            image_set=('train' if set_type == 'train' else 'val'),
            download=True,
            transform=T.Compose([
                T.ToTensor(),
                T.Resize(IMAGE_SIZE)
            ])
        )
        self.normalize = normalize
        self.augment = augment
        self.classes = CLASS_TO_IDX  # Use the predefined class mapping

    def __getitem__(self, i):
        """
        Get a single data sample.
        Args:
            i: Index of the sample
        Returns:
            Tuple of (image tensor, ground truth tensor, original image tensor)
        """
        data, label = self.dataset[i]
        original_data = data.clone()  # Clone the original image

        # Random shifts and scaling for data augmentation
        x_shift = int((0.2 * random.random() - 0.1) * IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Calculate grid cell sizes
        grid_size_x = data.size(dim=2) / S  # Width of each grid cell
        grid_size_y = data.size(dim=1) / S  # Height of each grid cell

        # Initialize ground truth tensor
        boxes = {}
        class_names = {}                    # Track which class is assigned to each grid cell
        depth = 5 * B + C                   # Total depth per grid cell
        ground_truth = torch.zeros((S, S, depth))

        # Extract bounding boxes from label
        width, height = int(label['annotation']['size']['width']), int(label['annotation']['size']['height'])
        x_scale = IMAGE_SIZE[0] / width
        y_scale = IMAGE_SIZE[1] / height
        objects = label['annotation']['object']
        if not isinstance(objects, list):
            objects = [objects]  # Ensure objects is a list

        for obj in objects:
            name = obj['name']
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            box = obj['bndbox']
            # Scale bounding box coordinates to match IMAGE_SIZE
            coords = (
                int(int(box['xmin']) * x_scale),
                int(int(box['xmax']) * x_scale),
                int(int(box['ymin']) * y_scale),
                int(int(box['ymax']) * y_scale)
            )
            x_min, x_max, y_min, y_max = coords

            # Augment labels if augmentation is enabled
            if self.augment:
                half_width = IMAGE_SIZE[0] / 2
                half_height = IMAGE_SIZE[1] / 2
                x_min = scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)  # Grid cell column
            row = int(mid_y // grid_size_y)  # Grid cell row

            if 0 <= col < S and 0 <= row < S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / IMAGE_SIZE[0],     # X coord relative to grid cell
                            (mid_y - row * grid_size_y) / IMAGE_SIZE[1],     # Y coord relative to grid cell
                            (x_max - x_min) / IMAGE_SIZE[0],                 # Width normalized
                            (y_max - y_min) / IMAGE_SIZE[1],                 # Height normalized
                            1.0                                              # Confidence
                        )

                        # Fill bbox slots with current bbox
                        bbox_start = 5 * bbox_index + C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(B - bbox_index)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        """
        Returns the size of the dataset.
        """
        return len(self.dataset)

class YOLOv1ResNet(nn.Module):
    """
    YOLOv1 model using ResNet-50 as the backbone.
    """
    def __init__(self):
        super().__init__()
        self.depth = B * 5 + C  # Output depth per grid cell

        # Load pre-trained ResNet-50 model
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.requires_grad_(False)            # Freeze backbone weights

        # Replace the last two layers with identity to remove them
        backbone.avgpool = nn.Identity()
        backbone.fc = nn.Identity()

        # Build the detection layers on top of the backbone
        self.model = nn.Sequential(
            backbone,
            Reshape(2048, 14, 14),
            DetectionNet(2048)              # Detection layers
        )

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input tensor
        Returns:
            Output tensor reshaped to (batch_size, S, S, depth)
        """
        return self.model.forward(x)

class DetectionNet(nn.Module):
    """
    The detection layers added on top of the backbone network.
    """
    def __init__(self, in_channels):
        super().__init__()
        inner_channels = 1024
        self.depth = B * 5 + C
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, stride=2, padding=1),   # Downsample to (7, 7)
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Flatten(),

            nn.Linear(7 * 7 * inner_channels, 4096),
            nn.LeakyReLU(negative_slope=0.1),

            nn.Linear(4096, S * S * self.depth)
        )

    def forward(self, x):
        """
        Forward pass through the detection network.
        Args:
            x: Input tensor
        Returns:
            Output tensor reshaped to (batch_size, S, S, depth)
        """
        return torch.reshape(
            self.model.forward(x),
            (-1, S, S, self.depth)
        )

class Reshape(nn.Module):
    """
    Module to reshape the tensor.
    """
    def __init__(self, *args):
        super().__init__()
        self.shape = tuple(args)

    def forward(self, x):
        """
        Reshape the input tensor.
        Args:
            x: Input tensor
        Returns:
            Reshaped tensor
        """
        return torch.reshape(x, (-1, *self.shape))

class SumSquaredErrorLoss(nn.Module):
    """
    Custom loss function for YOLOv1.
    """
    def __init__(self):
        super().__init__()
        self.l_coord = 5    # Weight for coordinate loss
        self.l_noobj = 0.5  # Weight for no-object confidence loss

    def forward(self, p, a):
        """
        Compute the loss between predictions and ground truth.
        Args:
            p: Predicted tensor
            a: Actual ground truth tensor
        Returns:
            Computed loss value
        """
        # Calculate IoU between predicted and actual bounding boxes
        iou = get_iou(p, a)                     # (batch, S, S, B, B)
        max_iou, _ = torch.max(iou, dim=-1)     # (batch, S, S, B)

        # Get masks for loss calculation
        bbox_mask = bbox_attr(a, 4) > 0.0                   # Mask where objects exist in ground truth
        p_template = bbox_attr(p, 4) > 0.0
        obj_i = bbox_mask[..., 0:1]                         # 1 if any object exists in the grid cell
        # Determine responsible bounding box for each cell
        responsible = torch.zeros_like(p_template).scatter_(
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),    # Index of the bbox with maximum IoU
            value=1
        )
        obj_ij = obj_i * responsible                        # Mask for responsible bounding boxes
        noobj_ij = ~obj_ij.bool()                           # Mask for non-responsible bounding boxes

        # Calculate losses for bounding box coordinates (x, y)
        x_losses = mse_loss(
            obj_ij * bbox_attr(p, 0),
            obj_ij * bbox_attr(a, 0)
        )
        y_losses = mse_loss(
            obj_ij * bbox_attr(p, 1),
            obj_ij * bbox_attr(a, 1)
        )
        pos_losses = x_losses + y_losses

        # Calculate losses for bounding box dimensions (width, height)
        p_width = bbox_attr(p, 2)
        a_width = bbox_attr(a, 2)
        width_losses = mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + EPSILON),
            obj_ij * torch.sqrt(a_width)
        )
        p_height = bbox_attr(p, 3)
        a_height = bbox_attr(a, 3)
        height_losses = mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + EPSILON),
            obj_ij * torch.sqrt(a_height)
        )
        dim_losses = width_losses + height_losses

        # Calculate confidence losses
        obj_confidence_losses = mse_loss(
            obj_ij * bbox_attr(p, 4),
            obj_ij * max_iou
        )
        noobj_confidence_losses = mse_loss(
            noobj_ij * bbox_attr(p, 4),
            noobj_ij * torch.zeros_like(max_iou)
        )

        # Calculate classification losses using Cross-Entropy Loss
        class_mask = obj_i.squeeze(-1).bool()               # Mask for cells with objects
        class_targets = torch.argmax(a[..., :C], dim=-1)[class_mask]
        class_predictions = p[..., :C][class_mask]
        class_losses = F.cross_entropy(class_predictions, class_targets, reduction='sum')

        # Total loss
        total = self.l_coord * (pos_losses + dim_losses) \
                + obj_confidence_losses \
                + self.l_noobj * noobj_confidence_losses \
                + class_losses
        return total / BATCH_SIZE

def mse_loss(a, b):
    """
    Mean Squared Error loss helper function.
    Args:
        a: Predicted values
        b: Target values
    Returns:
        Computed MSE loss
    """
    flattened_a = torch.flatten(a, end_dim=-2)
    flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(
        flattened_a,
        flattened_b,
        reduction='sum'
    )

############ Train and Save ####################
def train_and_save_model():
    """
    Function to train the model and save the trained weights and metrics.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.autograd.set_detect_anomaly(True)         # Enable anomaly detection for debugging
    writer = SummaryWriter()                        # For TensorBoard visualization
    now = datetime.now()

    # Initialize the model and move it to the device (GPU or CPU)
    model = YOLOv1ResNet().to(device)
    loss_function = SumSquaredErrorLoss()

    # Use Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=scheduler_lambda
    )

    # Load the datasets
    train_set = YoloPascalVocDataset('train', normalize=True, augment=True)
    test_set = YoloPascalVocDataset('test', normalize=True, augment=True)

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        num_workers=5,
        persistent_workers=True,
        drop_last=True,
        shuffle=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        num_workers=5,
        persistent_workers=True,
        drop_last=True
    )

    # Create directories for saving models and metrics
    root = os.path.join(
        'models',
        'yolo_v1',
        now.strftime('%m_%d_%Y'),
        now.strftime('%H_%M_%S')
    )
    weight_dir = os.path.join(root, 'weights')
    if not os.path.isdir(weight_dir):
        os.makedirs(weight_dir)

    # Initialize metrics arrays
    train_losses = np.empty((2, 0))
    test_losses = np.empty((2, 0))

    def save_metrics():
        """
        Save the training and testing losses to .npy files.
        """
        np.save(os.path.join(root, 'train_losses.npy'), train_losses)
        np.save(os.path.join(root, 'test_losses.npy'), test_losses)

    #### Training Loop ####
    for epoch in tqdm(range(WARMUP_EPOCHS + EPOCHS), desc='Epoch'):
        model.train()
        train_loss = 0
        for data, labels, _ in tqdm(train_loader, desc='Train', leave=False):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model.forward(data)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / len(train_loader)
            del data, labels

        # Update learning rate scheduler
        writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)
        scheduler.step()

        # Record training loss
        train_losses = np.append(train_losses, [[epoch], [train_loss]], axis=1)
        writer.add_scalar('Loss/train', train_loss, epoch)

        # Evaluate on the test set every 4 epochs
        if epoch % 4 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = 0
                for data, labels, _ in tqdm(test_loader, desc='Test', leave=False):
                    data = data.to(device)
                    labels = labels.to(device)

                    predictions = model.forward(data)
                    loss = loss_function(predictions, labels)

                    test_loss += loss.item() / len(test_loader)
                    del data, labels
            # Record testing loss
            test_losses = np.append(test_losses, [[epoch], [test_loss]], axis=1)
            writer.add_scalar('Loss/test', test_loss, epoch)
            save_metrics()
    save_metrics()
    # Save the trained model weights
    torch.save(model.state_dict(), os.path.join(weight_dir, 'final'))

############ Plot test images ####################
def plot_test_images():
    """
    Function to load the trained model and plot predictions on test images.
    """
    # Load class names
    classes = VOC_CLASSES

    # Load the dataset
    dataset = YoloPascalVocDataset('test', normalize=True, augment=False)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize the model and move it to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLOv1ResNet().to(device)
    model.eval()
    # Load the saved model weights
    model.load_state_dict(torch.load(os.path.join('models', 'yolo_v1', 'final'), map_location=device))

    count = 0
    with torch.no_grad():
        for image, labels, original in tqdm(loader):
            # Move images to the device
            image = image.to(device)
            # Get predictions
            predictions = model.forward(image)
            predictions = predictions.cpu()  # Move predictions back to CPU for plotting

            for i in range(image.size(dim=0)):
                # Plot and save the image with bounding boxes
                plot_boxes(
                    original[i, :, :, :],  # Original image
                    predictions[i, :, :, :],  # Predicted labels
                    classes,
                    file=os.path.join('results', f'{count}')
                )
                count += 1

############# Plot Loss Graphs ##############
def plot_loss_graphs():
    """
    Function to load loss data and plot training and testing loss curves.
    """
    # Load the loss data
    train_losses = np.load('train_losses.npy')
    test_losses = np.load('test_losses.npy')

    # Extract epochs and loss values
    train_epochs = train_losses[0]
    train_loss_values = train_losses[1]

    test_epochs = test_losses[0]
    test_loss_values = test_losses[1]

    # Plot the loss curves
    plt.figure(figsize=(10, 6))

    plt.plot(train_epochs, train_loss_values, label='Training Loss', color='blue', linewidth=2)
    plt.plot(test_epochs, test_loss_values, label='Testing Loss', color='red', linewidth=2)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

# Entry point of the script
if __name__ == '__main__':
    # Uncomment the function you want to run
    # train_and_save_model()    # Train the model
    # plot_test_images()        # Plot predictions on test images
    # plot_loss_graphs()        # Plot loss curves
    pass
