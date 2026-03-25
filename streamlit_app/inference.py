import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import VALID_L0_MAPPING, VALID_L1_MAPPING, VALID_L2_MAPPING, REVERSE_L0_MAP, REVERSE_L1_MAP, REVERSE_L2_MAP

# Ensuring inputs are shaped perfectly for the ViT backbone
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_strict_hierarchy(model, images, map_l0, map_l1, map_l2):
    """Predicts the hierarchy and returns both classes and probabilities."""
    model.eval()
    
    with torch.no_grad():
        logits = model(images)
        batch_size = images.shape[0]
        
        # Calculating probabilities from logits using Softmax
        probs_l0 = F.softmax(logits['L0'], dim=1)
        probs_l1 = F.softmax(logits['L1'], dim=1)
        probs_l2 = F.softmax(logits['L2'], dim=1)
        
        pred_l0_list, prob_l0_list = [], []
        pred_l1_list, prob_l1_list = [], []
        pred_l2_list, prob_l2_list = [], []
        
        for i in range(batch_size):
            # Level 0 Prediction
            l0_logits_clone = logits['L0'][i].clone()
            parent_l0 = "ROOT"
            
            if parent_l0 in map_l0:
                valid_l0_classes = map_l0[parent_l0]
                for class_idx in range(len(l0_logits_clone)):
                    if class_idx not in valid_l0_classes:
                        l0_logits_clone[class_idx] = float('-inf')
                l0_pred = torch.argmax(l0_logits_clone).item()
                l0_prob = probs_l0[i][l0_pred].item() 
            else:
                l0_pred = -1
                l0_prob = 1.0 
                
            pred_l0_list.append(l0_pred)
            prob_l0_list.append(l0_prob)

            # Level 1 Prediction
            l1_logits_clone = logits['L1'][i].clone()
            parent_l1 = l0_pred
            
            if parent_l1 in map_l1:
                valid_l1_classes = map_l1[parent_l1]
                for class_idx in range(len(l1_logits_clone)):
                    if class_idx not in valid_l1_classes:
                        l1_logits_clone[class_idx] = float('-inf')
                l1_pred = torch.argmax(l1_logits_clone).item()
                l1_prob = probs_l1[i][l1_pred].item()
            else:
                l1_pred = -1
                l1_prob = 1.0 
                
            pred_l1_list.append(l1_pred)
            prob_l1_list.append(l1_prob)

            # Level 2 Prediction
            l2_logits_clone = logits['L2'][i].clone()
            parent_l2 = l1_pred
            
            if parent_l2 in map_l2:
                valid_l2_classes = map_l2[parent_l2]
                for class_idx in range(len(l2_logits_clone)):
                    if class_idx not in valid_l2_classes:
                        l2_logits_clone[class_idx] = float('-inf')
                l2_pred = torch.argmax(l2_logits_clone).item()
                l2_prob = probs_l2[i][l2_pred].item()
            else:
                l2_pred = -1
                l2_prob = 1.0 
                
            pred_l2_list.append(l2_pred)
            prob_l2_list.append(l2_prob)

        # Converting to tensors
        device = images.device
        pred_l0_tensor = torch.tensor(pred_l0_list, device=device)
        prob_l0_tensor = torch.tensor(prob_l0_list, device=device)
        
        pred_l1_tensor = torch.tensor(pred_l1_list, device=device)
        prob_l1_tensor = torch.tensor(prob_l1_list, device=device)
        
        pred_l2_tensor = torch.tensor(pred_l2_list, device=device)
        prob_l2_tensor = torch.tensor(prob_l2_list, device=device)
        
        return (pred_l0_tensor, prob_l0_tensor), (pred_l1_tensor, prob_l1_tensor), (pred_l2_tensor, prob_l2_tensor)

def infer_single_image(image, model, device):
    """Processes a PIL Image from Streamlit and returns formatted predictions."""
    # Ensuring image has 3 RGB channels
    image = image.convert('RGB')
    
    # Applying transform and adding the missing batch dimension
    input_tensor = inference_transform(image).unsqueeze(0).to(device)
    
    # Running the strict pipeline
    (p_l0, prob_l0), (p_l1, prob_l1), (p_l2, prob_l2) = predict_strict_hierarchy(
        model, input_tensor, VALID_L0_MAPPING, VALID_L1_MAPPING, VALID_L2_MAPPING
    )
    
    # Extracting data and map back to strings
    results = {
        'Level 0': {'Class': REVERSE_L0_MAP.get(p_l0.item(), "Unknown"), 'Confidence': prob_l0.item()},
        'Level 1': {'Class': REVERSE_L1_MAP.get(p_l1.item(), "Unknown"), 'Confidence': prob_l1.item()},
        'Level 2': {'Class': REVERSE_L2_MAP.get(p_l2.item(), "Unknown"), 'Confidence': prob_l2.item()}
    }
    
    return results