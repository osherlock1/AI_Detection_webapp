import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from transformers import CLIPModel, CLIPProcessor 
from peft import PeftModel
import matplotlib.pyplot as plt 
import scipy.ndimage 

# --- Configuration ---
MODEL_ID = "openai/clip-vit-base-patch32"
PEFT_MODEL_PATH = 'final_model/'


GITHUB_EXAMPLES_URL = "https://drive.google.com/drive/u/2/folders/1oj2tbDBZ28SjJK1IZvAygkaOkw7O2yvl" 

AI_IMAGE_ARTIFACTS = [
    "a photo with distorted fingers or too many fingers",
    "the photo is unnatrualy blurry",
    "the photo has unatrual or nonsensical text",
    "a photo with unnaturally smooth textures",
    "a photo with strange repeating patterns or textures",
    "a photo with objects blending together incorrectly",
    "a photo with distorted or warped background elements",
    "uncanny valley look",
    "weird lighting or shadows",
    "inconsistent art style"
]
REAL_IMAGE_CHARACTERISTICS = [
    "a clear, sharp photograph",
    "natural lighting and shadows",
    "detailed skin texture with imperfections",
    "realistic background details",
    "consistent focus and depth of field",
    "normal-looking hands and eyes",
    "a typical photograph"
]


# --- Model Class ---
class CLIPBinaryClassifier(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_id) 
        self.classifier_head = nn.Linear(self.clip.config.vision_config.hidden_size, 1)

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, output_attentions=None, return_dict=None):
        if pixel_values is not None:
            vision_outputs = self.clip.vision_model(
                pixel_values=pixel_values,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            image_features = vision_outputs.pooler_output 
            logits = self.classifier_head(image_features)
            return {"logits": logits, "vision_outputs": vision_outputs}
        
        else:
             raise ValueError("Pixel values must be provided for binary classification")

# --- Load Model and Processor ---
@st.cache_resource
def load_resources():
    base_model_with_head = CLIPBinaryClassifier(MODEL_ID)
    model = PeftModel.from_pretrained(base_model_with_head, PEFT_MODEL_PATH)
    model.eval()

    processor = CLIPProcessor.from_pretrained(MODEL_ID) 
    return model, processor

model, processor = load_resources()

# --- Attention visualization ---
def visualize_attention(image, attentions, processor, layer_index=-1, head_index=None):
    try:
        attention_map = attentions[layer_index] 
        cls_token_attention = attention_map[0, :, 0, 1:] 
        num_patches = cls_token_attention.shape[-1]
        grid_size = int(np.sqrt(num_patches))

        if grid_size * grid_size != num_patches:
             st.warning(f"Cannot form square grid for {num_patches} patches. Skipping attention map.")
             return None



        if head_index is None:
            attention_to_plot = cls_token_attention.mean(dim=0) 

        else:
            if head_index >= cls_token_attention.shape[0]:
                 st.warning(f"Head index {head_index} out of range. Averaging heads.")
                 attention_to_plot = cls_token_attention.mean(dim=0)

            else:
                 attention_to_plot = cls_token_attention[head_index] 
        

        attention_grid = attention_to_plot.reshape(grid_size, grid_size).cpu().numpy()
        img_size = image.size
        scale_factors = (img_size[1] / grid_size, img_size[0] / grid_size)
        resized_attention_map = scipy.ndimage.zoom(attention_grid, scale_factors, order=1) 

        fig, ax = plt.subplots()
        ax.imshow(image)
        im = ax.imshow(resized_attention_map, cmap='viridis', alpha=0.6) 
        ax.axis('off')
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error generating attention map: {e}")


# --- Streamlit App ---
APP_SCRIPT_NAME = "streamlit_app.py" 

st.title("🖼️ AI vs Real Image Detector")

st.markdown(
    "Upload an image to predict whether it's AI-generated or real, "
    "get reasoning based on CLIP characteristics, and visualize attention."
)

# --- Link to Example Images 
st.markdown(f"""
You can find some AI-generated and Real images to test in this
[Google Drive Folder]({GITHUB_EXAMPLES_URL}).
""")
st.write("---")


uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB") 
    except Exception as e:
        st.error(f"Error opening image: {e}")
        st.stop()



    st.image(image, caption='Uploaded Image', use_column_width=True)

    with st.spinner('🔍 Analyzing Image...'):
        
        # --- Binary Classification ---
        inputs_for_classifier = processor(images=image, return_tensors="pt")
        
        is_ai_generated = False 
        prediction_text = "**Error during classification**" 
        classification_successful = False 
        
        try:
            with torch.no_grad():
                outputs = model(pixel_values=inputs_for_classifier['pixel_values'], output_attentions=True) 
                logits = outputs["logits"]
                prob = torch.sigmoid(logits)
                attentions = outputs.get("vision_outputs", {}).get("attentions")

            st.subheader("Binary Classification Result")
            if prob.item() > 0.5:
                is_ai_generated = True
                prediction_text = "**AI-Generated Image**"
            else:
                is_ai_generated = False
                prediction_text = "**Real Photograph**"
            st.success(f"✅ Prediction: {prediction_text} (Chance the Image is AI Generated: {prob.item() * 100 :.2f})%")
            classification_successful = True 
            
        except Exception as e:
             st.error(f"Error during binary classification: {e}")
             attentions = None 



        # ---  Attention Map Visualization ---
        if classification_successful and attentions: 
            st.subheader("Attention Map Visualization")
            st.markdown("Visualizing attention from the CLS token to image patches in the last layer (averaged across heads). Brighter areas indicate higher attention.")
            visualize_attention(image, attentions, processor)








        # --- Characteristic Check using CLIP Similarity ---
        if classification_successful: 
           
            if is_ai_generated:
                descriptions_to_use = AI_IMAGE_ARTIFACTS
                comparison_label = "Common AI Artifacts / Styles"
                reasoning_intro = "The model leans towards **AI-Generated** possibly because the image is most similar to these artifact descriptions:"
            else:
                descriptions_to_use = REAL_IMAGE_CHARACTERISTICS
                comparison_label = "Common Real Image Characteristics"
                reasoning_intro = "The model leans towards **Real Photograph** possibly because the image is most similar to these characteristics:"
                






            # --- MODIFIED SECTION ---
            st.subheader(f"Reasoning based on CLIP Comparison")
            st.markdown(f"(Comparing image against *{comparison_label}*)")
            
            try:
                 with torch.no_grad():
                  
                    inputs_for_similarity = processor(text=descriptions_to_use, images=image, return_tensors="pt", padding=True)
                    
                    outputs_similarity = model.clip(**inputs_for_similarity)
                    logits_per_image = outputs_similarity.logits_per_image 
                    probs_per_image = logits_per_image.softmax(dim=1) 
                    scores = probs_per_image.squeeze().cpu().numpy()
                    
                    scores_dict = {desc: score for desc, score in zip(descriptions_to_use, scores)}
                    sorted_scores = sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)
                    
                   
                    if len(sorted_scores) >= 1:
                         top1_desc = sorted_scores[0][0]
                         top1_score = sorted_scores[0][1]
                         reasoning_text = f"1. \"{top1_desc}\" (Score: {top1_score:.3f})"
                         if len(sorted_scores) >= 2:
                             top2_desc = sorted_scores[1][0]
                             top2_score = sorted_scores[1][1]
                             reasoning_text += f"\n2. \"{top2_desc}\" (Score: {top2_score:.3f})"
                         
                         st.info(f"{reasoning_intro}\n{reasoning_text}")
                    else:
                         st.warning("Could not calculate CLIP similarity scores.")

            except Exception as e:
                 st.error(f"Error during characteristic check: {e}")
            # --- END OF MODIFIED SECTION ---

else:
    st.info("Please upload an image to classify.")