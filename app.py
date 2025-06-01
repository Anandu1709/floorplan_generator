from flask import Flask, render_template, request, url_for, redirect, flash, send_from_directory
import pandas as pd
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline
import os
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import io
import base64
import uuid
import time

app = Flask(__name__)
app.secret_key = "floor_plan_generator_secret_key"

# Configure upload and output folders
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/generated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Floor Plan Generator class (from your code)
class FloorPlanGenerator:
    def __init__(self, csv_path, img_dir):
        """Initialize the floor plan generator with dataset paths"""
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.df = None
        self.knn_model = None
        self.feature_means = None
        self.feature_stds = None
        self.pipeline = None
        
        # Prepare the data
        self._prepare_data()
        
    def _prepare_data(self):
        """Load and prepare data for matching"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.csv_path)
        
        # Extract features for matching
        features = self.df[['Beds', 'Baths', 'Square Feet']].values
        if 'Garages' in self.df.columns:
            features = np.column_stack([features, self.df['Garages'].values])
        else:
            # Add a column of zeros if garages not in dataset
            self.df['garages'] = 0
            features = np.column_stack([features, np.zeros(len(self.df))])
        
        # Normalize features
        self.feature_means = np.mean(features, axis=0)
        self.feature_stds = np.std(features, axis=0)
        normalized_features = (features - self.feature_means) / self.feature_stds
        
        # Create KNN model
        print("Creating retrieval model...")
        self.knn_model = NearestNeighbors(n_neighbors=5, algorithm='auto')
        self.knn_model.fit(normalized_features)
    
    def load_model(self):
        """Load the Stable Diffusion model with optimizations for low memory"""
        print("Loading Stable Diffusion model...")
        model_id = "runwayml/stable-diffusion-v1-5"
        
        # Use lower precision to save memory
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16
        )
        self.pipeline.enable_attention_slicing()  # Memory optimization
        
        if torch.cuda.is_available():
            print("Using GPU for inference")
            self.pipeline = self.pipeline.to("cuda")
        else:
            print("GPU not available, using CPU (this will be slow)")
        
    def find_similar_plans(self, user_input):
        """Find similar floor plans based on user requirements"""
        # Make sure user input has 4 elements [bedrooms, bathrooms, sq_feet, garages]
        if len(user_input) < 4:
            user_input = list(user_input) + [0] * (4 - len(user_input))
        
        # Normalize user input using the same parameters
        normalized_input = (np.array(user_input) - self.feature_means) / self.feature_stds
        
        # Find nearest neighbors
        distances, indices = self.knn_model.kneighbors([normalized_input])
        
        # Return the details of similar floor plans
        similar_plans = self.df.iloc[indices[0]]
        return similar_plans
    
    def generate_plan(self, user_requirements, save_path=None):
        """Generate a floor plan based on user requirements"""
        if self.pipeline is None:
            self.load_model()
        
        # Find similar plans for reference
        similar_plans = self.find_similar_plans(user_requirements)
        print(f"Found {len(similar_plans)} similar floor plans")
        
        # Create prompt from user requirements
        bedrooms, bathrooms, sq_feet, garages = user_requirements
        
        # IMPROVED PROMPT ENGINEERING FOR FLOOR PLANS
        prompt = f"top-down 2D architectural floor plan, blueprint style, {bedrooms} bedrooms, {bathrooms} bathrooms, {sq_feet} square feet"
        if garages > 0:
            prompt += f", {garages} car garage"
            
        # Add specific floor plan elements to the prompt
        prompt += ",  clearly showing room labels in proper english in calluna font, dimensions, walls, doors, windows, furniture layout, scale bar, north arrow, clean lines, professional architectural drawing, black and white technical drawing,consider you as civil engineer who is going to draw a floor plan for a customer,add floor plan emelemnts,make sure that generated image is clean and clearly visible,also ensure that generating image is almost similar to the reference images,the image should be draw in like a canvas "
        
        # Add negative prompt to avoid house exteriors
        negative_prompt = "3D, exterior view, perspective, isometric, house facade, colored, rendering, photograph, landscape, sky, trees, outdoor scene, people, color"
            
        print(f"Generating floor plan with prompt: '{prompt}'")
        
        # Generate image with the improved prompt
        with torch.inference_mode():
            image = self.pipeline(
                prompt, 
                negative_prompt=negative_prompt,
                guidance_scale=8,  # Increased guidance scale for more prompt adherence
                num_inference_steps=50  # More steps for better quality
            ).images[0]
        
        # Load reference images
        reference_images = []
        reference_paths = []
        for i, row in similar_plans.iterrows():
            img_path = os.path.join(self.img_dir, row['Image Path'])
            try:
                ref_img = Image.open(img_path).convert('RGB')
                reference_images.append(ref_img)
                reference_paths.append(img_path)
            except Exception as e:
                print(f"Could not load image: {img_path} - {e}")
        
        # Save results if path provided
        saved_paths = []
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            # Generate unique filename
            timestamp = int(time.time())
            gen_filename = f"generated_floor_plan_{timestamp}.png"
            gen_path = os.path.join(save_path, gen_filename)
            image.save(gen_path)
            saved_paths.append(gen_path)
            
            for i, img in enumerate(reference_images[:3]):  # Save up to 3 reference images
                ref_filename = f"reference_plan_{timestamp}_{i}.png"
                ref_path = os.path.join(save_path, ref_filename)
                img.save(ref_path)
                saved_paths.append(ref_path)
        
        return image, reference_images, similar_plans, saved_paths

# Initialize generator globally - but lazy load the model
generator = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate-page')
def generate_page():
    return render_template('generate.html')

@app.route('/generate', methods=['POST'])
def generate():
    global generator
    
    # Initialize generator if not already done
    if generator is None:
        csv_path = request.form.get('csv_path', 'house_plans_details.csv')
        img_dir = request.form.get('img_dir', 'images/images')
        
        # Check if files exist
        if not os.path.exists(csv_path):
            flash(f"CSV file not found: {csv_path}")
            return redirect(url_for('generate_page'))
        if not os.path.exists(img_dir):
            flash(f"Image directory not found: {img_dir}")
            return redirect(url_for('generate_page'))
            
        try:
            generator = FloorPlanGenerator(csv_path, img_dir)
            flash("Data loaded successfully")
        except Exception as e:
            flash(f"Error initializing generator: {str(e)}")
            return redirect(url_for('generate_page'))
    
    try:
        # Get user inputs from form
        bedrooms = int(request.form.get('bedrooms', 3))
        bathrooms = float(request.form.get('bathrooms', 2.0))
        sq_feet = float(request.form.get('sq_feet', 1500))
        garages = int(request.form.get('garages', 1))
        
        # Generate floor plan
        user_requirements = [bedrooms, bathrooms, sq_feet, garages]
        generated_image, reference_images, similar_plans, saved_paths = generator.generate_plan(
            user_requirements, save_path=app.config['OUTPUT_FOLDER']
        )
        
        # Extract file paths for template
        generated_path = os.path.basename(saved_paths[0]) if saved_paths else ""
        
        # Return the results page
        return render_template(
            'results.html',
            generated_path=generated_path,
            requirements={
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sq_feet': sq_feet,
                'garages': garages
            }
        )
        
    except Exception as e:
        flash(f"Error generating floor plan: {str(e)}")
        return redirect(url_for('generate_page'))

if __name__ == "__main__":
    app.run(debug=True)