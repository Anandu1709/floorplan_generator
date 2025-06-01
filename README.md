🏠 AI-Powered Architectural Floor Plan Generator

A web-based application that intelligently generates realistic 2D architectural floor plans using user input such as the number of bedrooms, bathrooms, square footage, and garage spaces. Powered by Stable Diffusion and enhanced by nearest-neighbor reference retrieval, this tool helps architects and designers rapidly prototype floor plan layouts.



✨ Features
🔍 Input-based retrieval of similar floor plans using KNN

🎨 AI-generated 2D blueprint-style floor plans using Stable Diffusion

🧠 Enhanced prompt engineering for architectural details

🗂 Upload interface and visual outputs via Flask web app

⚙️ Support for GPU acceleration if available

🧰 Tech Stack
Component	Technology

Frontend	HTML + Jinja2 templates (Flask)

Backend	Python (Flask, scikit-learn, pandas)

Image Generation	Stable Diffusion v1.5

ML Support	PyTorch, diffusers

Data	Floor Plan Dataset (CSV + Images)


📁 Folder Structure

AI-FloorPlan-Generator/

 app.py                     # Flask backend with main logic
 house_plans_details.csv    # Dataset (CSV with floor plan attributes)
 
 images/

  └── images/                # Folder containing reference floor plan images

static/

 └── uploads/               # Uploaded files go here
   
  └── generated/             # AI-generated outputs stored here

 templates/

  ├── index.html             # Homepage

   ├── generate.html          # Input form page

  |── results.html           # Output display page
  

└── README.md                  # This file


📊 Dataset Requirements

To run this project, you’ll need a dataset structured as follows:

Floor plan image containing 2640 floor plan images available on kaggle.

CSV file: house_plans_details.csv

Columns required:

Beds

Baths

Square Feet

Garages (optional)

Image Path (relative path to floor plan image)

Image folder: images/images/

Contains corresponding floor plan images used for similarity matching.

If using your own dataset, ensure that image paths listed in the CSV are accurate and consistent.

🚀 How to Run the Project

🔧 Step 1: Clone the Repo

git clone https://github.com/your-username/AI-FloorPlan-Generator.git

cd AI-FloorPlan-Generator

📦 Step 2: Install Dependencies

It is recommended to use a virtual environment:

python -m venv venv

source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt

If requirements.txt isn't created yet, generate one with:

pip freeze > requirements.txt

📂 Step 3: Prepare Your Dataset

Place your CSV file in the root directory.

Place your floor plan images inside images/images/.

🧠 Step 4: Launch the Web App

python app.py

Then open your browser and navigate to:

👉 http://127.0.0.1:5000

💡 Usage Instructions:
Go to "Generate Page"
Enter your desired floor plan attributes (e.g., bedrooms, bathrooms, etc.)
The app retrieves 3–5 similar existing plans.
Then it generates a new AI-powered floor plan based on your input.
View the result and download the generated blueprint.

📌 Notes:
If you have a GPU, the app will automatically use it for faster generation.
If running on CPU, image generation may take significantly longer.
Negative prompts are used to avoid generating 3D or colored scenes.

🏗️ Future Improvements:
Add downloadable PDF/PNG outputs.
Improve dataset diversity.
Integrate ControlNet for precise layout control.
Add user login and history of generated plans.

🧑‍💻 Author

Anandu Biny

AI and Data Science Specialist


📜 License

This project is licensed under the MIT License. See the LICENSE file for more details.
