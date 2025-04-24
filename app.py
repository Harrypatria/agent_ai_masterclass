import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import openai
import base64

# Set page configuration
st.set_page_config(
    page_title="AI Health Copilot",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Add custom CSS for styling and animations
st.markdown(
    """
    <style>
    /* General Styling */
    body {
        background-color: #f9f9f9;
        color: #333;
    }
    /* Purple Hover Effects */
    button:hover, a:hover, [role="button"]:hover, label:hover, div[data-testid="stFileUploadDropzone"] div:hover {
        background-color: #8e44ad !important;
        color: #fff !important;
    }
    /* Sidebar Padding Optimized */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 10px !important;
        padding-left: 10px !important;
        padding-right: 10px !important;
        padding-bottom: 10px !important;
    }
    /* Input Field Animations */
    .input-field input {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
        transition: border-color 0.3s ease;
        width: 100%;
    }
    .input-field input:focus {
        border-color: #8e44ad;
        outline: none;
    }
    /* Footer Styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #8e44ad;
        color: white;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo in the sidebar
def display_logo():
    try:
        with open("logo.png", "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        st.sidebar.markdown(
            f'<div style="text-align:center;"><img src="data:image/png;base64,{img_base64}" width="200"></div>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.sidebar.error(f"Error loading logo: {e}")

display_logo()

# Getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Loading the saved models
diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))

# Function to generate OpenAI response
def generate_openai_response(prompt, api_key, model="gpt-4"):
    """Generate response using OpenAI Chat API."""
    client = openai.OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Error: {e}"

# Sidebar for navigation
with st.sidebar:
    st.header("🔑 API Configuration")
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = ''

    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key to access the service",
        value=st.session_state.openai_api_key,
        key="api_key_input"
    )

    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key
        st.success("API Key accepted!")
    else:
        st.warning("⚠️ Please enter your OpenAI API Key to proceed")
        st.markdown("[Get your API key here](https://platform.openai.com/signup/)")
        selected = None  # Prevent selection until API key is provided

    selected = option_menu(
        'AI Health Copilot',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Personalized Health Plan'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person', 'gear'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    def input_field(label, min_val, max_val, key):
        st.markdown(f'<p class="input-label">{label}</p>', unsafe_allow_html=True)
        return st.number_input('', min_value=min_val, max_value=max_val, key=key)

    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = input_field('Number of Pregnancies', 0, 17, key="diabetes_pregnancies")
    with col2:
        Glucose = input_field('Glucose Level', 0, 199, key="diabetes_glucose")
    with col3:
        BloodPressure = input_field('Blood Pressure value', 0, 122, key="diabetes_bp")
    with col1:
        SkinThickness = input_field('Skin Thickness value', 0, 99, key="diabetes_skin")
    with col2:
        Insulin = input_field('Insulin Level', 0, 846, key="diabetes_insulin")
    with col3:
        BMI = input_field('BMI value', 0.0, 67.1, key="diabetes_bmi")
    with col1:
        DiabetesPedigreeFunction = input_field('Diabetes Pedigree Function value', 0.0, 2.42, key="diabetes_pedigree")
    with col2:
        Age = input_field('Age of the Person', 21, 81, key="diabetes_age")

    diab_diagnosis = ''
    if st.button('Diabetes Test Result', key="diabetes_test_button"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
        diab_prediction = diabetes_model.predict([user_input])
        diab_diagnosis = 'The person is diabetic' if diab_prediction[0] == 1 else 'The person is not diabetic'

        # Chatbot explanation
        if st.session_state.openai_api_key:
            prompt = f"Explain the reasons for {diab_diagnosis} diagnosis based on predictors of {diab_prediction} Provide suggestions. Total max 3 paragraph succinctly"
            response = generate_openai_response(prompt, st.session_state.openai_api_key)
            st.write(response)
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    def input_field(label, min_val, max_val, key):
        st.markdown(f'<p class="input-label">{label}</p>', unsafe_allow_html=True)
        return st.number_input('', min_value=min_val, max_value=max_val, key=key)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = input_field('Age', 29, 77, key="heart_age")
    with col2:
        sex = input_field('Sex', 0, 1, key="heart_sex")
    with col3:
        cp = input_field('Chest Pain types', 0, 3, key="heart_cp")
    with col1:
        trestbps = input_field('Resting Blood Pressure', 94, 200, key="heart_trestbps")
    with col2:
        chol = input_field('Serum Cholestoral in mg/dl', 126, 564, key="heart_chol")
    with col3:
        fbs = input_field('Fasting Blood Sugar > 120 mg/dl', 0, 1, key="heart_fbs")
    with col1:
        restecg = input_field('Resting Electrocardiographic results', 0, 2, key="heart_restecg")
    with col2:
        thalach = input_field('Maximum Heart Rate achieved', 71, 202, key="heart_thalach")
    with col3:
        exang = input_field('Exercise Induced Angina', 0, 1, key="heart_exang")
    with col1:
        oldpeak = input_field('ST depression induced by exercise', 0.0, 6.2, key="heart_oldpeak")
    with col2:
        slope = input_field('Slope of the peak exercise ST segment', 0, 2, key="heart_slope")
    with col3:
        ca = input_field('Major vessels colored by fluorosopy', 0, 4, key="heart_ca")
    with col1:
        thal = input_field('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', 0, 3, key="heart_thal")

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result', key="heart_test_button"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        heart_prediction = heart_disease_model.predict([user_input])
        heart_diagnosis = 'The person is having heart disease' if heart_prediction[0] == 1 else 'The person does not have any heart disease'

        # Chatbot explanation
        if st.session_state.openai_api_key:
            prompt = f"Explain the reasons for {heart_diagnosis} diagnosis based on predictors of {heart_prediction} the user filled out. Provide suggestions. Total max 3 paragraph succinctly"
            response = generate_openai_response(prompt, st.session_state.openai_api_key)
            st.write(response)
    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    def input_field(label, min_val, max_val, key):
        st.markdown(f'<p class="input-label">{label}</p>', unsafe_allow_html=True)
        return st.number_input('', min_value=min_val, max_value=max_val, key=key)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        fo = input_field('MDVP:Fo(Hz)', 88.33, 260.10, key="parkinsons_fo")
    with col2:
        fhi = input_field('MDVP:Fhi(Hz)', 102.14, 592.03, key="parkinsons_fhi")
    with col3:
        flo = input_field('MDVP:Flo(Hz)', 65.47, 239.17, key="parkinsons_flo")
    with col4:
        Jitter_percent = input_field('MDVP:Jitter(%)', 0.00168, 0.03316, key="parkinsons_jitter_percent")
    with col5:
        Jitter_Abs = input_field('MDVP:Jitter(Abs)', 0.000007, 0.00261, key="parkinsons_jitter_abs")
    with col1:
        RAP = input_field('MDVP:RAP', 0.0068, 0.02144, key="parkinsons_rap")
    with col2:
        PPQ = input_field('MDVP:PPQ', 0.003446, 0.01958, key="parkinsons_ppq")
    with col3:
        DDP = input_field('Jitter:DDP', 0.00204, 0.06433, key="parkinsons_ddp")
    with col4:
        Shimmer = input_field('MDVP:Shimmer', 0.019, 0.119, key="parkinsons_shimmer")
    with col5:
        Shimmer_dB = input_field('MDVP:Shimmer(dB)', 0.165, 0.378, key="parkinsons_shimmer_db")
    with col1:
        APQ3 = input_field('Shimmer:APQ3', 0.0165, 0.0378, key="parkinsons_apq3")
    with col2:
        APQ5 = input_field('Shimmer:APQ5', 0.0165, 0.0378, key="parkinsons_apq5")
    with col3:
        APQ = input_field('MDVP:APQ', 0.0165, 0.0378, key="parkinsons_apq")
    with col4:
        DDA = input_field('Shimmer:DDA', 0.0165, 0.0378, key="parkinsons_dda")
    with col5:
        NHR = input_field('NHR', 0.0165, 0.0378, key="parkinsons_nhr")
    with col1:
        HNR = input_field('HNR', 0.0165, 0.0378, key="parkinsons_hnr")
    with col2:
        RPDE = input_field('RPDE', 0.0165, 0.0378, key="parkinsons_rpde")
    with col3:
        DFA = input_field('DFA', 0.0165, 0.0378, key="parkinsons_dfa")
    with col4:
        spread1 = input_field('spread1', 0.0165, 0.0378, key="parkinsons_spread1")
    with col5:
        spread2 = input_field('spread2', 0.0165, 0.0378, key="parkinsons_spread2")
    with col1:
        D2 = input_field('D2', 0.0165, 0.0378, key="parkinsons_d2")
    with col2:
        PPE = input_field('PPE', 0.0165, 0.0378, key="parkinsons_ppe")

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result", key="parkinsons_test_button"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        parkinsons_prediction = parkinsons_model.predict([user_input])
        parkinsons_diagnosis = "The person has Parkinson's disease" if parkinsons_prediction[0] == 1 else "The person does not have Parkinson's disease"

        # Chatbot explanation
        if st.session_state.openai_api_key:
            prompt = f"Explain the reasons for {parkinsons_diagnosis} diagnosis based on predictors of {parkinsons_prediction} the user filled out. Provide suggestions. Total max 3 paragraph succinctly"
            response = generate_openai_response(prompt, st.session_state.openai_api_key)
            st.write(response)
    st.success(parkinsons_diagnosis)

# Personalized Health Plan Page
if selected == "Personalized Health Plan":
    st.title("Personalized Health & Fitness Planner")

    def display_dietary_plan(plan_content):
        with st.expander("📋 Your Personalized Dietary Plan", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 🎯 Why this plan works")
                st.info(plan_content.get("why_this_plan_works", "Information not available"))
                st.markdown("### 🍽️ Meal Plan")
                st.write(plan_content.get("meal_plan", "Plan not available"))
            with col2:
                st.markdown("### ⚠️ Important Considerations")
                considerations = plan_content.get("important_considerations", "").split('\n')
                for consideration in considerations:
                    if consideration.strip():
                        st.warning(consideration)

    def display_fitness_plan(plan_content):
        with st.expander("💪 Your Personalized Fitness Plan", expanded=True):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 🎯 Goals")
                st.success(plan_content.get("goals", "Goals not specified"))
                st.markdown("### 🏋️‍♂️ Exercise Routine")
                st.write(plan_content.get("routine", "Routine not available"))
            with col2:
                st.markdown("### 💡 Pro Tips")
                tips = plan_content.get("tips", "").split('\n')
                for tip in tips:
                    if tip.strip():
                        st.info(tip)

    def main():
        if 'dietary_plan' not in st.session_state:
            st.session_state.dietary_plan = {}
            st.session_state.fitness_plan = {}
            st.session_state.qa_pairs = []
            st.session_state.plans_generated = False

        st.markdown("""
            <div style='background-color: #f0f8ff; padding: 1rem; border-radius: 0.5rem; margin-bottom: 2rem;'>
                <p style='color: #333333; font-size: 1.1rem; font-family: Arial, sans-serif;'>
                Get personalized dietary and fitness plans tailored to your goals and preferences.
                Our AI-powered system considers your unique profile to create the perfect plan for you.
                </p>
            </div>
        """, unsafe_allow_html=True)

        if st.session_state.openai_api_key:
            st.header("👤 Your Profile")
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=10, max_value=100, step=1, help="Enter your age", key="profile_age")
                height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, step=0.1, key="profile_height")
                activity_level = st.selectbox(
                    "Activity Level",
                    options=["Sedentary", "Lightly Active", "Moderately Active", "Very Active", "Extremely Active"],
                    help="Choose your typical activity level",
                    key="profile_activity"
                )
                dietary_preferences = st.selectbox(
                    "Dietary Preferences",
                    options=["Vegetarian", "Keto", "Gluten Free", "Low Carb", "Dairy Free"],
                    help="Select your dietary preference",
                    key="profile_diet"
                )
            with col2:
                weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, step=0.1, key="profile_weight")
                sex = st.selectbox("Sex", options=["Male", "Female", "Other"], key="profile_sex")
                fitness_goals = st.selectbox(
                    "Fitness Goals",
                    options=["Lose Weight", "Gain Muscle", "Endurance", "Stay Fit", "Strength Training"],
                    help="What do you want to achieve?",
                    key="profile_goals"
                )

            if st.button("🎯 Generate My Personalized Plan", use_container_width=True, key="generate_plan_button"):
                with st.spinner("Creating your perfect health and fitness routine..."):
                    try:
                        user_profile = f"""
                        Age: {age}
                        Weight: {weight}kg
                        Height: {height}cm
                        Sex: {sex}
                        Activity Level: {activity_level}
                        Dietary Preferences: {dietary_preferences}
                        Fitness Goals: {fitness_goals}
                        """
                        dietary_prompt = f"""
                        You are a dietary expert. Based on the following user profile, generate a personalized meal plan for the day (breakfast, lunch, dinner, and snacks), explain why this plan works, and list important considerations:
                        {user_profile}
                        """
                        fitness_prompt = f"""
                        You are a fitness expert. Based on the following user profile, generate a personalized fitness routine (warm-up, main workout, and cool-down), explain the benefits of the routine, and provide pro tips:
                        {user_profile}
                        """
                        dietary_response = generate_openai_response(dietary_prompt, st.session_state.openai_api_key)
                        fitness_response = generate_openai_response(fitness_prompt, st.session_state.openai_api_key)
                        dietary_plan = {
                            "why_this_plan_works": "Balanced macronutrients for the user's goals.",
                            "meal_plan": dietary_response,
                            "important_considerations": """
                            - Hydration: Drink plenty of water throughout the day
                            - Electrolytes: Monitor sodium, potassium, and magnesium levels
                            - Fiber: Ensure adequate intake through vegetables and fruits
                            - Listen to your body: Adjust portion sizes as needed
                            """
                        }
                        fitness_plan = {
                            "goals": "Build strength, improve endurance, and maintain overall fitness",
                            "routine": fitness_response,
                            "tips": """
                            - Track your progress regularly
                            - Allow proper rest between workouts
                            - Focus on proper form
                            - Stay consistent with your routine
                            """
                        }
                        st.session_state.dietary_plan = dietary_plan
                        st.session_state.fitness_plan = fitness_plan
                        st.session_state.plans_generated = True
                        st.session_state.qa_pairs = []
                        display_dietary_plan(dietary_plan)
                        display_fitness_plan(fitness_plan)
                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")

    main()

# Add footer
st.markdown(
    """
    <div class="footer">
        <span class="disclaimer-icon" 
              title="Disclaimer: AI may not always provide accurate or complete information. 
                     Agentic AI x Corporate Learning Division">
              ℹ️
        </span>
        <span>All rights reserved. © 2025 Patria & Co.</span>
    </div>
    """,
    unsafe_allow_html=True
)