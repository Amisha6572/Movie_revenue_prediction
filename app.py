# # 

# import streamlit as st
# import pickle
# import numpy as np
# import pandas as pd

# # Load Model + Scaler
# model = pickle.load(open("movie_revenue_model.pkl", "rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))
# feature_names = pickle.load(open("feature_names.pkl", "rb"))

# st.title("🎬 Movie Revenue Prediction App")

# st.write("Enter movie details to predict revenue.")

# # Example Inputs (Change according to dataset features)
# budget = st.number_input("Enter Budget (INR)", min_value=0)
# screens = st.number_input("Enter Number of Screens", min_value=1)

# if st.button("Predict Revenue"):

#     # Prepare input
#     input_df = pd.DataFrame(columns=feature_names)
#     input_data = np.array([[budget, screens]])

#     # Scale input
#     input_scaled = scaler.transform(input_data)

#     # Predict (log scale)
#     prediction_log = model.predict(input_scaled)

#     # Convert back to INR
#     prediction = np.expm1(prediction_log)

#     st.success(f"Predicted Revenue: ₹ {prediction[0]:,.2f}")


# import streamlit as st
# import pandas as pd
# import joblib

# # ---------------------------------------------------
# # Load Saved Pipeline + Columns
# # ---------------------------------------------------
# pipeline = joblib.load("movie_pipeline.pkl")
# training_columns = joblib.load("training_columns.pkl")

# st.title("🎬 Movie Revenue Prediction App")

# st.write("Enter movie details below:")

# # ---------------------------------------------------
# # User Inputs (Only Main Ones)
# # ---------------------------------------------------
# budget = st.number_input("Budget (INR)", min_value=0)
# screens = st.number_input("Number of Screens", min_value=1)

# genre = st.selectbox("Genre", ["Drama", "Action", "Comedy", "Thriller", "Other"])
# release_period = st.selectbox("Release Period", ["Holiday", "Normal", "Other"])

# whether_franchise = st.selectbox("Whether Franchise", ["Yes", "No"])
# whether_remake = st.selectbox("Whether Remake", ["Yes", "No"])

# new_actor = st.selectbox("New Actor", ["Yes", "No"])
# new_director = st.selectbox("New Director", ["Yes", "No"])
# new_music_director = st.selectbox("New Music Director", ["Yes", "No"])

# # ---------------------------------------------------
# # Predict Button
# # ---------------------------------------------------
# if st.button("Predict Revenue"):

#     # Step 1: Create empty row with ALL training columns
#     input_data = pd.DataFrame(columns=training_columns)
#     input_data.loc[0] = 0   # Fill numeric default

#     # Step 2: Fill user provided values
#     input_data["Budget(INR)"] = budget
#     input_data["Number of Screens"] = screens

#     input_data["Genre"] = genre
#     input_data["Release Period"] = release_period

#     input_data["Whether Franchise"] = whether_franchise
#     input_data["Whether Remake"] = whether_remake

#     input_data["New Actor"] = new_actor
#     input_data["New Director"] = new_director
#     input_data["New Music Director"] = new_music_director

#     # Step 3: Predict
#     prediction = pipeline.predict(input_data)[0]

#     st.success(f"🎉 Predicted Revenue: ₹ {prediction:,.2f}")
#--------------------------------------------------------------------------------------------
# second update
#--------------------------------------------------------------------------------------------




# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # ============================================================
# # Load Pipeline and Feature Names
# # ============================================================

# pipeline = joblib.load("movie_pipeline.pkl")
# features = joblib.load("training_columns.pkl")
# categorical_cols = joblib.load("categorical_cols.pkl")
# numeric_cols = joblib.load("numeric_cols.pkl")
# top_categories = joblib.load("top_categories.pkl")


# st.title("🎬 Movie Revenue Prediction App")

# st.write("Enter all movie details below:")

# # ============================================================
# # Create Input Form for ALL Features
# # ============================================================

# user_input = {}



# for col in categorical_cols:

#     # # Detect if column was categorical or numeric
#     # if col in pipeline.named_steps["preprocessing"].transformers_[1][2]:
#     #     user_input[col] = st.text_input(f"{col}", "")
#     # else:
#     #     user_input[col] = st.number_input(f"{col}", value=0.0)
#     if col in top_categories:
#         options = list(top_categories[col]) + ["Other"]
#         user_input[col] = st.selectbox(col, options)

#     # Else numeric feature → number input
#     else:
#         user_input[col] = st.number_input(f"{col}", value=0.0)


# # Convert input into DataFrame
# input_df = pd.DataFrame([user_input])

# # ============================================================
# # Prediction Button
# # ============================================================

# # if st.button("Predict Revenue"):

# #     prediction_log = pipeline.predict(input_df)[0]

# #     # Convert back from log scale
# #     prediction = np.expm1(prediction_log)

# #     st.success(f"🎉 Predicted Movie Revenue = ₹ {prediction:,.2f}")

# if st.button("🎯 Predict Revenue"):

#     try:
#         prediction_log = pipeline.predict(input_df)[0]

#         # Reverse log transform
#         prediction = np.expm1(prediction_log)

#         st.subheader("📌 Predicted Revenue:")
#         st.success(f"₹ {prediction:,.2f} Crores")

#     except Exception as e:
#         st.error("❌ Prediction Failed")
#         st.write("Error:", e)

#---------------------------------------------------------------------------------------------
# third update  
#---------------------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib

# # Load saved assets
# @st.cache_resource
# def load_assets():
#     pipeline = joblib.load("movie_pipeline.pkl")
#     cat_cols = joblib.load("categorical_cols.pkl")
#     num_cols = joblib.load("numeric_cols.pkl")
#     top_categories = joblib.load("top_categories.pkl")
#     return pipeline, cat_cols, num_cols, top_categories

# pipeline, cat_cols, num_cols, top_categories = load_assets()

# st.title("🎬 Movie Revenue Predictor")

# user_input = {}

# st.subheader("📌 Movie Characteristics")
# col1, col2 = st.columns(2)

# # Handle Categorical Inputs (Dropdowns)
# for i, col in enumerate(cat_cols):
#     container = col1 if i % 2 == 0 else col2
#     if col in top_categories:
#         options = top_categories[col] + ["Other"]
#         user_input[col] = container.selectbox(f"Select {col}", options)
#     else:
#         # For columns like 'Whether Remake' if they weren't dropped
#         user_input[col] = container.selectbox(f"{col}", ["No", "Yes"])

# st.subheader("📊 Numerical Statistics")
# col3, col4 = st.columns(2)

# # Handle Numerical Inputs
# for i, col in enumerate(num_cols):
#     container = col3 if i % 2 == 0 else col4
#     user_input[col] = container.number_input(f"Enter {col}", min_value=0, value=1000)

# # Prediction Logic
# if st.button("🎯 Predict Revenue"):
#     try:
#         # Create DataFrame for prediction
#         input_df = pd.DataFrame([user_input])
        
#         # Make prediction
#         pred_log = pipeline.predict(input_df)[0]
        
#         # Convert back from log scale
#         prediction = np.expm1(pred_log)

#         st.markdown("---")
#         st.metric(label="Estimated Revenue", value=f"₹ {prediction:,.2f}")
        
#     except Exception as e:
#         st.error(f"Prediction Error: {e}")


#============================================================================================
# Final Update - Cleaned and Optimized Code 
#============================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved assets
@st.cache_resource
def load_assets():
    pipeline = joblib.load("movie_pipeline.pkl")
    cat_cols = joblib.load("categorical_cols.pkl")
    num_cols = joblib.load("numeric_cols.pkl")
    top_categories = joblib.load("top_categories.pkl")
    return pipeline, cat_cols, num_cols, top_categories

pipeline, cat_cols, num_cols, top_categories = load_assets()

st.set_page_config(page_title="Movie Revenue Predictor", layout="wide")
st.title("🎬 Bollywood Movie Revenue Predictor")
st.write("Fill in the details below to estimate the box office collection.")

user_input = {}

# --- Section 1: Categorical Inputs ---
st.subheader("📽️ Movie Details")
col1, col2 = st.columns(2)

for i, col in enumerate(cat_cols):
    # Alternate between two columns for a better UI
    container = col1 if i % 2 == 0 else col2
    
    if col in top_categories:
        # This will now include Genre, Lead Star, etc.
        options = sorted(top_categories[col])
        if "Other" not in options:
            options.append("Other")
        user_input[col] = container.selectbox(f"Select {col}", options)
    else:
        # For simple Yes/No columns if any exist
        user_input[col] = container.selectbox(f"{col}", ["No", "Yes"])

# --- Section 2: Numerical Inputs ---
st.markdown("---")
st.subheader("📊 Statistics & Budget")
col3, col4 = st.columns(2)

for i, col in enumerate(num_cols):
    container = col3 if i % 2 == 0 else col4
    user_input[col] = container.number_input(f"Enter {col}", min_value=0, value=1000000, step=50000)

# --- Prediction Logic ---
st.markdown("###")
if st.button("🎯 Predict Revenue", use_container_width=True):
    try:
        # 1. Create DataFrame
        input_df = pd.DataFrame([user_input])
        
        # 2. Predict (output is in log scale)
        pred_log = pipeline.predict(input_df)[0]
        
        # 3. Inverse log transform
        prediction = np.expm1(pred_log)

        # 4. Display results
        st.success(f"### Predicted Revenue: ₹ {prediction:,.2f}")
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")