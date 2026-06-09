import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model bundle
bundle = joblib.load("models/best_model.joblib")

# Get the model, selected features, and category thresholds
model = bundle["model"]
feature_columns = bundle["feature_columns"]
thresholds = bundle["category_thresholds"]

# Create session state lists to store prediction history
if "zone1_pwr" not in st.session_state:
    st.session_state.zone1_pwr = []

if "zone2_pwr" not in st.session_state:
    st.session_state.zone2_pwr = []

if "zone3_pwr" not in st.session_state:
    st.session_state.zone3_pwr = []

if "temp" not in st.session_state:
    st.session_state.temp = []

if "humid" not in st.session_state:
    st.session_state.humid = []

if "hd" not in st.session_state:
    st.session_state.hd = []


# Set the Streamlit page layout and title
st.set_page_config(
    page_title="Power Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Main page heading
st.title("Power Consumption Predictor")

st.markdown(
    """
    Predict Zone 1 Power Consumption using the trained
    Random Forest Model.
    """
)

# Create tabs for input, model details, and statistics
tab1, tab2, tab3 = st.tabs(
    ["Input Area", "Model Information", "Statistics"]
)

with tab1:
    # Create two columns for input and output
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Features")

        # User inputs for the model
        zone2_power = st.number_input(
            "Zone 2 Power Consumption",
            min_value=0.0,
            value=20000.0
        )

        zone3_power = st.number_input(
            "Zone 3 Power Consumption",
            min_value=0.0,
            value=18000.0
        )

        temperature = st.number_input(
            "Temperature (°C)",
            value=22.0
        )

        humidity = st.slider(
            "Humidity (%)",
            0,
            100,
            60
        )

        hour = st.slider(
            "Hour of Day",
            0,
            23,
            12
        )

        # Calculate heating and cooling degree values
        hdd = max(0, 18 - temperature)
        cdd = max(0, temperature - 24)

        st.markdown("*Those are calculated based on the provided values.*")
        st.write(f"Heating Degree Day (HDD): {hdd:.2f}")
        st.write(f"Cooling Degree Day (CDD): {cdd:.2f}")
        st.divider()

    with col2:
        st.subheader("Predicted Results")

        # Create extra features used by the trained model
        power_sum_23 = zone2_power + zone3_power

        # Convert hour into cyclic features because 23:00 and 00:00 are close in time
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        # Run prediction when the user clicks the button
        if st.button("Predict Power Consumption"):

            with st.spinner("Predicting power consumption..."):

                # Create input data in the same column order used during training
                input_data = pd.DataFrame(
                    [[
                        zone2_power,
                        zone3_power,
                        power_sum_23,
                        hour_sin,
                        hour_cos,
                        temperature,
                        humidity,
                        hdd,
                        cdd
                    ]],
                    columns=feature_columns
                )

                # Predict Zone 1 power consumption
                prediction = model.predict(input_data)[0]

                # Convert the prediction into a readable consumption category
                if prediction <= thresholds["low"]:
                    category = "Low"
                elif prediction <= thresholds["medium"]:
                    category = "Medium"
                elif prediction <= thresholds["high"]:
                    category = "High"
                else:
                    category = "Very High"

                # Save prediction history for the statistics tab
                st.session_state.zone1_pwr.append(prediction)
                st.session_state.zone2_pwr.append(zone2_power)
                st.session_state.zone3_pwr.append(zone3_power)
                st.session_state.temp.append(temperature)
                st.session_state.humid.append(humidity)
                st.session_state.hd.append(hour)

                st.success(
                    f"Predicted Zone 1 Power Consumption: {prediction:.2f} W"
                )

                # Set progress bar colour and width based on prediction category
                if category == "Low":
                    width = 25
                    color = "#28a745"
                elif category == "Medium":
                    width = 50
                    color = "#ffc107"
                elif category == "High":
                    width = 75
                    color = "#fd7e14"
                else:
                    width = 100
                    color = "#dc3545"

                # Display the consumption level using a custom HTML bar
                st.markdown(
                    f"""
                    <h4>⚡ Consumption Level</h4>

                    <div style="
                        width:100%;
                        height:24px;
                        background:#e0e0e0;
                        border-radius:12px;
                        overflow:hidden;
                    ">
                        <div style="
                            width:{width}%;
                            height:100%;
                            background:{color};
                        "></div>
                    </div>

                    <p style="
                        font-size:20px;
                        font-weight:bold;
                        color:{color};
                        margin-top:10px;
                    ">
                        {category}
                    </p>
                    """,
                    unsafe_allow_html=True
                )

    # Show all feature values used for prediction
    st.subheader("Feature Summary")

    summary_df = pd.DataFrame({
        "Feature": feature_columns,
        "Value": [
            zone2_power,
            zone3_power,
            power_sum_23,
            hour_sin,
            hour_cos,
            temperature,
            humidity,
            hdd,
            cdd
        ]
    })

    st.dataframe(summary_df, use_container_width=True)
    st.markdown("---")

with tab2:
    # Display basic information about the loaded model
    st.subheader("Model Information")
    st.write(type(model))
    st.write(model)

with tab3:
    st.text(
        "Predicted Zone 1 power consumption vs Zone 2 and Zone 3 power consumption graph"
    )

    # Create a DataFrame from prediction history
    df = pd.DataFrame({
        "Temperature": st.session_state.temp,
        "Zone1_pwr": st.session_state.zone1_pwr,
        "Zone2_pwr": st.session_state.zone2_pwr,
        "Zone3_pwr": st.session_state.zone3_pwr,
        "Hour": st.session_state.hd,
        "Humidity": st.session_state.humid
    })

    # Let the user choose which variable to use on the x-axis
    sel = st.selectbox(
        "Select X option",
        ["Hour", "Humidity", "Temperature"]
    )

    # Show the line chart based on the selected x-axis
    if sel == "Hour":
        st.line_chart(
            df,
            x="Hour",
            y=["Zone1_pwr", "Zone2_pwr", "Zone3_pwr"]
        )
    elif sel == "Humidity":
        st.line_chart(
            df,
            x="Humidity",
            y=["Zone1_pwr", "Zone2_pwr", "Zone3_pwr"]
        )
    else:
        st.line_chart(
            df,
            x="Temperature",
            y=["Zone1_pwr", "Zone2_pwr", "Zone3_pwr"]
        )


st.caption(
    "COS40007 AI Engineering | Zone 1 Power Prediction"
)