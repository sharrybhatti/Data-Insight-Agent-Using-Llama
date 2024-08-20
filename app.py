import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import os
from langchain_ollama import OllamaLLM

# Function to generate basic insights from the DataFrame
def generate_basic_insights(data):
    insights = {}
    insights['columns'] = data.columns.tolist()
    insights['missing_values'] = data.isnull().sum().to_dict()
    insights['data_types'] = data.dtypes.to_dict()
    insights['description'] = data.describe(include='all').to_dict()  
    return insights

# Function to create the report writer agent
def create_report_writer_agent():
    llm = OllamaLLM(model="llama3.1")  
    return llm  

# Function to generate a summary report
def generate_report(llm, insights):
    prompt = f"""
    Based on the following insights, create a concise summary report:

    Insights:
    {insights}
    """
    report = llm.invoke(prompt)
    return report

# Streamlit app title
st.title("CSV Data Insights App")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Check if the uploaded file is empty
    if uploaded_file.size == 0:
        st.error("The uploaded file is empty. Please upload a valid CSV file.")
    else:
        # Load CSV into DataFrame
        try:
            data = pd.read_csv(uploaded_file)
            # Debugging: Print the DataFrame to check its contents
            st.write("Data Loaded:")
            st.write(data)

            if data.empty:
                st.error("The uploaded CSV file does not contain any data. Please upload a valid CSV file.")
            else:
                st.success("CSV file uploaded successfully!")

                # Display DataFrame preview
                st.subheader("Data Preview:")
                st.write(data.head())

                # Show basic statistics
                st.subheader("Basic Statistics:")
                st.write(data.describe())

                # Select a column for visualization
                st.subheader("Select a column for graphical insights:")
                column = st.selectbox("Choose a column:", data.columns)

                # Select type of plot
                plot_type = st.selectbox("Select plot type:", ["Histogram", "Box Plot", "Scatter Plot"])

                # Generate visualizations based on user selection
                if plot_type == "Histogram":
                    st.subheader(f"Histogram of {column}")
                    plt.figure(figsize=(10, 5))
                    sns.histplot(data[column], bins=30, kde=True)
                    st.pyplot(plt)

                elif plot_type == "Box Plot":
                    st.subheader(f"Box Plot of {column}")
                    plt.figure(figsize=(10, 5))
                    sns.boxplot(y=data[column])
                    st.pyplot(plt)

                elif plot_type == "Scatter Plot":
                    # For scatter plot, we need two columns
                    if len(data.columns) > 1:
                        x_column = st.selectbox("Select X-axis column:", data.columns)
                        y_column = st.selectbox("Select Y-axis column:", data.columns)
                        st.subheader(f"Scatter Plot of {y_column} vs {x_column}")
                        plt.figure(figsize=(10, 5))
                        sns.scatterplot(data=data, x=x_column, y=y_column)
                        st.pyplot(plt)
                    else:
                        st.warning("Please select another column for the scatter plot.")

                # Get insights from the DataFrame
                if st.button("Get Insights", type="primary"):
                    insights = generate_basic_insights(data)  # Generate insights without using an agent
                    st.subheader("Insights:")
                    st.write("Columns:", insights['columns'])
                    st.write("Missing Values:", insights['missing_values'])
                    st.write("Data Types:", insights['data_types'])
                    st.write("Description:", insights['description'])

                    # Generate report from insights
                    report_writer = create_report_writer_agent()
                    report = generate_report(report_writer, insights)
                    st.subheader("Summary Report:")
                    st.write(report)

        except pd.errors.EmptyDataError:
            st.error("The uploaded file is empty. Please upload a valid CSV file.")
        except Exception as e:
            st.error(f"An error occurred while reading the CSV file: {e}")
