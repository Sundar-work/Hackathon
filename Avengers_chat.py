import streamlit as st
import boto3
import pandas as pd
import plotly.express as px
import json

# Initialize Boto3 clients
s3_client = boto3.client('s3')
comprehend_client = boto3.client('comprehend')
bedrock_client = boto3.client('bedrock')  # AWS Bedrock for Claude

S3_BUCKET = 'YOUR_S3_BUCKET_NAME'  # Replace with your S3 bucket name
CLAUDE_MODEL_ID = 'claude-3-sonnet'  # Use the Claude 3 (Sonnet) model ID

def load_data_from_s3(bucket, key):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    file_content = response['Body'].read().decode('utf-8')
    
    if key.endswith('.csv'):
        data = pd.read_csv(file_content)
    elif key.endswith('.xlsx'):
        data = pd.read_excel(file_content)
    elif key.endswith('.json'):
        data = pd.read_json(file_content)
    elif key.endswith('.parquet'):
        data = pd.read_parquet(file_content)
    else:
        raise ValueError("Unsupported file type")
    
    data.columns = data.columns.str.lower()
    return data

def process_query_comprehend(query):
    response = comprehend_client.detect_entities(
        Text=query,
        LanguageCode='en'
    )
    
    entities = response['Entities']
    
    chart_types = ["bar", "line", "scatter", "histogram", "pie", "box", "heatmap", "violin", "map"]
    
    attributes = []
    chart_type = None
    
    for entity in entities:
        if entity['Type'] == 'QUANTITY':
            attributes.append(entity['Text'].lower())
        elif entity['Type'] == 'OTHER' and entity['Text'].lower() in chart_types:
            chart_type = entity['Text'].lower()
    
    return chart_type, attributes

def generate_text_with_claude(prompt):
    response = bedrock_client.invoke_model(
        ModelId=CLAUDE_MODEL_ID,
        ContentType='application/json',
        Accept='application/json',
        Body=json.dumps({"text": prompt})
    )
    
    result = json.loads(response['Body'].read().decode('utf-8'))
    return result['text']

def determine_chart_type_with_claude(query):
    prompt = f"Determine the most appropriate chart type for the following query: {query}"
    chart_type = generate_text_with_claude(prompt)
    return chart_type.strip().lower()

def generate_data_summary(data):
    data_dict = data.to_dict()
    prompt = (
        "Based on the following dataset, provide a detailed summary about what this data is about, "
        "explaining the overall nature and key insights of the data. Here is the dataset:\n\n"
        f"{json.dumps(data_dict, indent=2)}"
    )
    summary = generate_text_with_claude(prompt)
    return summary

st.title("Interactive Data Visualization")
st.write("Upload your data file and enter your query below:")

with st.expander("Upload Data File"):
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'json', 'parquet', 'txt'])

if uploaded_file is not None:
    file_key = uploaded_file.name
    s3_client.upload_fileobj(uploaded_file, S3_BUCKET, file_key)
    data = load_data_from_s3(S3_BUCKET, file_key)
    columns = data.columns.tolist()
    st.write("Data loaded successfully!")
    
    with st.expander("Data Preview"):
        st.write("Here's a preview of your data:")
        st.dataframe(data.head())  # Show a preview of the data
    
    summary = data.describe(include='all').transpose()
    with st.expander("Statistics"):
        st.write(summary)

    # Generate and display data summary using Claude
    data_summary = generate_data_summary(data)
    with st.expander("Data Summary"):
        st.write(data_summary)

    user_query = st.text_input("Query", key='user_query')
    
    if st.button('Submit'):
        if user_query:
            with st.spinner('Processing...'):
                chart_type, attributes = process_query_comprehend(user_query)
                st.write(f"Extracted attributes: {attributes}")
                
                if len(attributes) < 2:
                    st.error("Not enough attributes found for the query.")
                else:
                    if not chart_type:
                        chart_type = determine_chart_type_with_claude(user_query)
                        st.write(f"Determined chart type: {chart_type}")
                    
                    x, y = attributes[0], attributes[1]
                    if chart_type == 'bar':
                        fig = px.bar(data, x=x, y=y)
                    elif chart_type == 'line':
                        fig = px.line(data, x=x, y=y)
                    elif chart_type == 'scatter':
                        fig = px.scatter(data, x=x, y=y)
                    elif chart_type == 'histogram':
                        fig = px.histogram(data, x=x, y=y)
                    elif chart_type == 'pie':
                        fig = px.pie