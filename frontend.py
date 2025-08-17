import streamlit as st
import requests
import json

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Data Science Agent", layout="wide")

st.title("🤖 AI Data Science Agent")
st.markdown("Interact with your data using natural language!")

# Upload CSV
st.subheader("📤 Upload Dataset")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(f"{API_URL}/upload/", files=files)
        
        if response.status_code == 200:
            st.success(f"Uploaded {uploaded_file.name} successfully!")
        else:
            st.error(f"Upload failed. Status: {response.status_code}")
            if response.text:
                st.error(f"Error details: {response.text}")
    except Exception as e:
        st.error(f"Upload error: {str(e)}")

# Chat interface
st.subheader("💬 Chat with Agent")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your query (e.g., 'Summarize the dataset train.csv')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent thinking..."):
            try:
                response = requests.post(f"{API_URL}/ask/", params={"query": prompt})
                
                if response.status_code == 200:
                    try:
                        # Check if response has content
                        if not response.text.strip():
                            error_msg = "Received empty response from API"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                        else:
                            result_json = response.json()
                            
                            # Check if result_json is None or empty
                            if result_json is None:
                                error_msg = "API returned null response"
                                st.error(error_msg)
                                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                            else:
                                agent_response = result_json.get("response", "No response generated")
                                
                                st.markdown(agent_response)
                                
                                # Optionally show intermediate steps in an expander
                                if "intermediate_steps" in result_json and result_json["intermediate_steps"]:
                                    with st.expander("Show intermediate steps"):
                                        st.json(result_json["intermediate_steps"])
                                
                                # Add assistant response to chat history
                                st.session_state.messages.append({"role": "assistant", "content": agent_response})
                        
                    except json.JSONDecodeError as e:
                        error_msg = f"Failed to parse JSON response: {str(e)}\nRaw response: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"Unexpected error processing response: {str(e)}\nRaw response: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                else:
                    error_msg = f"Request failed with status {response.status_code}: {response.text}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Network error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})



# Sidebar with example queries
st.sidebar.subheader("💡 Example Queries")
st.sidebar.markdown("""
- "Load and summarize train.csv"
- "Clean the train.csv dataset"
- "Train a model on train.csv"
- "Show the first 10 rows of train.csv"
- "What are the column types in train.csv?"
- "Perform feature selection on cleaned_train.csv"
""")

# Clear chat button
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()