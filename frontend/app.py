import streamlit as st
import requests
import json

st.title('🎯 SHL Assessment Recommender')

# Create a form to enable Enter key
with st.form('recommendation_form'):
    query = st.text_input('Enter your job description or requirements:', 
                         placeholder='e.g., Java developer with collaboration skills')
    
    # Submit button (will trigger on Enter)
    submitted = st.form_submit_button('Get Recommendations')

if submitted and query:
    try:
        # Call our API
        response = requests.post('http://127.0.0.1:8000/recommend', 
                               json={'query': query})
        
        if response.status_code == 200:
            data = response.json()
            recommendations = data['recommended_assessments']  # Fix: access the correct key
            
            st.success(f'Found {len(recommendations)} recommendations:')
            
            for i, rec in enumerate(recommendations, 1):
                rec_name = rec['name']
                with st.expander(f'{i}. {rec_name}'):
                    st.write(f'**Description:** {rec["description"]}')
                    st.write(f'**Duration:** {rec["duration"]} minutes')
                    st.write(f'**Test Type:** {rec["test_type"]}')
                    st.write(f'**Remote Support:** {rec["remote_support"]}')
                    st.write(f'**Adaptive Support:** {rec["adaptive_support"]}')
                    st.write(f'**URL:** {rec["url"]}')
        else:
            st.error('Error getting recommendations')
    except Exception as e:
        st.error(f'Could not connect to API. Make sure it is running. Error: {str(e)}')
elif submitted:
    st.warning('Please enter a query')
