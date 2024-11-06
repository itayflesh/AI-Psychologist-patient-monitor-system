import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from streamlit_option_menu import option_menu

# Placeholder for functions to handle database interactions and data processing
from database_handler import get_patient_info, get_all_patients, add_patient, initialize_database , fetch_session_data
from embedding_handler import search_similar_sentences
from Project import process_audio_file, generate_sentiment_graph, detect_drastic_changes, identify_topic_of_change, identify_topics_to_revisit

# Initialize session state variables
if 'search_results' not in st.session_state:
    st.session_state.search_results = []
if 'show_transcript' not in st.session_state:
    st.session_state.show_transcript = {}

# Sentiment score dictionaries
patient_sentiment_scores = {
    "Despair": -5, "Anger": -4, "Anxiety": -3, "Sadness": -2, "Discomfort": -1,
    "Natural": 0, "Contentment": 1, "Hopefulness": 2, "Happiness": 3, "Excitement": 4, "Euphoria": 5
}

psychologist_sentiment_scores = {
    "Overwhelm": -5, "Helplessness": -4, "Sadness": -3, "Frustration": -2, "Concern": -1,
    "Natural": 0, "Contentment": 1, "Encouragement": 2, "Empathy": 3, "Optimism": 4, "Fulfillment": 5
}

# Initialize the database
initialize_database()

# Set page config to wide mode and add custom CSS
st.set_page_config(layout="wide", page_title="Psychological Sessions Monitor")
st.markdown("""
<style>
    .sidebar .sidebar-content {
        width: 300px;
    }
    .sidebar-text {
        font-size: 16px;
    }
    .stRadio > div {
        display: flex;
        flex-direction: column;
        align-items: stretch;
    }
    .stRadio > div > label {
        padding: 10px;
        background-color: #f0f2f6;
        border-radius: 5px;
        margin-bottom: 5px;
        cursor: pointer;
    }
    .stRadio > div > label:hover {
        background-color: #e0e2e6;
    }
    .big-font {
        font-size: 24px !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to create the Add New Patient form
def add_new_patient_page():
    st.title("Add New Patient")
    
    new_patient_name = st.text_input("Enter patient name:")
    new_patient_birthdate = st.date_input("Enter birthdate:")
    new_patient_notes = st.text_area("Enter notes:")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Create Patient"):
            if new_patient_name and new_patient_birthdate:
                patients = get_all_patients()
                if len(patients) == 0:
                    new_patient_id = 1
                else:
                    new_patient_id = max(patient['patient_id'] for patient in patients) + 1

                birthdate_str = new_patient_birthdate.strftime("%Y-%m-%d")
                add_patient(new_patient_id, new_patient_name, birthdate_str, new_patient_notes)
                st.success(f"Patient '{new_patient_name}' added successfully!")
                st.session_state.add_patient_mode = False
                st.rerun()
            else:
                st.error("Please enter at least the patient name and birthdate.")
    
    with col2:
        if st.button("Cancel"):
            st.session_state.add_patient_mode = False
            st.rerun()

def show_transcript_section(patient_id, session_id, sentence_number):
    """
    Fetches and displays a section of the transcript around the given sentence number.
    """
    # Fetch the entire session data
    session_data = fetch_session_data(patient_id, session_id)
    
    # Determine the range of sentences to display (5 before and 5 after)
    start_index = max(0, sentence_number - 6)
    end_index = min(len(session_data), sentence_number + 5)
    
    # Create an expander for the transcript section
    with st.expander(f"Transcript Section (Session {session_id})", expanded=True):
        for i in range(start_index, end_index):
            entry = session_data[i]
            if entry['speaker'].lower() == "patient":
                speaker_display = "Patient"
                sentence_color = "#a455e0"
            else:
                speaker_display = "Psychologist"
                sentence_color = "#1263e6"
            
            # Highlight the searched sentence
            if i == sentence_number - 1:
                st.markdown(f"<span style='color: {sentence_color}; background-color: yellow;'><strong>{i+1}. {speaker_display}:</strong> {entry['sentence']}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color: {sentence_color};'><strong>{i+1}. {speaker_display}:</strong> {entry['sentence']}</span>", unsafe_allow_html=True)

def toggle_transcript(key):
    st.session_state[key] = not st.session_state[key]
    
# Function to display the home page
def home_page():
    st.title("Welcome to the Psychological Sessions Monitor")
    st.markdown("### Your comprehensive tool for managing and analyzing psychological sessions")
    
    st.markdown("#### Key Features:")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- 📊 **Session Analysis**: View detailed transcripts and sentiment analysis")
        st.markdown("- 🔍 **Smart Search**: Find relevant information across all sessions")
    with col2:
        st.markdown("- 👤 **Patient Management**: Easily add and manage patient profiles")
        st.markdown("- 📈 **Emotional Trends**: Track emotional changes over time")
    
    st.markdown("#### Getting Started:")
    st.markdown("1. Use the sidebar to select an existing patient or add a new one")
    st.markdown("2. Navigate through different sections using the menu")
    st.markdown("3. Upload new session recordings or analyze existing ones")
    
    

    st.markdown("---")
    st.markdown("Ready to begin? Select a patient from the sidebar or add a new one to get started!")

# Main application logic
def main():
    # Initialize session state
    if 'add_patient_mode' not in st.session_state:
        st.session_state.add_patient_mode = False
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Home"

    # Sidebar
    with st.sidebar:
        st.title("Psychological Sessions Monitor")

        # Home button
        if st.button("🏠 Home"):
            st.session_state.current_page = "Home"
            st.session_state.add_patient_mode = False
            st.rerun()

        if not st.session_state.add_patient_mode:
            # Patient selection
            patients = get_all_patients()
            selected_patient = st.selectbox("Select a patient", ["Select a patient"] + patients, format_func=lambda x: x['name'] if isinstance(x, dict) else x)

            if selected_patient and selected_patient != "Select a patient":
                # Navigation menu
                selected_option = option_menu("Menu", ["Information", "Search", "Transcripts", "Add New Session"],
                                              icons=['info-circle', 'search', 'file-text', 'plus-circle'],
                                              menu_icon="list", default_index=0)
                st.session_state.current_page = selected_option

        # Add New Patient button
        if st.button("➕ Add New Patient"):
            st.session_state.add_patient_mode = True
            st.session_state.current_page = "Add New Patient"
            st.rerun()

    # Main content area
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.add_patient_mode:
        add_new_patient_page()
    elif selected_patient and selected_patient != "Select a patient":
        patient_info = get_patient_info(selected_patient['patient_id'])

        if st.session_state.current_page == "Information":
            st.title(f"Patient Information: {selected_patient['name']}")

            # Display patient info
            st.markdown(f"**Patient ID:** {patient_info['patient_id']}")
            st.markdown(f"**Name:** {patient_info['name']}")
            st.markdown(f"**Birthdate:** {patient_info['birthdate']}")
            st.markdown(f"**Notes:** {patient_info['notes']}")

            # Display number of sessions
            num_sessions = len(patient_info['sessions'])
            st.markdown(f"**Number of Sessions:** {num_sessions}")
            
            # Fetch topics to revisit using the identify_topics_to_revisit function
            topics_to_revisit = identify_topics_to_revisit(selected_patient['patient_id'])

            # Display topics to revisit under a clear and descriptive title
            st.subheader("Potential Topics to Revisit")
            if topics_to_revisit:
                for topic in topics_to_revisit:
                    st.write(f"{topic}")
            else:
                st.write("No specific topics identified for revisiting at this time.")

        elif st.session_state.current_page == "Search":
            st.title(f"Search: {selected_patient['name']}")

            search_query = st.text_input("Type here to search:")

            # Generate session options dynamically
            session_options = ['All sessions'] + [f"Session {session['session_id']}" for session in patient_info['sessions']]
            sessions_to_search = st.multiselect("Select sessions to search", options=session_options, default='All sessions')

            if st.button("Search"):
                # Convert session options into actual session IDs for searching
                if 'All sessions' in sessions_to_search or not sessions_to_search:
                    selected_session_ids = [session['session_id'] for session in patient_info['sessions']]
                else:
                    selected_session_ids = [option.split()[1] for option in sessions_to_search]

                st.session_state.search_results = search_similar_sentences(selected_patient['patient_id'], search_query, selected_session_ids)
                
                # Initialize show_transcript state for new search results
                for result in st.session_state.search_results:
                    key = f"show_transcript_{result['session_id']}_{result['sentence_number']}"
                    if key not in st.session_state:
                        st.session_state[key] = False

            if st.session_state.search_results:
                st.markdown("<span style='font-size:20px; font-weight:bold; text-decoration:underline;'>Search Results:</span>", unsafe_allow_html=True)
                for i, result in enumerate(st.session_state.search_results):
                    # Determine the speaker and sentence color based on the speaker
                    if result['speaker'].lower() == 'patient':
                        speaker_name = patient_info['name']
                        sentence_color = '#a455e0'
                    elif result['speaker'].lower() == 'psychologist':
                        speaker_name = 'you'
                        sentence_color = '#1295e6'
                    else:
                        speaker_name = result['speaker']
                        sentence_color = 'black'
                    
                    # Create a formatted string for the session, sentence number, and speaker
                    session_info = f"**<span style='font-size:18px;'>Session {result['session_id']} | Sentence {result['sentence_number']} | {speaker_name}:</span>**"
                    
                    # Display the session info in bold and larger font size, and the sentence in the desired color
                    st.markdown(session_info, unsafe_allow_html=True)
                    st.markdown(f"<span style='font-size:16px; color:{sentence_color};'>{result['sentence']}</span>", unsafe_allow_html=True)
                    
                    # Create a unique key for each show_transcript state
                    transcript_key = f"show_transcript_{result['session_id']}_{result['sentence_number']}"
                    
                    # Add a button to show/hide the transcript section
                    button_text = "Hide Transcript" if st.session_state[transcript_key] else "Show Transcript"
                    st.button(
                        f"{button_text} (Session {result['session_id']})",
                        key=f"btn_{i}",
                        on_click=toggle_transcript,
                        args=(transcript_key,)
                    )
                    
                    # Show transcript if state is True
                    if st.session_state[transcript_key]:
                        show_transcript_section(selected_patient['patient_id'], result['session_id'], result['sentence_number'])
            elif st.session_state.search_results == []:
                st.write("No results found.")


        elif st.session_state.current_page == "Transcripts":
            st.title(f"Session Transcript: {selected_patient['name']}")

            # Session selection
            sessions = [f"Session {session['session_id']}" for session in patient_info['sessions']]
            selected_session = st.selectbox("Select a session", sessions, key="session_selector")

            if selected_session:
                # Get the index of the selected session
                selected_index = sessions.index(selected_session)

                # Display the selected session
                session = patient_info['sessions'][selected_index]
                st.subheader(f"Transcript for {selected_session}")

                # Display the session transcript line by line with color coding
                for i, entry in enumerate(session['session_data'], start=1):
                    # Check if the speaker is the patient or psychologist
                    if entry['speaker'].lower() == "patient":
                        speaker_display = selected_patient['name']  # Use patient's name
                        sentence_color = "#a455e0"  # Purple color for patient's sentences
                    else:  # Psychologist
                        speaker_display = "you"
                        sentence_color = "#1263e6"  # Blue color for psychologist's sentences

                    # Display each line with the speaker's name and the sentence in the appropriate color
                    st.markdown(
                        f"<span style='color: {sentence_color};'><strong>{i}. {speaker_display}:</strong> {entry['sentence']}</span>",
                        unsafe_allow_html=True
                    )
                # Display patient's sentiment graph
                st.subheader("Patient's Sentiment Analysis")
                patient_data = [entry for entry in session['session_data'] if entry['speaker'].lower() == 'patient']
                sentiment_graph_patient = generate_sentiment_graph(patient_data, f"Patient Sentiment Analysis - {selected_session}", patient_sentiment_scores, "patient")
                st.plotly_chart(sentiment_graph_patient)

                # Display emotional changes analysis
                st.subheader("Topics Which Caused Emotional Changes During This Session")
                
                # Get topics directly from session info and clean them
                topics = [topic.replace("Topic: ", "") for topic in session['session_topics']]

                if topics:
                    for topic in topics:
                        st.write(f"- {topic}")
                else:
                    st.write("No drastic emotional changes detected in this session.")

                
                # Display psychologist's sentiment graph
                st.subheader("Psychologist's Sentiment Analysis")
                sentiment_graph_Psychologist = generate_sentiment_graph(session['session_data'], f"Psychologist Sentiment Analysis - {selected_session}", psychologist_sentiment_scores, "psychologist")
                st.plotly_chart(sentiment_graph_Psychologist)
            else:
                st.write("No sessions available for this patient.")

        elif st.session_state.current_page == "Add New Session":
            st.title(f"Add New Session: {selected_patient['name']}")
            audio_file = st.file_uploader("Upload session audio file", type=["wav", "mp3", "mp4"])

            if st.button("Upload"):
                if audio_file is not None:
                    with st.spinner('Processing audio file... This may take a few minutes.'):
                        process_audio_file(selected_patient['patient_id'], audio_file)
                    st.success("Session added successfully!")
                else:
                    st.error("Please upload an audio file before clicking 'Upload'.")
    else:
        home_page()

if __name__ == "__main__":
    main()