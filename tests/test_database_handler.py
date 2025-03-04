import pytest
import sqlite3
import os
import numpy as np
from database_handler import (
    create_tables,
    add_patient,
    insert_session_data,
    update_session_embedding,
    fetch_patient_embeddings,
    fetch_session_data,
    fetch_session_topics,
    get_all_patients,
    get_patient_info
)

# Test database file name
TEST_DB = 'test_patient_sessions.db'

@pytest.fixture
def setup_database():
    """Fixture to create a test database and clean it up after tests"""
    # Setup - create a test database
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    
    # Create a connection to test database
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    
    # Create the database tables
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY,
        name TEXT,
        birthdate TEXT,
        notes TEXT
    )
    ''')

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        session_id TEXT,
        sentence TEXT,
        speaker TEXT,
        embedding BLOB,
        sentiment TEXT,
        sentiment_score REAL,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS topics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id INTEGER,
        session_id TEXT,
        topic TEXT,
        FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    # Patch the database name temporarily - use the test database for testing
    original_connect = sqlite3.connect
    
    #Patching the connect function of sqlite3 module to use the test database
    def patched_connect(database_name, *args, **kwargs):
        if database_name == 'patient_sessions.db':
            return original_connect(TEST_DB, *args, **kwargs)
        return original_connect(database_name, *args, **kwargs)
    
    # Apply the patch
    sqlite3.connect = patched_connect
    
    yield
    
    # Teardown - restore the original function and remove the test database
    sqlite3.connect = original_connect
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_add_patient(setup_database):
    """Test adding a patient to the database"""
    # Add a test patient
    add_patient(1, "Test Patient", "2000-01-01", "Test notes")
    
    # Verify patient was added correctly
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = 1")
    result = cursor.fetchone()
    conn.close()
    
    # Assert that patient data matches what was inserted
    assert result is not None
    assert result[0] == 1  # patient_id
    assert result[1] == "Test Patient"  # name
    assert result[2] == "2000-01-01"  # birthdate
    assert result[3] == "Test notes"  # notes


def test_insert_and_fetch_session_data(setup_database):
    """Test inserting session data and retrieving it"""
    # Setup - add a test patient
    add_patient(1, "Test Patient", "2000-01-01", "Test notes")
    
    # Insert test session data
    insert_session_data(1, "1", "This is a test sentence.", "patient", "Neutral", 0.0)
    
    # Fetch the session data
    session_data = fetch_session_data(1, "1")
    
    # Assert the session data was retrieved correctly
    assert len(session_data) == 1
    assert session_data[0]["sentence"] == "This is a test sentence."
    assert session_data[0]["speaker"] == "patient"
    assert session_data[0]["sentiment"] == "Neutral"
    assert session_data[0]["sentiment_score"] == 0.0


def test_embedding_update_and_retrieval(setup_database):
    """Test updating embeddings and retrieving them"""
    # Setup - add a test patient and session
    add_patient(1, "Test Patient", "2000-01-01", "Test notes")
    insert_session_data(1, "1", "This is a test sentence.", "patient", "Neutral", 0.0)
    
    # Create a test embedding
    test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Update the session with the embedding
    update_session_embedding("1", "This is a test sentence.", test_embedding)
    
    # Fetch the embeddings
    embeddings = fetch_patient_embeddings(1)
    
    # Assert embeddings were stored and retrieved correctly
    assert len(embeddings) == 1
    session_id, sentence_id, sentence, speaker, embedding = embeddings[0]
    assert session_id == "1"
    assert sentence_id == 1
    assert sentence == "This is a test sentence."
    assert speaker == "patient"
    # Convert the embedding back to a list and round to 1 decimal place for comparison
    retrieved_embedding = [round(float(value), 1) for value in embedding]
    assert retrieved_embedding == test_embedding


def test_get_all_patients(setup_database):
    """Test retrieving all patients"""
    # Add multiple test patients
    add_patient(1, "Patient 1", "2000-01-01", "Notes 1")
    add_patient(2, "Patient 2", "1995-05-05", "Notes 2")
    add_patient(3, "Patient 3", "1990-10-10", "Notes 3")
    
    # Get all patients
    patients = get_all_patients()
    
    # Assert the correct number of patients are retrieved
    assert len(patients) == 3
    
    # Assert patient data is correct
    assert patients[0]["patient_id"] == 1
    assert patients[0]["name"] == "Patient 1"
    assert patients[1]["patient_id"] == 2
    assert patients[1]["name"] == "Patient 2"
    assert patients[2]["patient_id"] == 3
    assert patients[2]["name"] == "Patient 3"


def test_fetch_session_topics(setup_database):
    """Test fetching session topics"""
    # Setup - add a test patient
    add_patient(1, "Test Patient", "2000-01-01", "Test notes")
    
    # Create connection to add test topics directly
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    
    # Insert test topics
    cursor.execute(
        "INSERT INTO topics (patient_id, session_id, topic) VALUES (?, ?, ?)",
        (1, "1", "Anxiety")
    )
    cursor.execute(
        "INSERT INTO topics (patient_id, session_id, topic) VALUES (?, ?, ?)",
        (1, "1", "Work stress")
    )
    conn.commit()
    conn.close()
    
    # Fetch the topics
    topics = fetch_session_topics(1, "1")
    
    # Assert topics were retrieved correctly
    assert len(topics) == 2
    assert "Anxiety" in topics
    assert "Work stress" in topics


def test_get_patient_info(setup_database):
    """Test retrieving detailed patient information"""
    # Setup - add a test patient
    add_patient(1, "Test Patient", "2000-01-01", "Test notes")
    
    # Add session data
    insert_session_data(1, "1", "This is sentence 1.", "patient", "Neutral", 0.0)
    insert_session_data(1, "1", "This is sentence 2.", "psychologist", "Empathy", 3.0)
    
    # Add topic
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO topics (patient_id, session_id, topic) VALUES (?, ?, ?)",
        (1, "1", "Anxiety")
    )
    conn.commit()
    conn.close()
    
    # Get patient info
    patient_info = get_patient_info(1)
    
    # Assert patient info is correct
    assert patient_info["patient_id"] == 1
    assert patient_info["name"] == "Test Patient"
    assert patient_info["birthdate"] == "2000-01-01"
    assert patient_info["notes"] == "Test notes"
    
    # Check sessions data
    assert len(patient_info["sessions"]) == 1
    session = patient_info["sessions"][0]
    assert session["session_id"] == "1"
    assert len(session["session_data"]) == 2
    assert session["session_topics"] == ["Anxiety"]

# Add a parametrized test to show more advanced testing concepts
@pytest.mark.parametrize("patient_id, name, birthdate, notes", [
    (1, "John Doe", "1990-01-01", "Regular patient"),
    (2, "Jane Smith", "1985-05-15", "New patient"),
    (3, "Bob Johnson", "1976-12-30", "")
])
def test_add_multiple_patients(setup_database, patient_id, name, birthdate, notes):
    """Test adding multiple patients with different data"""
    # Add the patient
    add_patient(patient_id, name, birthdate, notes)
    
    # Verify patient was added correctly
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,))
    result = cursor.fetchone()
    conn.close()
    
    # Assert that patient data matches what was inserted
    assert result is not None
    assert result[0] == patient_id
    assert result[1] == name
    assert result[2] == birthdate
    assert result[3] == notes