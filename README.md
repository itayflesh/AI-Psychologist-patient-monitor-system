# Patient AI - Monitor System for Psychologists 

## Description
The Patient Monitor System is an innovative AI-powered platform designed to enhance psychological therapy sessions through advanced analytics and natural language processing. This system transforms traditional therapy session recordings into actionable insights, helping psychologists better understand patient progress and emotional patterns.

🎥 [Watch Demo Video](https://drive.google.com/file/d/17XpYFbPvRPJ2uoOnV7O0R-yd9wMt-Rdz/view?usp=share_link)

### Key Features
- Automated transcription of therapy sessions with speaker identification
- Real-time sentiment analysis of patient and therapist interactions
- Detection and visualization of emotional changes during sessions
- Semantic search capabilities across all session transcripts
- Automated topic identification and tracking
- Structured database storage for easy data retrieval and analysis
- Comprehensive test suite with pytest for quality assurance

## Architecture

The system follows a modular architecture with these key components:

1. **Core Processing Module** (`Project.py`): Central module that handles audio processing, sentiment analysis, and topic detection
2. **Database Handler** (`database_handler.py`): Manages all database operations with SQLite
3. **Embedding Handler** (`embedding_handler.py`): Handles vector embeddings generation and semantic search
4. **Web Interface** (`webpage_handler.py`): Creates the Streamlit-based user interface
5. **Test Suite** (`tests/`): Contains comprehensive unit tests for all system components

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-Psychologist-patient-monitor-system.git
cd patient-monitor-system
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up API keys:
- Create an account at [OpenAI](https://openai.com) and [AssemblyAI](https://www.assemblyai.com)
- Create a `.env` file in the root directory with your API keys:
```
ASSEMBLY_API_KEY=your_assemblyai_api_key
OPENAI_API_KEY=your_openai_api_key
```

## How It Works

1. **Speech-to-Text Processing**
   - Audio recordings are processed using AssemblyAI's API
   - Advanced speaker diarization separates patient and therapist voices
   - Timestamps and speaker labels are preserved for accurate analysis

2. **Sentiment Analysis & Topic Detection**
   - Each utterance is analyzed using OpenAI's GPT models
   - Sentiment scores are assigned on a scale from -10 to 10
   - Significant emotional changes are automatically detected
   - Key discussion topics are identified and categorized

3. **Semantic Search**
   - Vector embeddings are generated for each sentence using OpenAI's GPT models
   - Cosine similarity is used for context-aware searching
   - Search results include surrounding conversation context

4. **Data Visualization**
   - Interactive sentiment graphs show emotional patterns
   - Topic tracking across multiple sessions
   - Easy-to-read transcripts with speaker identification

## Running the Project

1. Launch the Streamlit interface:
```bash
streamlit run webpage_handler.py
```

3. Using the interface:
   - Add new patients through the sidebar
   - Upload session recordings in WAV, MP3, or MP4 format
   - View transcripts and analysis in real-time
   - Search across sessions using natural language queries
   - Track emotional patterns and significant changes

## Quality Assurance

The project includes a comprehensive test suite to ensure reliability and correctness:

### Running Tests
```bash
# Run all tests
python -m pytest tests/
```

### Test Structure
- **Database Tests** (`test_database_handler.py`): Verify database operations and data integrity
- **Embedding Tests** (`test_embedding_handler.py`): Ensure correct embedding generation and search functionality
- **Core Functionality Tests** (`test_project.py`): Test sentiment analysis, speaker identification, and topic detection

### CI/CD Integration
The project includes a GitHub Actions workflow (`.github/workflows/ci.yml`) that:
- Runs on every push to the main branch
- Sets up the Python environment
- Installs dependencies
- Creates a test environment with dummy API keys
- Runs the full test suite

## Technologies Used

### Core Technologies
- **Python 3.11+**: Primary programming language
- **OpenAI API**: Powers sentiment analysis and embedding generation
- **AssemblyAI API**: Provides accurate speech-to-text conversion
- **Streamlit**: Creates the interactive web interface

### Libraries and Frameworks
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive data visualization
- **SQLite3**: Local database management
- **scikit-learn**: Machine learning utilities
- **pytest**: Test framework for automated testing

### APIs and Models
- OpenAI's GPT models for natural language processing
- OpenAI's text-embedding-3-small for vector embeddings
- AssemblyAI's speech recognition API with speaker diarization

This project was developed in collaboration with my partner. You can find the complete project history and commits at:
[Original Project Repository](https://github.com/NoyaArav/Final_Project-From-Idea-To-Reality)