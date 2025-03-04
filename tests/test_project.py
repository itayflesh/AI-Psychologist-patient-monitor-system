import pytest
from unittest.mock import patch, MagicMock, mock_open
import json
import numpy as np
import os
from Project import (
    get_sentiment,
    determine_speaker_roles,
    detect_drastic_changes,
    identify_topic_of_change,
    remove_similar_topics
)

class TestProject:
    """Test suite for key functions in Project.py"""

    @patch('Project.client.chat.completions.create')
    def test_get_sentiment(self, mock_create):
        """Test the get_sentiment function properly parses sentiment from OpenAI response"""
        # Mock response from OpenAI
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Sentiment: Happiness, Score: 3"
        mock_create.return_value = mock_response

        # Call the function
        sentiment, score = get_sentiment("I feel great today!", is_patient=True)

        # Verify the correct parameters were passed to OpenAI
        call_args = mock_create.call_args[1]
        assert call_args['model'] == "gpt-4o-mini"
        assert len(call_args['messages']) == 2
        assert call_args['messages'][0]['role'] == "system"
        assert "patient" in call_args['messages'][1]['content']

        # Check the results
        assert sentiment == "Happiness"
        assert score == 3

    @patch('Project.client.chat.completions.create')
    def test_get_sentiment_with_unexpected_response(self, mock_create):
        """Test get_sentiment handles unexpected response formats gracefully"""
        # Mock response with unexpected format
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is an unexpected response format"
        mock_create.return_value = mock_response

        # Call the function
        sentiment, score = get_sentiment("Test text", is_patient=True)

        # Verify the function handles the error gracefully
        assert sentiment is None
        assert score == 0

    @patch('Project.client.chat.completions.create')
    def test_determine_speaker_roles(self, mock_create):
        """Test the determine_speaker_roles function correctly identifies speakers"""
        # Create a mock transcript
        transcript = MagicMock()
        transcript.utterances = [
            MagicMock(speaker="A", text="How are you feeling today?"),
            MagicMock(speaker="B", text="I've been feeling anxious lately."),
            MagicMock(speaker="A", text="Can you tell me more about that?")
        ]

        # Mock response from OpenAI
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"psychologist": "A", "patient": "B"}'
        mock_create.return_value = mock_response

        # Call the function
        result = determine_speaker_roles(transcript)

        # Verify the API was called correctly
        assert mock_create.called
        call_args = mock_create.call_args[1]
        assert call_args['model'] == "gpt-4o-mini"
        assert call_args['temperature'] == 0
        
        # Check the result
        assert result == {"psychologist": "A", "patient": "B"}

    def test_detect_drastic_changes(self):
        """Test the detect_drastic_changes function identifies significant sentiment shifts"""
        # Create test session data
        session_data = [
            {'id': 1, 'sentence': 'I am feeling okay today.', 'speaker': 'patient', 'sentiment': 'Neutral', 'sentiment_score': 0},
            {'id': 2, 'sentence': 'But I had a really tough week at work.', 'speaker': 'patient', 'sentiment': 'Anxiety', 'sentiment_score': -3},
            {'id': 3, 'sentence': 'My boss criticized me in front of everyone.', 'speaker': 'patient', 'sentiment': 'Anger', 'sentiment_score': -4},
            {'id': 4, 'sentence': 'However, I got some good news yesterday.', 'speaker': 'patient', 'sentiment': 'Hopefulness', 'sentiment_score': 2},
            {'id': 5, 'sentence': 'I got promoted!', 'speaker': 'patient', 'sentiment': 'Excitement', 'sentiment_score': 4}
        ]

        # Set threshold
        threshold = 3

        # Call the function
        result = detect_drastic_changes(session_data, threshold)

        # Check the results - expecting 2 drastic changes
        assert len(result) == 2
        
        # First drastic change: Neutral (0) to Anxiety (-3)
        assert result[0][0] == 1  # first sentence id
        assert result[0][1] == 2  # second sentence id
        assert result[0][2] == -3  # score change
        
        # Second drastic change: Anger (-4) to Hopefulness (2)
        assert result[1][0] == 3  # first sentence id
        assert result[1][1] == 4  # second sentence id
        assert result[1][2] == 6  # score change

    @patch('Project.client.chat.completions.create')
    def test_identify_topic_of_change(self, mock_create):
        """Test the identify_topic_of_change function correctly identifies topics"""
        # Mock sentences for context
        sentences = [
            "I've been thinking about my job a lot.",
            "How has that been affecting you?",
            "I've been feeling really stressed about the deadlines.",
            "That sounds challenging. Have you talked to your manager?",
            "Actually, yesterday my manager extended the deadline!",
            "That's great news. How did that make you feel?",
            "Much more relaxed now."
        ]
        
        # Mock response from OpenAI
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Work deadlines"
        mock_create.return_value = mock_response

        # Call the function
        result = identify_topic_of_change(
            sentences,
            "I've been feeling really stressed about the deadlines.",
            "Actually, yesterday my manager extended the deadline!",
            6,  # Score change from -3 to +3
            "Anxiety",
            "Relief",
            "patient"
        )

        # Verify the API was called correctly
        assert mock_create.called
        
        # Check the result
        assert result == "Work deadlines"

    @patch('Project.client.chat.completions.create')
    def test_remove_similar_topics(self, mock_create):
        """Test the remove_similar_topics function filters out similar topics"""
        # Test topics
        topics = ["Anxiety at work", "Work stress", "Family problems", "Weekend plans"]
        
        # Mock responses from OpenAI to indicate that first two topics are similar
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = "yes"  # "Work stress" similar to "Anxiety at work"
        
        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = "no"   # "Family problems" not similar to "Anxiety at work"
        
        mock_response3 = MagicMock()
        mock_response3.choices = [MagicMock()]
        mock_response3.choices[0].message.content = "no"   # "Weekend plans" not similar to "Anxiety at work"
        
        mock_response4 = MagicMock()
        mock_response4.choices = [MagicMock()]
        mock_response4.choices[0].message.content = "no"   # "Weekend plans" not similar to "Family problems"
        
        # Set up the side effects to return different responses for each call
        mock_create.side_effect = [mock_response1, mock_response2, mock_response3, mock_response4]
        
        # Call the function
        result = remove_similar_topics(topics)
        
        # Check that we filtered out "Work stress" as it's similar to "Anxiety at work"
        assert len(result) == 3
        assert "Anxiety at work" in result
        assert "Family problems" in result
        assert "Weekend plans" in result
        assert "Work stress" not in result