import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from embedding_handler import (
    generate_embedding,
    generate_query_embedding,
    search_similar_sentences
)

class TestEmbeddingHandler:
    """Test suite for the embedding_handler module"""

    @patch('embedding_handler.client.embeddings.create')
    def test_generate_embedding(self, mock_create):
        """Test the generate_embedding function with a mocked OpenAI response"""
        # Set up mock return value
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=mock_embedding)]
        mock_create.return_value = mock_response

        # Call the function
        test_text = "This is a test sentence."
        result = generate_embedding(test_text)

        # Verify the OpenAI API was called with the right parameters
        mock_create.assert_called_once_with(
            model="text-embedding-3-small",
            input=test_text
        )

        # Verify the result matches our mock embedding
        assert result == mock_embedding

    @patch('embedding_handler.generate_embedding')
    def test_generate_query_embedding(self, mock_generate_embedding):
        """Test that generate_query_embedding calls generate_embedding with the right parameters"""
        # Set up mock return value
        mock_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_generate_embedding.return_value = mock_embedding

        # Call the function
        test_query = "search query text"
        result = generate_query_embedding(test_query)

        # Verify generate_embedding was called with the right parameters
        mock_generate_embedding.assert_called_once_with(test_query)

        # Verify the result matches our mock embedding
        assert result == mock_embedding

    @patch('embedding_handler.generate_query_embedding')
    @patch('embedding_handler.fetch_patient_embeddings')
    @patch('embedding_handler.cosine_similarity')
    def test_search_similar_sentences(self, mock_cosine_similarity, mock_fetch_embeddings, mock_query_embedding):
        """Test the search_similar_sentences function"""
        # Set up mocks
        query = "test query"
        patient_id = 1
        sessions_to_search = ["1", "2"]
        
        # Mock query embedding
        mock_query_vector = np.array([0.1, 0.2, 0.3])
        mock_query_embedding.return_value = mock_query_vector
        
        # Mock patient embeddings - creating a list of tuples as returned by fetch_patient_embeddings
        # (session_id, sentence_id, sentence, speaker, embedding)
        mock_patient_data = [
            ("1", 1, "First test sentence with enough words", "patient", np.array([0.2, 0.3, 0.4])),
            ("1", 2, "Second test sentence with enough words", "psychologist", np.array([0.3, 0.4, 0.5])),
            ("2", 1, "Third test sentence with enough words", "patient", np.array([0.4, 0.5, 0.6])),
            ("3", 1, "Sentence from session not in search scope", "patient", np.array([0.5, 0.6, 0.7]))
        ]
        mock_fetch_embeddings.return_value = mock_patient_data
        
        # Mock cosine similarity scores
        # Needs to return a 2D array where [0][0] is the similarity score
        mock_cosine_similarity.side_effect = [
            np.array([[0.7]]),  # First sentence
            np.array([[0.8]]),  # Second sentence
            np.array([[0.9]]),  # Third sentence
            np.array([[0.6]])   # Fourth sentence (should be ignored as not in sessions_to_search)
        ]
        
        # Call the function
        results = search_similar_sentences(patient_id, query, sessions_to_search)
        
        # Verify the mocks were called correctly
        mock_query_embedding.assert_called_once_with(query)
        mock_fetch_embeddings.assert_called_once_with(patient_id)
        
        # Verify the right number of cosine_similarity calls (3 calls for sessions 1 and 2)
        assert mock_cosine_similarity.call_count == 3
        
        # Verify the results
        assert len(results) == 3  # Should return top 5 but we only have 3 in our test data
        
        # Results should be sorted by similarity score (descending)
        assert results[0]['similarity_score'] == 0.9
        assert results[0]['sentence'] == "Third test sentence with enough words"
        assert results[0]['session_id'] == "2"
        
        assert results[1]['similarity_score'] == 0.8
        assert results[1]['sentence'] == "Second test sentence with enough words"
        assert results[1]['session_id'] == "1"
        
        assert results[2]['similarity_score'] == 0.7
        assert results[2]['sentence'] == "First test sentence with enough words"
        assert results[2]['session_id'] == "1"
        
        # Make sure the session not in sessions_to_search is not included
        for result in results:
            assert result['session_id'] in sessions_to_search

    @patch('embedding_handler.generate_query_embedding')
    @patch('embedding_handler.fetch_patient_embeddings')
    def test_search_similar_sentences_short_sentences(self, mock_fetch_embeddings, mock_query_embedding):
        """Test that search_similar_sentences filters out sentences with fewer than 5 words"""
        # Set up mocks
        query = "test query"
        patient_id = 1
        sessions_to_search = ["1"]
        
        # Mock query embedding
        mock_query_vector = np.array([0.1, 0.2, 0.3])
        mock_query_embedding.return_value = mock_query_vector
        
        # Mock patient embeddings with a mix of short and longer sentences
        mock_patient_data = [
            ("1", 1, "Short", "patient", np.array([0.2, 0.3, 0.4])),
            ("1", 2, "Also very short", "patient", np.array([0.3, 0.4, 0.5])),
            ("1", 3, "This sentence has enough words", "patient", np.array([0.4, 0.5, 0.6])),
            ("1", 4, "This one too has enough words", "patient", np.array([0.5, 0.6, 0.7]))
        ]
        mock_fetch_embeddings.return_value = mock_patient_data
        
        # Call the function
        results = search_similar_sentences(patient_id, query, sessions_to_search)
        
        # Verify only the longer sentences were included
        assert len(results) == 2
        assert "This sentence has enough words" in [r['sentence'] for r in results]
        assert "This one too has enough words" in [r['sentence'] for r in results]
        assert "Short" not in [r['sentence'] for r in results]
        assert "Also very short" not in [r['sentence'] for r in results]