"""
test_milan2026_search_agent.py
────────────────────────────────
Unit tests for the Milan 2026 Olympics RSS feed search agent.
Tests RSS parsing, Olympic content filtering, and deduplication logic.
"""

import unittest
from unittest.mock import patch, MagicMock
import sys
import os

# Import the search agent module
# Adjust the import based on your actual module structure
try:
    from milan2026_search_agent__2_ import (
        deterministic_id,
        truncate,
        strip_html,
        filter_olympic_content,
        OLYMPIC_KEYWORDS,
        MAX_WORDS,
        MIN_WORDS,
    )
except ImportError:
    # Fallback if file has different name
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "search_agent", 
        os.path.join(os.path.dirname(__file__), "milan2026_search_agent__2_.py")
    )
    search_agent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(search_agent)
    
    deterministic_id = search_agent.deterministic_id
    truncate = search_agent.truncate
    strip_html = search_agent.strip_html
    filter_olympic_content = search_agent.filter_olympic_content
    OLYMPIC_KEYWORDS = search_agent.OLYMPIC_KEYWORDS
    MAX_WORDS = search_agent.MAX_WORDS
    MIN_WORDS = search_agent.MIN_WORDS


class TestHelperFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_deterministic_id(self):
        """Test that IDs are consistent and unique."""
        id1 = deterministic_id("source1", "unique_field_1")
        id2 = deterministic_id("source1", "unique_field_1")
        id3 = deterministic_id("source1", "unique_field_2")
        
        # Same input -> same ID
        self.assertEqual(id1, id2)
        # Different input -> different ID
        self.assertNotEqual(id1, id3)
        # ID is 32 chars hex
        self.assertEqual(len(id1), 32)
        self.assertTrue(all(c in "0123456789abcdef" for c in id1))
    
    def test_truncate(self):
        """Test text truncation."""
        short_text = "This is short."
        long_text = " ".join(["word"] * 400)
        
        # Short text unchanged
        self.assertEqual(truncate(short_text, 100), short_text)
        
        # Long text truncated to MAX_WORDS
        truncated = truncate(long_text)
        self.assertEqual(len(truncated.split()), MAX_WORDS)
        
        # Custom word limit
        truncated_50 = truncate(long_text, 50)
        self.assertEqual(len(truncated_50.split()), 50)
    
    def test_strip_html(self):
        """Test HTML tag removal."""
        html_text = "This has <b>bold</b> and <a href='url'>link</a> tags."
        clean = strip_html(html_text)
        
        self.assertNotIn("<b>", clean)
        self.assertNotIn("</b>", clean)
        self.assertNotIn("<a", clean)
        self.assertIn("bold", clean)
        self.assertIn("link", clean)


class TestOlympicContentFilter(unittest.TestCase):
    """Test Olympic content filtering."""
    
    def test_filter_keeps_olympic_content(self):
        """Test that Olympic-related content is kept."""
        chunks = [
            {
                "id": "1",
                "text": "Milan 2026 Winter Olympics figure skating preview",
                "source_key": "test",
                "url": "http://example.com/1"
            },
            {
                "id": "2",
                "text": "Alpine skiing gold medal race heats up",
                "source_key": "test",
                "url": "http://example.com/2"
            },
            {
                "id": "3",
                "text": "NBA playoffs predictions",
                "source_key": "test",
                "url": "http://example.com/3"
            }
        ]
        
        filtered = filter_olympic_content(chunks)
        
        # Should keep Olympic content, filter out NBA
        self.assertEqual(len(filtered), 2)
        self.assertTrue(any("Milan 2026" in c["text"] for c in filtered))
        self.assertTrue(any("Alpine skiing" in c["text"] for c in filtered))
        self.assertFalse(any("NBA" in c["text"] for c in filtered))
    
    def test_filter_empty_list(self):
        """Test filter handles empty input."""
        filtered = filter_olympic_content([])
        self.assertEqual(filtered, [])
    
    def test_filter_no_matches(self):
        """Test filter when no Olympic content."""
        chunks = [
            {
                "id": "1",
                "text": "NFL draft analysis",
                "source_key": "test",
                "url": "http://example.com/1"
            }
        ]
        
        filtered = filter_olympic_content(chunks)
        self.assertEqual(len(filtered), 0)
    
    def test_olympic_keywords_comprehensive(self):
        """Test that key Olympic keywords are present."""
        # Core event keywords
        self.assertIn("milano cortina", OLYMPIC_KEYWORDS)
        self.assertIn("milan 2026", OLYMPIC_KEYWORDS)
        self.assertIn("winter olympics", OLYMPIC_KEYWORDS)
        
        # Winter sports
        self.assertIn("figure skating", OLYMPIC_KEYWORDS)
        self.assertIn("alpine skiing", OLYMPIC_KEYWORDS)
        self.assertIn("ice hockey", OLYMPIC_KEYWORDS)
        self.assertIn("curling", OLYMPIC_KEYWORDS)
        
        # Medal terms
        self.assertIn("gold medal", OLYMPIC_KEYWORDS)
        self.assertIn("olympic champion", OLYMPIC_KEYWORDS)


class TestSearchAgentIntegration(unittest.TestCase):
    """Integration tests for the full search agent flow."""
    
    @patch.dict(os.environ, {"PINECONE_API_KEY": "test-key-12345"})
    def test_environment_variable_check(self):
        """Test that API key environment variable is read."""
        api_key = os.getenv("PINECONE_API_KEY")
        self.assertEqual(api_key, "test-key-12345")
    
    def test_winter_olympics_only_filter(self):
        """Test that summer sports are filtered out."""
        summer_chunks = [
            {
                "id": "1",
                "text": "Paris 2024 swimming finals feature Katie Ledecky",
                "source_key": "test",
                "url": "http://example.com/1"
            },
            {
                "id": "2",
                "text": "Track and field records broken at Olympics",
                "source_key": "test",
                "url": "http://example.com/2"
            }
        ]
        
        winter_chunks = [
            {
                "id": "3",
                "text": "Milan 2026 ice hockey tournament draw announced",
                "source_key": "test",
                "url": "http://example.com/3"
            }
        ]
        
        # Summer content should be filtered (no winter keywords)
        filtered_summer = filter_olympic_content(summer_chunks)
        self.assertEqual(len(filtered_summer), 0)
        
        # Winter content should pass
        filtered_winter = filter_olympic_content(winter_chunks)
        self.assertEqual(len(filtered_winter), 1)


class TestConfiguration(unittest.TestCase):
    """Test configuration constants."""
    
    def test_word_limits_sensible(self):
        """Test that word limits are reasonable."""
        self.assertGreater(MAX_WORDS, MIN_WORDS)
        self.assertGreater(MIN_WORDS, 0)
        self.assertLess(MAX_WORDS, 1000)  # Not too large
    
    def test_namespace_correct(self):
        """Test that namespace is set correctly."""
        # Import namespace from module
        try:
            from milan2026_search_agent__2_ import NAMESPACE
        except ImportError:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "search_agent", 
                os.path.join(os.path.dirname(__file__), "milan2026_search_agent__2_.py")
            )
            search_agent = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(search_agent)
            NAMESPACE = search_agent.NAMESPACE
        
        self.assertEqual(NAMESPACE, "narratives")


if __name__ == "__main__":
    unittest.main()
