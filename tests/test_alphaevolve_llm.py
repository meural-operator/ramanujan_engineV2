import unittest
from unittest.mock import patch, MagicMock
from modules.continued_fractions.math_ai.llm.llm_client import LMStudioClient, random_mutation

class TestLMStudioClient(unittest.TestCase):
    
    def setUp(self):
        self.client = LMStudioClient(base_url="http://dummy:1234")
        
    @patch('urllib.request.urlopen')
    def test_is_available_true(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"data": [{"id": "model-id"}]}'
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        self.assertTrue(self.client.is_available())
        
    @patch('urllib.request.urlopen')
    def test_is_available_false(self, mock_urlopen):
        mock_urlopen.side_effect = Exception("Connection refused")
        
        self.assertFalse(self.client.is_available())
        
    @patch('urllib.request.urlopen')
    def test_propose_mutation_parsing(self, mock_urlopen):
        # Setup mock to return a properly formatted LM Studio / OpenAI response
        mock_response = MagicMock()
        mock_response.read.return_value = b'''
        {
            "choices": [{
                "message": {
                    "content": "Here is the mutated program:\\nlambda n: 2*n + 5\\nlambda n: -(n**2)"
                }
            }]
        }
        '''
        mock_urlopen.return_value.__enter__.return_value = mock_response
        
        a_n, b_n = self.client.propose_mutation(
            "lambda n: 2*n + 1", "lambda n: -n", "pi", 1.5
        )
        
        self.assertEqual(a_n, "lambda n: 2*n + 5")
        self.assertEqual(b_n, "lambda n: -(n**2)")

    def test_random_mutation_fallback(self):
        a_n = "lambda n: n**2"
        b_n = "lambda n: n"
        
        mut_a, mut_b = random_mutation(a_n, b_n)
        
        self.assertTrue(mut_a.startswith("lambda n:"))
        self.assertTrue(mut_b.startswith("lambda n:"))
        self.assertNotEqual((a_n, b_n), (mut_a, mut_b))


if __name__ == '__main__':
    unittest.main()
