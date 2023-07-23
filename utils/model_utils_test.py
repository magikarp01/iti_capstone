import unittest
from transformers import AutoModel, AutoTokenizer
from model_utils import llama2_7b, vicuna_7b, llama2_7b_chat
import torch

class testModelUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = llama2_7b_chat() # MODIFY
        cls.input_string = "The planet earth" # MODIFY

    def test_model_loads(self):
        """Test that model loads correctly"""
        self.assertIsNotNone(self.model)

    def test_tokenizer_loads(self):
        """Test that tokenizer loads correctly"""
        self.assertIsNotNone(self.model.tokenizer)

    def test_encode_input(self):
        """Test encoding an input string"""
        input_ids = self.model.tokenizer.encode(self.input_string, return_tensors="pt")
        self.assertIsInstance(input_ids, torch.Tensor)

    def test_generate_via_string(self):
        """Test passing input string and getting output"""
        output = self.model.generate(self.input_string, max_new_tokens = 500)
        print(output)
        self.assertIsInstance(output, str)

    def test_generate_via_tokens(self):
        """Test passing input tokens and getting output"""
        input_ids = self.model.tokenizer.encode(self.input_string, return_tensors="pt")
        output = self.model.generate(input_ids, max_new_tokens = 500)
        generated_text = self.model.tokenizer.decode(output[0], skip_special_tokens=True)
        print(generated_text)
        self.assertIsInstance(generated_text, str)

if __name__ == '__main__':
    unittest.main()