import unittest

from rules.teacher_ollama import extract_json_payload


class TeacherOllamaIngestionTest(unittest.TestCase):
    def test_extract_json_payload_plain_json(self):
        payload = extract_json_payload('{"rules":[{"raw_rule":"if A then B","score":0.9}]}')
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["rules"][0]["raw_rule"], "if A then B")

    def test_extract_json_payload_markdown_fence(self):
        text = """```json
{"facts":[{"fact":{"kind":"Inventory","id":"u1","owner":"alice","item":"apple","qty":2,"phase":"post"},"score":0.8}]}
```"""
        payload = extract_json_payload(text)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["facts"][0]["fact"]["owner"], "alice")


if __name__ == "__main__":
    unittest.main()
