import unittest
import os
import sys


class TestSanity(unittest.TestCase):
    def test_imports(self):
        """Test that core modules can be imported"""
        # Verify that the agents module can be imported
        import agents
        self.assertIsNotNone(agents, "agents module imported successfully")

    def test_basic_operations(self):
        """Test that basic Python operations work"""
        # Simple arithmetic to verify environment is working
        self.assertEqual(1 + 1, 2)
        self.assertTrue(True)

    def test_environment(self):
        """Test that environment variables are accessible"""
        # Just check that we can access environment variables
        self.assertIsNotNone(os.environ)
        self.assertIsNotNone(sys.version)


if __name__ == '__main__':
    unittest.main()
""
