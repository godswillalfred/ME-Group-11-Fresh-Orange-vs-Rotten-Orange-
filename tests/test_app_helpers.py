import unittest

from inference_utils import display_label, interpret_probability


class ProbabilityInterpretationTests(unittest.TestCase):
    def setUp(self):
        self.classes = ["fresh_orange", "rotten_orange"]

    def test_probability_above_threshold_is_rotten(self):
        label, confidence = interpret_probability(0.82, self.classes, 0.5)
        self.assertEqual(label, "rotten_orange")
        self.assertAlmostEqual(confidence, 0.82)

    def test_probability_below_threshold_is_fresh(self):
        label, confidence = interpret_probability(0.18, self.classes, 0.5)
        self.assertEqual(label, "fresh_orange")
        self.assertAlmostEqual(confidence, 0.82)

    def test_display_label(self):
        self.assertEqual(display_label("fresh_orange"), "Fresh Orange")


if __name__ == "__main__":
    unittest.main()
