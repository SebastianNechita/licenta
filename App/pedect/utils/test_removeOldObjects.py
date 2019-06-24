from unittest import TestCase

from pedect.utils.trackedObjectsOperations import removeOldObjects, TrackedObject


class TestRemoveOldObjects(TestCase):
    def setUp(self):
        self.t = TrackedObject((1, 1, 1, 1), 1, "")
        self.t2 = TrackedObject((1, 1, 1, 1), 0, "")

    def test_BB1_WB1(self):
        self.assertRaises(AssertionError, removeOldObjects, None, 10, 10)

    def test_BB2(self):
        self.assertRaises(AssertionError, removeOldObjects, "", 10, 10)

    def test_BB3_WB5(self):
        self.assertRaises(AssertionError, removeOldObjects, {"1": "", "2": self.t}, 10, 10)

    def test_BB4_WB2(self):
        self.assertRaises(AssertionError, removeOldObjects, {}, None, 10)

    def test_BB5(self):
        self.assertRaises(AssertionError, removeOldObjects, {}, "", 10)

    def test_BB6_WB3(self):
        self.assertRaises(AssertionError, removeOldObjects, {}, 10, None)

    def test_BB7(self):
        self.assertRaises(AssertionError, removeOldObjects, {}, 10, "")

    def test_BB8(self):
        self.assertEqual(removeOldObjects({"1": self.t}, 2, 2), {"1": self.t})

    def test_BB9(self):
        self.assertEqual(removeOldObjects({"1": self.t2}, 2, 2), {})

    def test_BB10_WB6(self):
        self.assertEqual(removeOldObjects({"1": self.t, "2": self.t2}, 2, 2), {"1": self.t})

    def test_BB11_WB4(self):
        self.assertEqual(removeOldObjects({}, 1, 10), {})


