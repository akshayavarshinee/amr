"""Unit tests for the FastAPI prediction service."""

import unittest

try:
    from fastapi.testclient import TestClient
    from amr_guard.api.main import app
    client = TestClient(app)
    TEST_CLIENT_AVAILABLE = True
except Exception:
    client = None
    TEST_CLIENT_AVAILABLE = False


class TestApiEndpoints(unittest.TestCase):

    def test_health_endpoint(self):
        if not TEST_CLIENT_AVAILABLE:
            self.skipTest("TestClient / httpx dependency not available")
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("model_loaded", data)

    def test_models_current_endpoint(self):
        if not TEST_CLIENT_AVAILABLE:
            self.skipTest("TestClient / httpx dependency not available")
        response = client.get("/models/current")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_version", data)
        self.assertIn("decision_threshold", data)

    def test_monitoring_drift_endpoint(self):
        if not TEST_CLIENT_AVAILABLE:
            self.skipTest("TestClient / httpx dependency not available")
        response = client.get("/monitoring/drift")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)


if __name__ == "__main__":
    unittest.main()
