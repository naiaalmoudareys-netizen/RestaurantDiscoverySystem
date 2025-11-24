"""
Complete System Test: All Tasks (1.1, 1.2, 2.1, 3.1)
Tests the entire system including API endpoints
"""

import requests
import json
import time
import sys
from typing import List, Tuple

BASE_URL = "http://localhost:8000"

# Track test results
test_results: List[Tuple[str, bool, str]] = []  # (test_name, passed, message)

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(title)
    print("="*80)

def test_api_health():
    """Test API health endpoint."""
    print_section("TEST: API Health Check")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"Status: {result['status']}")
            print(f"Uptime: {result['uptime_seconds']:.2f} seconds")
            print(f"Systems: {json.dumps(result['systems'], indent=2)}")
            test_results.append(("API Health Check", True, f"Status: {result['status']}"))
            return True
        else:
            error_msg = f"Health check failed: {response.status_code}"
            print(error_msg)
            test_results.append(("API Health Check", False, error_msg))
            return False
    except requests.exceptions.ConnectionError:
        error_msg = "API server is not running!"
        print(f"ERROR: {error_msg}")
        print("Please start the server: python api_server.py")
        test_results.append(("API Health Check", False, error_msg))
        return False
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        test_results.append(("API Health Check", False, error_msg))
        return False

def test_api_search():
    """Test API search endpoint."""
    print_section("TEST: API Restaurant Search (Task 1.1 + 1.2)")
    
    queries = [
        "Find Italian restaurants in downtown Dubai",
        "Show me romantic restaurants",
        "What are the best rated Chinese restaurants?"
    ]
    
    search_passed = 0
    search_failed = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        payload = {"query": query, "thread_id": f"test_{i}"}
        
        try:
            response = requests.post(f"{BASE_URL}/api/v1/search", json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                print(f"  Found {result['num_results']} restaurant(s)")
                print(f"  Latency: {result['latency_ms']:.2f} ms")
                if result['restaurants']:
                    print(f"  Top result: {result['restaurants'][0]['name']}")
                search_passed += 1
            else:
                error_msg = f"Error: {response.status_code} - {response.text[:100]}"
                print(f"  {error_msg}")
                search_failed += 1
        except Exception as e:
            error_msg = f"Error: {e}"
            print(f"  {error_msg}")
            search_failed += 1
    
    if search_failed == 0:
        test_results.append(("API Search (3 queries)", True, f"All {search_passed} queries successful"))
    else:
        test_results.append(("API Search (3 queries)", False, f"{search_passed} passed, {search_failed} failed"))

def test_api_predict():
    """Test API rating prediction endpoint."""
    print_section("TEST: API Rating Prediction (Task 2.1)")
    
    restaurant_ids = [1, 5, 10, 15, 20]
    
    predict_passed = 0
    predict_failed = 0
    
    for restaurant_id in restaurant_ids:
        payload = {"restaurant_id": restaurant_id}
        
        try:
            response = requests.post(f"{BASE_URL}/api/v1/predict", json=payload, timeout=10)
            if response.status_code == 200:
                result = response.json()
                print(f"Restaurant {restaurant_id} ({result['restaurant_name']}):")
                print(f"  Predicted: {result['predicted_rating']}/5.0")
                if result['actual_rating']:
                    print(f"  Actual: {result['actual_rating']}/5.0")
                print(f"  Confidence: {result['confidence']}")
                predict_passed += 1
            else:
                error_msg = f"Error {response.status_code}"
                print(f"Restaurant {restaurant_id}: {error_msg}")
                predict_failed += 1
        except Exception as e:
            error_msg = f"Error - {e}"
            print(f"Restaurant {restaurant_id}: {error_msg}")
            predict_failed += 1
    
    if predict_failed == 0:
        test_results.append(("API Predict (5 restaurants)", True, f"All {predict_passed} predictions successful"))
    else:
        test_results.append(("API Predict (5 restaurants)", False, f"{predict_passed} passed, {predict_failed} failed"))

def test_api_metrics():
    """Test API metrics endpoint."""
    print_section("TEST: API Metrics (Task 3.1)")
    
    try:
        response = requests.get(f"{BASE_URL}/api/v1/metrics", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"Total Requests: {result['total_requests']}")
            print(f"  - Search: {result['search_requests']}")
            print(f"  - Predict: {result['predict_requests']}")
            print(f"Errors: {result['errors']}")
            print(f"Average Latency: {result['avg_latency_ms']:.2f} ms")
            print(f"Uptime: {result['uptime_seconds']:.2f} seconds")
            test_results.append(("API Metrics", True, f"{result['total_requests']} total requests"))
        else:
            error_msg = f"Error: {response.status_code}"
            print(error_msg)
            test_results.append(("API Metrics", False, error_msg))
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        test_results.append(("API Metrics", False, error_msg))

def main():
    """Run complete system test."""
    print("="*80)
    print("COMPLETE SYSTEM TEST - ALL TASKS")
    print("="*80)
    print("\nTasks being tested:")
    print("  - Task 1.1: RAG System (via API)")
    print("  - Task 1.2: Agentic Workflow (via API)")
    print("  - Task 2.1: Rating Prediction Model (via API)")
    print("  - Task 3.1: REST API & Deployment")
    print("\n" + "="*80)
    print("IMPORTANT: Make sure API server is running!")
    print("Start it with: python api_server.py")
    print("="*80)
    
    # Check if server is running
    if not test_api_health():
        print("\n" + "="*80)
        print("TEST ABORTED: API server is not running")
        print("="*80)
        sys.exit(1)
    
    # Run all tests
    test_api_search()
    test_api_predict()
    test_api_metrics()
    
    # Print test summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_tests = [t for t in test_results if t[1]]
    failed_tests = [t for t in test_results if not t[1]]
    
    print(f"\nTotal Tests: {len(test_results)}")
    print(f"Passed: {len(passed_tests)}")
    print(f"Failed: {len(failed_tests)}")
    
    if passed_tests:
        print("\n" + "-"*80)
        print("PASSED TESTS:")
        print("-"*80)
        for test_name, _, message in passed_tests:
            print(f"  [PASS] {test_name}")
            if message:
                print(f"         {message}")
    
    if failed_tests:
        print("\n" + "-"*80)
        print("FAILED TESTS:")
        print("-"*80)
        for test_name, _, message in failed_tests:
            print(f"  [FAIL] {test_name}")
            if message:
                print(f"         {message}")
    
    print("\n" + "="*80)
    print("COMPLETE SYSTEM TEST FINISHED!")
    print("="*80)
    
    if len(failed_tests) == 0:
        print("\nAll systems tested via REST API:")
        print("  [OK] Task 1.1: RAG System")
        print("  [OK] Task 1.2: Agentic Workflow")
        print("  [OK] Task 2.1: Rating Prediction")
        print("  [OK] Task 3.1: REST API")
        print("\n" + "="*80)
        return 0
    else:
        print(f"\nWARNING: {len(failed_tests)} test(s) failed!")
        print("Please check the errors above.")
        print("="*80)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

