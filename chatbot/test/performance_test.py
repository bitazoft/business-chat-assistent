#!/usr/bin/env python3
"""
Performance monitoring and testing script for the chatbot
Run this to test response times before and after optimizations
"""

import time
import requests
import json
import statistics
from typing import List, Dict
import asyncio
import aiohttp

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_MESSAGES = [
    "What products do you have?",
    "I want to buy a laptop",
    "Track my order 12345",
    "What's the price of iPhone?",
    "Update my email address",
    "Show me all available products",
    "Place an order for 2 phones",
    "Check my account details",
    "What's the status of order ABC123?",
    "I need help with my account"
]

TEST_PAYLOAD = {
    "session_id": "test-session",
    "seller_id": "1",
    "user_id": "test-user",
    "chat_history": []
}

def test_single_request(message: str) -> Dict:
    """Test a single request and measure response time"""
    payload = {**TEST_PAYLOAD, "message": message}
    
    start_time = time.time()
    try:
        response = requests.post(f"{BASE_URL}/chat", json=payload, timeout=60)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        if response.status_code == 200:
            data = response.json()
            return {
                "message": message,
                "response_time": response_time,
                "status": "success",
                "response": data.get("response", ""),
                "processing_time": data.get("processing_time", "N/A")
            }
        else:
            return {
                "message": message,
                "response_time": response_time,
                "status": "error",
                "error": f"HTTP {response.status_code}"
            }
    except requests.exceptions.Timeout:
        return {
            "message": message,
            "response_time": 60.0,
            "status": "timeout",
            "error": "Request timeout"
        }
    except Exception as e:
        return {
            "message": message,
            "response_time": time.time() - start_time,
            "status": "error",
            "error": str(e)
        }

async def test_async_request(session: aiohttp.ClientSession, message: str) -> Dict:
    """Test a single async request"""
    payload = {**TEST_PAYLOAD, "message": message}
    
    start_time = time.time()
    try:
        async with session.post(f"{BASE_URL}/chat", json=payload, timeout=60) as response:
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status == 200:
                data = await response.json()
                return {
                    "message": message,
                    "response_time": response_time,
                    "status": "success",
                    "response": data.get("response", ""),
                    "processing_time": data.get("processing_time", "N/A")
                }
            else:
                return {
                    "message": message,
                    "response_time": response_time,
                    "status": "error",
                    "error": f"HTTP {response.status}"
                }
    except Exception as e:
        return {
            "message": message,
            "response_time": time.time() - start_time,
            "status": "error",
            "error": str(e)
        }

def run_sequential_tests() -> Dict:
    """Run sequential performance tests"""
    print("🚀 Running sequential performance tests...")
    
    results = []
    for i, message in enumerate(TEST_MESSAGES, 1):
        print(f"  [{i}/{len(TEST_MESSAGES)}] Testing: {message[:30]}...")
        result = test_single_request(message)
        results.append(result)
        print(f"    ⏱️  {result['response_time']:.2f}s - {result['status']}")
    
    return analyze_results(results, "Sequential")

async def run_concurrent_tests() -> Dict:
    """Run concurrent performance tests"""
    print("🚀 Running concurrent performance tests...")
    
    async with aiohttp.ClientSession() as session:
        tasks = [test_async_request(session, msg) for msg in TEST_MESSAGES]
        results = await asyncio.gather(*tasks)
    
    return analyze_results(results, "Concurrent")

def analyze_results(results: List[Dict], test_type: str) -> Dict:
    """Analyze test results and print statistics"""
    print(f"\n📊 {test_type} Test Results:")
    print("=" * 50)
    
    successful_results = [r for r in results if r['status'] == 'success']
    error_results = [r for r in results if r['status'] != 'success']
    
    if successful_results:
        response_times = [r['response_time'] for r in successful_results]
        
        stats = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(error_results),
            "success_rate": len(successful_results) / len(results) * 100,
            "avg_response_time": statistics.mean(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "median_response_time": statistics.median(response_times)
        }
        
        if len(response_times) > 1:
            stats["std_dev"] = statistics.stdev(response_times)
        else:
            stats["std_dev"] = 0
        
        print(f"✅ Successful requests: {stats['successful_requests']}/{stats['total_requests']} ({stats['success_rate']:.1f}%)")
        print(f"⏱️  Average response time: {stats['avg_response_time']:.2f}s")
        print(f"🏃 Fastest response: {stats['min_response_time']:.2f}s")
        print(f"🐌 Slowest response: {stats['max_response_time']:.2f}s")
        print(f"📊 Median response time: {stats['median_response_time']:.2f}s")
        print(f"📈 Standard deviation: {stats['std_dev']:.2f}s")
        
        # Performance ratings
        avg_time = stats['avg_response_time']
        if avg_time < 1.0:
            rating = "🟢 EXCELLENT"
        elif avg_time < 2.0:
            rating = "🟡 GOOD"
        elif avg_time < 5.0:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 NEEDS IMPROVEMENT"
        
        print(f"🎯 Performance rating: {rating}")
        
    else:
        print("❌ All requests failed!")
        stats = {"total_requests": len(results), "successful_requests": 0, "failed_requests": len(error_results)}
    
    if error_results:
        print(f"\n❌ Failed requests ({len(error_results)}):")
        for result in error_results:
            print(f"  - {result['message'][:30]}... : {result['error']}")
    
    return stats

def check_server_health():
    """Check if the server is running"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    """Main performance test runner"""
    print("🔍 Chatbot Performance Testing Tool")
    print("=" * 40)
    
    # Check server health
    if not check_server_health():
        print("❌ Server is not running or not responding")
        print(f"   Please start the server at {BASE_URL}")
        return
    
    print("✅ Server is running and healthy")
    
    # Run sequential tests
    sequential_stats = run_sequential_tests()
    
    # Run concurrent tests
    print("\n" + "=" * 50)
    concurrent_stats = asyncio.run(run_concurrent_tests())
    
    # Summary comparison
    print("\n📈 Performance Summary:")
    print("=" * 50)
    
    if sequential_stats.get('successful_requests', 0) > 0 and concurrent_stats.get('successful_requests', 0) > 0:
        seq_avg = sequential_stats['avg_response_time']
        conc_avg = concurrent_stats['avg_response_time']
        
        print(f"Sequential average: {seq_avg:.2f}s")
        print(f"Concurrent average: {conc_avg:.2f}s")
        
        if conc_avg < seq_avg:
            improvement = ((seq_avg - conc_avg) / seq_avg) * 100
            print(f"🚀 Concurrent testing is {improvement:.1f}% faster!")
        else:
            degradation = ((conc_avg - seq_avg) / seq_avg) * 100
            print(f"⚠️  Concurrent testing is {degradation:.1f}% slower")
    
    print("\n🎯 Optimization Recommendations:")
    avg_time = sequential_stats.get('avg_response_time', 10)
    
    if avg_time > 5:
        print("🔴 CRITICAL: Response times are too slow (>5s)")
        print("   - Check database connection pooling")
        print("   - Optimize vector store loading")
        print("   - Consider caching strategies")
    elif avg_time > 2:
        print("🟠 MODERATE: Response times could be improved (>2s)")
        print("   - Implement result caching")
        print("   - Optimize LLM settings")
        print("   - Reduce RAG search scope")
    elif avg_time > 1:
        print("🟡 MINOR: Good performance with room for improvement (>1s)")
        print("   - Fine-tune caching parameters")
        print("   - Consider async processing")
    else:
        print("🟢 EXCELLENT: Response times are optimal (<1s)")

if __name__ == "__main__":
    main()
