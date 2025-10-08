#!/bin/bash
# Quick test runner for CausalWanModel tests

set -e

cd "$(dirname "$0")/.."

echo "=================================================="
echo "CausalWanModel Test Suite"
echo "=================================================="
echo ""

# Parse arguments
TEST_TYPE=${1:-"all"}

case $TEST_TYPE in
    "correctness")
        echo "Running correctness tests..."
        pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_correctness -v -s
        pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_forward_with_kv_cache -v -s
        ;;

    "benchmark")
        echo "Running performance benchmark..."
        pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_performance_benchmark -v -s
        echo ""
        echo "Baseline saved to tests/baseline_performance.txt"
        cat tests/baseline_performance.txt
        ;;

    "compare")
        echo "Running model comparison..."
        pytest tests/test_causal_wan_model.py::TestCausalWanModel::test_model_comparison -v -s
        ;;

    "all")
        echo "Running all tests..."
        pytest tests/test_causal_wan_model.py -v -s
        ;;

    "help")
        echo "Usage: ./tests/run_tests.sh [TEST_TYPE]"
        echo ""
        echo "TEST_TYPE options:"
        echo "  correctness  - Run correctness tests only"
        echo "  benchmark    - Run performance benchmark"
        echo "  compare      - Run model comparison test"
        echo "  all          - Run all tests (default)"
        echo "  help         - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./tests/run_tests.sh correctness"
        echo "  ./tests/run_tests.sh benchmark"
        echo "  ./tests/run_tests.sh compare"
        ;;

    *)
        echo "Unknown test type: $TEST_TYPE"
        echo "Run './tests/run_tests.sh help' for usage"
        exit 1
        ;;
esac

echo ""
echo "=================================================="
echo "Tests completed!"
echo "=================================================="
