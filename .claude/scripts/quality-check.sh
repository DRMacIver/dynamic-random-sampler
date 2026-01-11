#!/bin/bash
# Quality check script - runs all quality gates

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FAILED=0
WARNINGS=0

echo "# Quality Check"
echo ""
echo "## Running Quality Gates"
echo ""

# 1. Lint check (ruff + basedpyright)
echo "### Linting"
if just lint 2>/dev/null; then
    echo -e "${GREEN}Lint: PASSED${NC}"
else
    echo -e "${RED}Lint: FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# 2. Tests
echo "### Tests"
if just test 2>/dev/null; then
    echo -e "${GREEN}Tests: PASSED${NC}"
else
    echo -e "${RED}Tests: FAILED${NC}"
    FAILED=$((FAILED + 1))
fi
echo ""

# Summary
echo "---"
echo ""
echo "## Summary"
echo ""

if [ $FAILED -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}All quality checks passed!${NC}"
    echo ""
    echo "Quality gate: **PASSED**"
    exit 0
elif [ $FAILED -eq 0 ]; then
    echo -e "${YELLOW}Quality checks passed with $WARNINGS warning(s)${NC}"
    echo ""
    echo "Quality gate: **PASSED WITH WARNINGS**"
    exit 0
else
    echo -e "${RED}$FAILED quality check(s) failed${NC}"
    echo ""
    echo "Quality gate: **FAILED**"
    echo ""
    echo "Fix the failing checks before proceeding."
    exit 1
fi
