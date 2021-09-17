#!/bin/bash -e

SCRIPT_NAME=$(basename $0)
TEST_NAME="test-valgrind-memleak"
which ${TEST_NAME}
valgrind --leak-check=full --undef-value-errors=no --error-limit=no \
         ${TEST_NAME} > ${SCRIPT_NAME}.log 2>&1

cat ${SCRIPT_NAME}.log
echo ""
COUNT=$(grep 'definitely lost: [1-9][0-9]*' ${SCRIPT_NAME}.log | wc -l)
rm -f ${SCRIPT_NAME}.log

if [ $COUNT -eq 0 ] ; then
  echo "ERROR: Valgrind was suppose to find memory leaks"
  exit 1
else
  echo "ALL OK"
fi
