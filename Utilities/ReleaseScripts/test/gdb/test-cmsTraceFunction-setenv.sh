#!/bin/bash -ex

TRACE="cmsTraceFunction --startAfterFunction ScheduleItems::initMisc setenv -f putenv --abort"

# Check setenv
g++ -o test-cmsTraceFunction-setenv $(dirname $0)/test-cmsTraceFunction-setenv.cpp
set +e
$TRACE ./test-cmsTraceFunction-setenv 2>&1 > setenv_raw.log
ret_setenv=$?
set -e
grep setenv setenv_raw.log > setenv.log
#rm -f test-cmsTraceFunction-setenv

if [ ${ret_setenv} = 0 ]; then
  echo "cmsTraceFunction exited with exit code 0, expected non-zero exit code"
  exit 1
fi

setenv_count=$(grep -c '^setenv() called' setenv.log)
break_setenv=$(grep -c 'Breakpoint .* in setenv ()' setenv.log)
if [ ${setenv_count} != 1 ] || [ ${break_setenv} != 1 ] ; then
  echo "Unexpected number of setenv calls ${setenv_count} or breakpoints ${break_setenv}; expecting both to be 1"
  exit 1
fi

# Check putenv
g++ -o test-cmsTraceFunction-putenv $(dirname $0)/test-cmsTraceFunction-putenv.cpp
set +e
$TRACE ./test-cmsTraceFunction-putenv 2>&1 > putenv_raw.log
ret_putenv=$?
set -e
grep putenv putenv_raw.log > putenv.log
rm -f test-cmsTraceFunction-putenv

if [ ${ret_puttenv} = 0 ]; then
  echo "cmsTraceFunction exited with exit code 0, expected non-zero exit code"
  exit 1
fi

putenv_count=$(grep -c '^putenv() called' putenv.log)
break_putenv=$(grep -c 'Breakpoint .* in putenv ()' putenv.log)
if [ ${putenv_count} != 1 ] || [ ${break_tenv} != 1 ] ; then
  echo "Unexpected number of putenv calls ${putenv_count} or breakpoints ${break_putenv}; expecting both to be 1"
  exit 1
fi
