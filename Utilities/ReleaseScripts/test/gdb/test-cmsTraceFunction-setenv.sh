#!/bin/bash -ex
g++ -o test-cmsTraceFunction-setenv $(dirname $0)/test-cmsTraceFunction-setenv.cpp
cmsTraceFunction         --startAfterFunction ScheduleItems::initMisc setenv ./test-cmsTraceFunction-setenv 2>&1 | grep setenv > setenv.log
cmsTraceFunction --abort --startAfterFunction ScheduleItems::initMisc setenv ./test-cmsTraceFunction-setenv 2>&1 | grep setenv > setenv-abort.log
rm -f test-cmsTraceFunction-setenv
setenv_count=$(grep '^setenv() called' setenv.log | wc -l)
break_setenv=$(grep 'Breakpoint .* in setenv ()' setenv.log | wc -l)
if [ ${setenv_count} != 3 ] || [ ${break_setenv} != 2 ] ; then
  exit 1
fi
setenv_count=$(grep '^setenv() called' setenv-abort.log | wc -l)
break_setenv=$(grep 'Breakpoint .* in setenv ()' setenv-abort.log | wc -l)
if [ ${setenv_count} != 1 ] || [ ${break_setenv} != 1 ] ; then
  exit 1
fi
