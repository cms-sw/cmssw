#!/bin/bash -ex

TRACE="cmsTraceFunction --startAfterFunction ScheduleItems::initMisc setenv"

check_failure() {
  local test_name="$1"
  local check_func="$2"
  local exe_name="test-cmsTraceFunction-${test_name}"
  local src_name="${exe_name}.cpp"

  g++ -o "$exe_name" "$(dirname $0)/$src_name"

  for when in before after; do
    local log="${test_name}_${when}.log"
    set +e
    $TRACE ./$exe_name $when 2>&1 > "$log"
    local ret=$?
    set -e

    if [ ${ret} = 0 ]; then
      echo "cmsTraceFunction for ${test_name} ${when} exited with exit code 0, expected non-zero exit code"
      exit 1
    fi

    if [ -n "${check_func}" ]; then
      ${check_func} ${log}
    fi
  done

  log_after="${test_name}_after.log"
  if ! grep -q "ScheduleItems::initMisc() called" ${log_after}; then
    echo "Did not find expected ScheduleItems::initMisc call in ${log_after}"
    exit 1
  fi
}

# Non-zero exit code
check_failure "nonzero"

# Assertion failure
check_assert_log() {
  local logfile="$1"
  if ! grep -q "signal SIGABRT" ${logfile}; then
    echo "Did not find signal message in ${logfile}"
    exit 1
  fi
  if ! grep -q "in my_assert" ${logfile}; then
    echo "Did not find my_assert function stack trace in ${logfile}"
    exit 1
  fi
}
check_failure "assert" check_assert_log

# Segfault
check_segfault_log() {
  local logfile="$1"
  if ! grep -q "signal SIGSEGV" ${logfile}; then
    echo "Did not find signal message in ${logfile}"
    exit 1
  fi
  if ! grep -q "in my_segfault" ${logfile}; then
    echo "Did not find my_segfault function in stack trace in ${logfile}"
    exit 1
  fi
}
check_failure "segfault"
