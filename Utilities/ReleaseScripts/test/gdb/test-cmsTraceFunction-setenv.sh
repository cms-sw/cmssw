#!/bin/bash -ex

TRACE="cmsTraceFunction --startAfterFunction ScheduleItems::initMisc setenv -f putenv"

check_func() {
  local func_name="$1"
  local src_name="$2"
  local trace_opts="$3"
  local exe_name="test-cmsTraceFunction-${func_name}"
  local raw_log="${func_name}_raw.log"
  local log="${func_name}.log"

  g++ -o "$exe_name" "$(dirname $0)/$src_name"
  set +e
  $TRACE $trace_opts ./$exe_name 2>&1 > "$raw_log"
  local ret=$?
  set -e
  grep "$func_name" "$raw_log" > "$log"

  if [ ${trace_opts} = "--abort" ]; then
    call_count_expected=1
    break_count_expected=1

    if [ ${ret} = 0 ]; then
      echo "cmsTraceFunction exited with exit code 0, expected non-zero exit code"
      exit 1
    fi
  else
    call_count_expected=3
    break_count_expected=2

    if [ ${ret} != 0 ]; then
      echo "cmsTraceFunction exited with exit code $ret, expected zero exit code"
      exit 1
    fi
  fi

  local call_count=$(grep -c "^${func_name}() called" "$log")
  local break_count=$(grep -c "Breakpoint .* in ${func_name} ()" "$log")
  if [ ${call_count} != ${call_count_expected} ] || [ ${break_count} != ${break_count_expected} ] ; then
    echo "Unexpected number of ${func_name} calls ${call_count} or breakpoints ${break_count}; expecting calls ${call_count_expected} and breakpoints ${break_count_expected}"
    exit 1
  fi
}

# Check setenv
check_func "setenv" "test-cmsTraceFunction-setenv.cpp" ""
check_func "setenv" "test-cmsTraceFunction-setenv.cpp" "--abort"


# Check putenv
check_func "putenv" "test-cmsTraceFunction-putenv.cpp" ""
check_func "putenv" "test-cmsTraceFunction-putenv.cpp" "--abort"
