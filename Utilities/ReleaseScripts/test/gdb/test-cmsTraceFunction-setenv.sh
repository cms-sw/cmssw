#!/bin/bash -ex

TRACE="cmsTraceFunction --startAfterFunction ScheduleItems::initMisc setenv -f putenv --abort"

check_func() {
  local func_name="$1"
  local src_name="$2"
  local exe_name="test-cmsTraceFunction-${func_name}"
  local raw_log="${func_name}_raw.log"
  local log="${func_name}.log"

  g++ -o "$exe_name" "$(dirname $0)/$src_name"
  set +e
  $TRACE ./$exe_name 2>&1 > "$raw_log"
  local ret=$?
  set -e
  grep "$func_name" "$raw_log" > "$log"

  if [ ${ret} = 0 ]; then
    echo "cmsTraceFunction exited with exit code 0, expected non-zero exit code"
    exit 1
  fi

  local call_count=$(grep -c "^${func_name}() called" "$log")
  local break_count=$(grep -c "Breakpoint .* in ${func_name} ()" "$log")
  if [ ${call_count} != 1 ] || [ ${break_count} != 1 ] ; then
    echo "Unexpected number of ${func_name} calls ${call_count} or breakpoints ${break_count}; expecting both to be 1"
    exit 1
  fi
}

# Check setenv
check_func "setenv" "test-cmsTraceFunction-setenv.cpp"

# Check putenv
check_func "putenv" "test-cmsTraceFunction-putenv.cpp"
