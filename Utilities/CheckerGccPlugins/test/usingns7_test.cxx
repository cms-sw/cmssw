// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

#include "usingns7_test.h"

using namespace rapidjson;

Value& MemoryMonitor (Value& d) {
  return d["Max"];
}

