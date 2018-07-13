// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

namespace std { int qq; }
using namespace std;
using std::qq;
#include "usingns1_test.h"


//// Test for a crash.
void
tree_to_shwi()
{
  const char* s __attribute__((unused)) = __FUNCTION__;
}

