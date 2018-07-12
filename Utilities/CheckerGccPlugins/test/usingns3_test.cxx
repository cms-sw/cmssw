// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

#include "usingns3_test.h"

int stoi() { return stoa<long>(); }

namespace {}
namespace Gaudi { namespace Units {}}
using namespace Gaudi::Units;
