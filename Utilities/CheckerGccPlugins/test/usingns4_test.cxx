// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

#include <typeinfo>


class BaseInfoBase
{
public:
  static void addInit (const std::type_info* tinfo);
};

#include "usingns4_test.h"


template struct RegisterBaseInit<int>;


namespace CLHEP {class RandGauss;}
using CLHEP::RandGauss;

