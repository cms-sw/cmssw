// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

class BaseInfoBase {};

template <class T>
class BaseInfo
{
public:
  static const BaseInfoBase& baseinfo ();

  struct instance_holder
  {
   instance_holder();
    BaseInfoBase* instance;
  };
  static instance_holder s_instance;
};


#include "usingns5_test.h"

const BaseInfoBase* pp = &BaseInfo<int>::baseinfo();

namespace CLHEP {class RandGauss; }
using CLHEP::RandGauss;

