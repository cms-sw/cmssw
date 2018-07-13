// Copyright (C) 2002-2018 CERN for the benefit of the ATLAS collaboration

#include "usingns6_test1.h"

class StoreGateSvc {};


class TileCablingSvc {
public:
  virtual ~TileCablingSvc() {}
  ServiceHandle<StoreGateSvc> m_detStore;
};

namespace CLHEP { class RandGauss; }
using CLHEP::RandGauss;
 
