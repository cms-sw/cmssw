#ifndef RPCDigitizer_RPCSimModelTiming_h
#define RPCDigitizer_RPCSimModelTiming_h

/** \class RPCSimAverage
 *   Class for the RPC strip response simulation based
 *   on a parametrized model (ORCA-based)
 *
 *  \author Borislav Pavlov -- University of Sofia
 */
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAsymmetricCls.h"
#include "SimMuon/RPCDigitizer/src/RPCSimAverageNoiseEffCls.h"

#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class RPCGeometry;

namespace CLHEP {
  class HepRandomEngine;
}

class RPCSimModelTiming : public RPCSimAverageNoiseEffCls// inherits RPCSimAverageNoiseEffCls to coupe with the condidions
{
 public:
  RPCSimModelTiming(const edm::ParameterSet& config);
  ~RPCSimModelTiming();
  void simulateIRPC(const RPCRoll* roll,
                const edm::PSimHitContainer& rpcHits,
                 CLHEP::HepRandomEngine*) override;
  void simulateIRPCNoise(const RPCRoll*,
                     CLHEP::HepRandomEngine*) override;
};
#endif
