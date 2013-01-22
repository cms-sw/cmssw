#ifndef RPCDigitizer_RPCSynchronizer_h
#define RPCDigitizer_RPCSynchronizer_h

/** \class RPCSynchronizer
 *   Class for the RPC strip response simulation based
 *   on a parametrized model (ORCA-based)
 *
 *  \author Raffaello Trentadue -- INFN Bari
 */

#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>

//#include "CLHEP/config/CLHEP.h"
#include "CLHEP/Random/Random.h"
#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGaussQ.h"
#include "CLHEP/Random/RandPoissonQ.h"

#include <FWCore/Framework/interface/Frameworkfwd.h>
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <set>

class PSimHit;
class RPCSimSetUp;

namespace edm{
  class ParameterSet;
}

namespace CLHEP {
  class RandGaussianQ;
  class RandPoissonQ;
  class RandFlat;
}

class RPCSynchronizer
{
 public:
  RPCSynchronizer(const edm::ParameterSet& config);
  ~RPCSynchronizer();

  int getSimHitBx(const PSimHit*);
  void setRPCSimSetUp(RPCSimSetUp *simsetup){theSimSetUp = simsetup;}
  RPCSimSetUp* getRPCSimSetUp(){ return theSimSetUp; }

  void setRandomEngine(CLHEP::HepRandomEngine& eng);

 private:

  double resRPC;
  double timOff;
  double dtimCs;
  double resEle;
  double sspeed;
  double cspeed;
  double lbGate;
  double lbGateNew;
  double cosmicPar;
  double LHCGate;
  bool cosmics;

  CLHEP::RandGaussQ *gauss1;
  CLHEP::RandGaussQ *gauss2;
  RPCSimSetUp * theSimSetUp;

};
#endif

