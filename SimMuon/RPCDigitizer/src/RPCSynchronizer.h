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
#include<cstdlib>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <set>

class PSimHit;
class RPCSimSetUp;

namespace edm{
  class ParameterSet;
}

namespace CLHEP {
  class HepRandomEngine;
}

class RPCSynchronizer
{
 public:
  RPCSynchronizer(const edm::ParameterSet& config);
  ~RPCSynchronizer();

  int getSimHitBx(const PSimHit*, CLHEP::HepRandomEngine*);
  int getSimHitBxAndTimingForIRPC(const PSimHit*, CLHEP::HepRandomEngine*);
  void setRPCSimSetUp(RPCSimSetUp *simsetup){theSimSetUp = simsetup;}
  RPCSimSetUp* getRPCSimSetUp(){ return theSimSetUp; }
  double getExactTime() const {return the_exact_time;}
  double getSmearedTime() const {return the_smeared_time;} 

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
  double irpc_timing_res;
  double irpc_electronics_jitter;
  double the_exact_time;
  double the_smeared_time;
  RPCSimSetUp * theSimSetUp;
  int N_BX;
};
#endif
