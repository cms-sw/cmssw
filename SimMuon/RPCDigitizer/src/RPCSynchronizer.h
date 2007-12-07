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

#include "CLHEP/config/CLHEP.h"
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

class RPCGeometry;
class PSimHit;

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
  float getReadOutTime(const RPCDetId& rpcDetId);
  void setReadOutTime(const RPCGeometry*);
  int getSimHitBx(const PSimHit*);

  /// sets geometry
  void setGeometry(const RPCGeometry * geom) {theGeometry = geom;}

 private:
  std::map<RPCDetId, float> _bxmap;
  const RPCGeometry * theGeometry;

  double resRPC;
  double timOff;
  double dtimCs;
  double resEle;
  double sspeed;
  double lbGate;
  double lbGateNew;

  std::string filename;
  bool file;
  bool cosmics;
  std::fstream* infile;

  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandGaussQ *gaussian_;
  CLHEP::RandPoissonQ *poissonDistribution_;
  CLHEP::RandFlat *flatDistribution_;

};
#endif
