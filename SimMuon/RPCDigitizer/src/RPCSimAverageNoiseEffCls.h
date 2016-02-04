#ifndef RPCDigitizer_RPCSimAverageNoiseEffCls_h
#define RPCDigitizer_RPCSimAverageNoiseEffCls_h

/** \class RPCSimAverage
 *   Class for the RPC strip response simulation based
 *   on a parametrized model (ORCA-based)
 *
 *  \author Borislav Pavlov -- University of Sofia
 */
#include "SimMuon/RPCDigitizer/src/RPCSim.h"
#include "SimMuon/RPCDigitizer/src/RPCSynchronizer.h"

#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

class RPCGeometry;
//class RPCSimSetUp;

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
  class RandPoisson;
}

class RPCSimAverageNoiseEffCls : public RPCSim
{
 public:
  RPCSimAverageNoiseEffCls(const edm::ParameterSet& config);
  ~RPCSimAverageNoiseEffCls();

  void simulate(const RPCRoll* roll,
		const edm::PSimHitContainer& rpcHits);

  void simulateNoise(const RPCRoll*);

  void setRandomEngine(CLHEP::HepRandomEngine& eng);

  int getClSize(float posX);
  int getClSize(uint32_t id,float posX);

 private:
  void init(){};
 private:
  double aveEff;
  double aveCls;
  double resRPC;
  double timOff;
  double dtimCs;
  double resEle;
  double sspeed;
  double lbGate;
  bool rpcdigiprint;
  
  int N_hits;
  int nbxing;
  double rate;
  double gate;
  double frate;

  std::map< int, std::vector<double> > clsMap;
  std::vector<double> sum_clsize;
  std::vector<double> clsForDetId;
  std::ifstream *infile;
 
  RPCSynchronizer* _rpcSync;

  //Defining the engines in the constructor and the method
  //CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;
  //Adding a second flatDistribution, since it was redefined 
  //in a method with different interval
  CLHEP::RandFlat* flatDistribution2;
  CLHEP::RandPoissonQ *poissonDistribution_;

};
#endif
