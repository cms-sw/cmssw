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

#include <cstring>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "FWCore/Framework/interface/EventSetup.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

class RPCGeometry;
//class RPCSimSetUp;

namespace CLHEP {
  class HepRandomEngine;
}

class RPCSimAverageNoiseEffCls : public RPCSim {
public:
  RPCSimAverageNoiseEffCls(const edm::ParameterSet& config);
  ~RPCSimAverageNoiseEffCls() override;

  void simulate(const RPCRoll* roll, const edm::PSimHitContainer& rpcHits, CLHEP::HepRandomEngine*) override;

  void simulateNoise(const RPCRoll*, CLHEP::HepRandomEngine*) override;

  int getClSize(float posX, CLHEP::HepRandomEngine*);
  int getClSize(uint32_t id, float posX, CLHEP::HepRandomEngine*);

  // private:
protected:
  void init() override{};

  double aveEff;
  double aveCls;
  double resRPC;
  double timOff;
  double dtimCs;
  double resEle;
  double sspeed;
  double lbGate;
  bool rpcdigiprint;
  bool eledig;

  int N_hits;
  int nbxing;
  double rate;
  double gate;
  double frate;

  std::map<int, std::vector<double> > clsMap;
  std::vector<double> sum_clsize;
  std::vector<double> clsForDetId;
  std::ifstream* infile;

  RPCSynchronizer* _rpcSync;
};
#endif
