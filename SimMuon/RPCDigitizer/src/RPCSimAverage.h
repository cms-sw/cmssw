#ifndef RPCDigitizer_RPCSimAverage_h
#define RPCDigitizer_RPCSimAverage_h

/** \class RPCSimAverage
 *   Class for the RPC strip response simulation based
 *   on a parametrized model (ORCA-based)
 *
 *  \author Raffaello Trentadue -- INFN Bari
 */
#include "SimMuon/RPCDigitizer/src/RPCSim.h"


#include<cstring>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<stdlib.h>

namespace CLHEP {
  class HepRandomEngine;
  class RandFlat;
}
class RPCSimAverage : public RPCSim
{
 public:
  RPCSimAverage(const edm::ParameterSet& config);
  ~RPCSimAverage(){}
  void simulate(const RPCRoll* roll,
			const edm::PSimHitContainer& rpcHits );

  int getClSize(float posX); 

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
  
  std::map< int, std::vector<double> > clsMap;
  std::vector<double> sum_clsize;
  std::ifstream *infile;
 
  std::fstream *MyOutput1; 
  std::fstream *MyOutput2;
  std::fstream *MyOutput3;

  CLHEP::HepRandomEngine* rndEngine;
  CLHEP::RandFlat* flatDistribution;

};
#endif
