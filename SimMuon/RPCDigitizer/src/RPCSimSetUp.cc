#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include <cmath>
#include <math.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include<cstring>
#include<string>
#include<vector>
#include<stdlib.h>
#include <utility>
#include <map>

using namespace std;

RPCSimSetUp::RPCSimSetUp(const edm::ParameterSet& ps) {

  _mapDetIdNoise.clear();
  _mapDetIdEff.clear();
  _bxmap.clear();
  _clsMap.clear();

}

void RPCSimSetUp::setRPCSetUp(std::vector<RPCStripNoises::NoiseItem> vnoise, std::vector<double> vcls){

  double sum = 0;
  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<double> sum_clsize;

  for(unsigned int n = 0; n < vcls.size(); ++n){
    //    std::cout<<"SUM: "<<vcls[n]<<std::endl;

    sum_clsize.push_back(vcls[n]);

    if(counter == row*20) {
      //    std::cout<<"ROW: "<<row<<std::endl;
      _clsMap[row] = sum_clsize;
      row++;
      sum = 0;
      sum_clsize.clear();
    }
    counter++;
  }

  for(std::vector<RPCStripNoises::NoiseItem>::iterator it = vnoise.begin(); it != vnoise.end(); ++it){

    _bxmap[RPCDetId(it->dpid)] = it->time;
    std::vector<float> veff, vnoise;

    veff.clear();
    vnoise.clear();

    for(unsigned int j = 0; j < 96;++j){
      vnoise.push_back((it->noise)[j]);
      veff.push_back((it->eff)[j]);
    }

    _mapDetIdNoise[it->dpid]= vnoise;
    _mapDetIdEff[it->dpid] = veff;

  }
}

std::vector<float> RPCSimSetUp::getNoise(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdNoise.find(id);
  std::vector<float> vnoise = iter->second;

  return vnoise;
}

std::vector<float> RPCSimSetUp::getEff(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdEff.find(id);
  std::vector<float> veff = iter->second;

  return veff;
}

float RPCSimSetUp::getTime(uint32_t id)
{
  RPCDetId rpcid(id);
  std::map<RPCDetId, float>::iterator iter = _bxmap.find(rpcid);
  return iter->second;
}

std::map< int, std::vector<double> > RPCSimSetUp::getClsMap()
{
  return _clsMap;
}

RPCSimSetUp::~RPCSimSetUp(){}
