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

void RPCSimSetUp::setRPCSetUp(std::vector<RPCStripNoises::NoiseItem> vnoise, std::vector<float> vcls){

  double sum = 0;
  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<double> sum_clsize;

  for(unsigned int n = 0; n < vcls.size(); ++n){

    sum_clsize.push_back(vcls[n]);

    if(counter == row*20) {

      _clsMap[row] = sum_clsize;
      row++;
      sum = 0;
      sum_clsize.clear();
    }
    counter++;
  }

  unsigned int n = 0; uint32_t temp; 
  std::vector<float> veff, vvnoise;
  veff.clear();
  vvnoise.clear();

  for(std::vector<RPCStripNoises::NoiseItem>::iterator it = vnoise.begin(); it != vnoise.end(); ++it){

    if(n%96 == 0) {
      if(n > 0 ){
	_mapDetIdNoise[temp]= vvnoise;
	_mapDetIdEff[temp] = veff;
	_bxmap[RPCDetId(temp)] = it->time;
      }

      veff.clear();
      vvnoise.clear();


      vvnoise.push_back((it->noise));
      veff.push_back((it->eff));

    } else if (n == vnoise.size()-1 ){
      temp = it->dpid;
      vvnoise.push_back((it->noise));
      veff.push_back((it->eff));
      _mapDetIdNoise[temp]= vvnoise;
      _mapDetIdEff[temp] = veff;
    } else {
      temp = it->dpid;
      vvnoise.push_back((it->noise));
      veff.push_back((it->eff));
    }
    n++;
  }
}

const std::vector<float>& RPCSimSetUp::getNoise(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdNoise.find(id);
  return iter->second;
}

const std::vector<float>& RPCSimSetUp::getEff(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdEff.find(id);
  return iter->second;
}

float RPCSimSetUp::getTime(uint32_t id)
{
  RPCDetId rpcid(id);
  std::map<RPCDetId, float>::iterator iter = _bxmap.find(rpcid);
  return iter->second;
}

const std::map< int, std::vector<double> >& RPCSimSetUp::getClsMap()
{
  return _clsMap;
}

RPCSimSetUp::~RPCSimSetUp(){}
