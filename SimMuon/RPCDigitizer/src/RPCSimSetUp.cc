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

  std::cout<<"RPCSimSetup :: RPCSimSetup"<<std::endl;

}

void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<float>& vcls){

  std::cout<<"RPCSimSetup :: setRPCSetUp(RPCStripNoises::NoiseItem, float)"<<std::endl;

  unsigned int counter = 1;
  unsigned int row = 1;
  std::vector<double> sum_clsize;

  for(unsigned int n = 0; n < vcls.size(); ++n){

    sum_clsize.push_back(vcls[n]);

    if(counter == row*20) {

      _clsMap[row] = sum_clsize;
      row++;
      sum_clsize.clear();
    }
    counter++;
  }

  uint32_t  detId;
  RPCDetId rpcId;

  unsigned int n_tot  = 0;
  unsigned int n_roll = 0; 
  uint32_t temp = 0; 
  std::vector<float> veff, vvnoise;
  veff.clear();
  vvnoise.clear();

  for(std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end(); ++it){

    detId = it->dpid;
    // std::cout<<"Looking in theGeometry "<<theGeometry;
    // std::cout<<" for RPCDetId "<<detId;
    rpcId = RPCDetId(detId);
    // std::cout<<" aka "<<rpcId;
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(rpcId));
    // std::cout<<" roll "<<roll;

    if(roll !=0 ) {
      unsigned int numbStrips = roll->nstrips();
      // std::cout<<" with "<<numbStrips<<" strips"<<std::endl;
      // std::cout<<"Got RPCDetId "<<detId<<" aka "<<rpcId<<" with "<<numbStrips<<"strips"<<std::endl;

      // This dirty mess should be cleaned up
      // such that is is more clear to the reader
      // what is going on
      // -----------------------------------------
      if(n_roll%numbStrips == 0) {
	if(n_tot > 0 ){
	  _mapDetIdNoise[temp]= vvnoise;
	  _mapDetIdEff[temp] = veff;
	  _bxmap[RPCDetId(it->dpid)] = it->time;
	  
	  veff.clear();
	  vvnoise.clear();
	  vvnoise.push_back((it->noise));
	  veff.push_back((it->eff));
	}
	else if(n_tot == 0 ){
	vvnoise.push_back((it->noise));
	veff.push_back((it->eff));
	_bxmap[RPCDetId(it->dpid)] = it->time;
	}
      } else if (n_tot == vnoise.size()-1 ){
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
      ++n_tot;
      if(n_roll<numbStrips-1) ++n_roll;
      else n_roll = 0;
      // -----------------------------------------

    }
    // else std::cout<<std::endl;
  }
}

void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<RPCClusterSize::ClusterSizeItem>& vClusterSize){

  std::cout<<"RPCSimSetup :: setRPCSetUp(RPCStripNoises::NoiseItem, RPCClusterSize::ClusterSizeItem)"<<std::endl;

  std::vector<RPCClusterSize::ClusterSizeItem>::const_iterator itCls;

  uint32_t  detId;
  RPCDetId rpcId;

  int clsCounter(1);
  std::vector<double> clsVect;

  for(itCls = vClusterSize.begin(); itCls != vClusterSize.end(); ++itCls){
    clsVect.push_back(((double)(itCls->clusterSize)));
    if((!(clsCounter%100)) && (clsCounter!=0)){
      detId=itCls->dpid;
      _mapDetClsMap[detId]=clsVect;
      clsVect.clear();
      clsCounter=0;
    }
    ++clsCounter;
  }

  unsigned int n_tot  = 0; 
  unsigned int n_roll = 0; 
  uint32_t temp = 0; 
  std::vector<float> veff, vvnoise;
  veff.clear();
  vvnoise.clear();

  std::cout<<"size of vector of noise vectors :: vnoise.size() = "<<vnoise.size()<<std::endl;

  for(std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end(); ++it){

    detId = it->dpid;
    // std::cout<<"Loooking in theGeometry "<<theGeometry;
    // std::cout<<" for RPCDetId "<<detId;
    rpcId = RPCDetId(detId);
    // std::cout<<" aka "<<rpcId;
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(rpcId));
    // std::cout<<" roll "<<roll;

    if(roll !=0 ) {
      unsigned int numbStrips = roll->nstrips();
      // std::cout<<" with "<<numbStrips<<" strips"<<std::endl;
      // std::cout<<"Got RPCDetId "<<detId<<" aka "<<rpcId<<" with "<<numbStrips<<"strips"<<std::endl;

      // This dirty mess should be cleaned up
      // such that is is more clear to the reader
      // what is going on
      // -----------------------------------------
      if(n_roll%numbStrips == 0) {
	// std::cout<<"Got RPCDetId "<<detId<<" aka "<<rpcId<<" with "<<numbStrips<<"strips || n_tot = "<<n_tot<<" n_roll = "<<n_roll<<" => n_roll%numbStrips == 0"<<std::endl;
	if(n_tot > 0 ){
	  _mapDetIdNoise[temp]= vvnoise;
	  _mapDetIdEff[temp] = veff;
	  _bxmap[RPCDetId(it->dpid)] = it->time;
	  
	  veff.clear();
	  vvnoise.clear();
	  vvnoise.push_back((it->noise));
	  veff.push_back((it->eff));
	}
	else if(n_tot == 0 ){
	  vvnoise.push_back((it->noise));
	  veff.push_back((it->eff));
	  _bxmap[RPCDetId(it->dpid)] = it->time;
	}
	// std::cout<<" _mapDetIdNoise.size() = "<< _mapDetIdNoise.size() <<" _mapDetIdEff.size() = "<< _mapDetIdEff.size() <<" _bxmap.size() =  "<< _bxmap.size()<<std::endl;
      } 
      else if (n_tot == vnoise.size()-1 ){ // last element 
	temp = it->dpid;
	vvnoise.push_back((it->noise));
	veff.push_back((it->eff));
	_mapDetIdNoise[temp]= vvnoise;
	_mapDetIdEff[temp] = veff;
      } 
      else {
	temp = it->dpid;
	vvnoise.push_back((it->noise));
	veff.push_back((it->eff));
      }
      n_tot++;
      if(n_roll<numbStrips-1) ++n_roll;
      else n_roll = 0;
      // -----------------------------------------
    }
    // else std::cout<<std::endl;
  }
}


const std::vector<float>& RPCSimSetUp::getNoise(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdNoise.find(id);
  if(iter == _mapDetIdNoise.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no noise information for DetId\t"<<id<< std::endl;
  }
  return iter->second;
}

const std::vector<float>& RPCSimSetUp::getEff(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdEff.find(id);

  if(iter == _mapDetIdEff.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no efficiency information for DetId\t"<<id<< std::endl;
  }

  RPCDetId rpcId = RPCDetId(id);
  const RPCRoll* roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(rpcId));
  unsigned int numbStrips = roll->nstrips();

  if((iter->second).size() < numbStrips){
    // std::cout<< "Exception comming from RPCSimSetUp - efficiency information in a wrong format for DetId\t"<<id<<" aka "<<RPCDetId(id);
    // std::cout<<" number of strips in Conditions\t"<<(iter->second).size()<<" number of strips in Geometry\t"<<numbStrips<<std::endl;
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - efficiency information in a wrong format for DetId\t"<<id<< std::endl;
  }
  return iter->second;
}

float RPCSimSetUp::getTime(uint32_t id)
{
  RPCDetId rpcid(id);
  std::map<RPCDetId, float>::iterator iter = _bxmap.find(rpcid);
  if(iter == _bxmap.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no timing information for rpcid.rawId()\t"<<rpcid.rawId()<< std::endl;
  }
  return iter->second;
}

const std::map< int, std::vector<double> >& RPCSimSetUp::getClsMap()
{
  if(_clsMap.size()!=5){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - cluster size - a wrong format "<< std::endl;
  }
  return _clsMap;
}


//const std::map<int, std::vector<double> >& RPCSimSetUp::getClsMap(uint32_t id)
const std::vector<double>& RPCSimSetUp::getCls(uint32_t id)
{

  map<uint32_t,std::vector<double> >::iterator iter = _mapDetClsMap.find(id);
  if(iter == _mapDetClsMap.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 100){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - cluster size information in a wrong format for DetId\t"<<id<< std::endl;
  }
  return iter->second;
}

RPCSimSetUp::~RPCSimSetUp(){}
