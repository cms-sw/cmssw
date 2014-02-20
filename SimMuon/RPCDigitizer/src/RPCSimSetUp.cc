#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"

#include "SimMuon/RPCDigitizer/src/RPCDigiProducer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "SimMuon/RPCDigitizer/src/RPCSimSetUp.h"

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"





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

void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<float>& vcls){

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
    rpcId = RPCDetId(detId);
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(rpcId));

    if(roll !=0 ) {
      unsigned int numbStrips = roll->nstrips();
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
  }
}

void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<RPCClusterSize::ClusterSizeItem>& vClusterSize){

  // Old idea to determine how many strips there are foreseen for each roll
  // Depricated now since now the program is checking explicitly for the detid
  // -------------------------------------------------------------------------
  // Number of Rolls in this Geometry
  // const std::vector< RPCRoll * > & rollvector = theGeometry->rolls();
  // int nRolls = rollvector.size();
  // Number of Noise items in these noise vector
  // int nConds = vnoise.size();
  // Calculate here how many entries there are in the conditions for each roll
  // int nStrips = 0;
  // vnoise.size() should be an integer multiple of the amount of rolls
  // if(vnoise.size()%nRolls != 0){
  //   throw cms::Exception("DataCorrupt")
  //     << "Exception comming from RPCSimSetUp - Number of entries in Noise item is not an integer multiple of the number of rolls in this geometry\n"
  //    << "no of noise items = "<< nConds <<" no of rolls = "<<nRolls<<" no of noise items / no of rolls = "<< nConds/nRolls <<std::endl;
  // }
  // nStrips = nConds/nRolls;
  // -------------------------------------------------------------------------

  uint32_t detId, current_detId, this_detId;
  RPCDetId rpcId, current_rpcId, this_rpcId;
  const RPCRoll * current_roll,* this_roll;
  unsigned int current_nStrips;

  // ### ClusterSizeItem #######################################################
  std::vector<RPCClusterSize::ClusterSizeItem>::const_iterator itCls;
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
  // ###########################################################################


  // ### NoiseItem #############################################################
  unsigned int count_strips = 1;
  unsigned int count_all    = 1;
  std::vector<float> vveff, vvnoise;
  // vveff.clear();
  // vvnoise.clear();

  // DetId to start with
  current_detId   = vnoise.begin()->dpid;
  current_rpcId   = RPCDetId(current_detId);
  current_roll    = dynamic_cast<const RPCRoll* >(theGeometry->roll(current_rpcId));
  current_nStrips = current_roll->nstrips(); 

  LogDebug ("rpssimsetup") <<"Start Position ::            current_detId = "<<current_detId<<" aka "<<current_rpcId;
  LogDebug ("rpssimsetup") <<" is a valid roll with pointer "<<current_roll<<" and has "<<current_roll->nstrips()<<" strips"<<std::endl;
  LogDebug ("rpssimsetup") <<" ------------------------------------------------------------------------------------------------------------------------------------- "<<std::endl;
  for(std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end(); ++it) {

    // roll associated to the conditions of this strip (iterator)
    this_detId = it->dpid;
    this_rpcId = RPCDetId(this_detId);
    // Test whether this roll (picked up from the conditions) is inside the RPC Geometry 
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(this_rpcId));
    if(roll==0) {
      LogDebug ("rpssimsetup") <<"Inside Loop :: ["<<std::setw(6)<<count_all<<"]["<<std::setw(3)<<count_strips<<"] :: this_detId = "<<this_detId<<" aka "<<this_rpcId<<" which is not in current Geometry --> Skip "<<std::endl;
      continue;
    }

    // Case 1 :: FIRST ENTRY
    // ---------------------
    if(this_detId == current_detId && count_strips == 1) {

      LogDebug ("rpssimsetup") <<"Inside Loop :: ["<<std::setw(6)<<count_all<<"]["<<std::setw(3)<<count_strips<<"] :: this_detId = "<<this_detId<<" aka "<<this_rpcId<<" Noise = "<<it->noise<<" Hz/cm2";
      LogDebug ("rpssimsetup") <<" is a valid roll with pointer "<<roll<<" and has "<<roll->nstrips()<<" strips"<<std::endl;

      // fill bx in map
      _bxmap[current_detId] = it->time;
      // clear vectors
      vveff.clear();
      vvnoise.clear();
      // fill the vectors
      vvnoise.push_back((it->noise));
      vveff.push_back((it->eff));
      // update counter
      ++count_strips;
      ++count_all;
    }

    // Case 2 :: 2ND ENTRY --> LAST-1 ENTRY
    // ------------------------------------
    if(this_detId == current_detId && count_strips > 1 && count_strips < current_nStrips) {

      LogDebug ("rpssimsetup") <<"Inside Loop :: ["<<std::setw(6)<<count_all<<"]["<<std::setw(3)<<count_strips<<"] :: this_detId = "<<this_detId<<" aka "<<this_rpcId<<" Noise = "<<it->noise<<" Hz/cm2"<<std::endl;

      // fill the vectors
      vvnoise.push_back((it->noise));
      vveff.push_back((it->eff));
      // update counter
      ++count_strips;
      ++count_all;
    }

    // Case 3 :: LAST ENTRY
    // --------------------
    if(this_detId == current_detId && count_strips == current_nStrips) {
      // fill last value in the vector
      LogDebug ("rpssimsetup") <<"Last Value ::  ["<<std::setw(6)<<count_all<<"]["<<std::setw(3)<<count_strips<<"] :: this_detId = "<<this_detId<<" aka "<<this_rpcId<<" Noise = "<<it->noise<<" Hz/cm2";
      vvnoise.push_back((it->noise));
      vveff.push_back((it->eff));
      // update counter
      ++count_strips;
      ++count_all;
      // fill vectors into map
      LogDebug ("rpssimsetup") <<" fill vectors into map"<<std::endl;
      _mapDetIdNoise[current_detId]= vvnoise;
      _mapDetIdEff[current_detId] = vveff;
      // look for next different detId and rename it to the current_detId
      // at this point we skip all the conditions for the strips that are not in this roll
      // and we will go to the conditions for the first strip of the next roll 
      bool next_detId_found = 0;
      LogDebug ("rpssimsetup") <<"look for next different detId"<<std::endl;
      while(next_detId_found==0 && it != vnoise.end()-1) {
	++it; 
	this_detId = it->dpid;
	this_rpcId = RPCDetId(this_detId);
	this_roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(this_rpcId));
	if(!this_roll) continue;
	LogDebug ("rpssimsetup") <<"Inside While:: ["<<std::setw(6)<<count_all<<"]["<<std::setw(3)<<count_strips<<"] :: this_detId = "<<this_detId<<" aka "<<this_rpcId<<" Noise = "<<it->noise<<" Hz/cm2"<<std::endl;
	++count_strips;
	// ++count_all;
	if(this_detId != current_detId) {
	  LogDebug ("rpssimsetup") <<"Different detId is found ::                  "<<this_detId<<" aka "<<this_rpcId<<" Noise = "<<it->noise<<" Hz/cm2";
	  // next roll is found. update current_detId to this newly found detId 
	  // and update also the number of strips
	  current_detId = this_detId;
	  current_rpcId = RPCDetId(current_detId);
	  next_detId_found = 1;
	  current_nStrips = dynamic_cast<const RPCRoll* >(theGeometry->roll(current_rpcId))->nstrips();
	  LogDebug ("rpssimsetup") <<" with "<<current_nStrips<<" strips"<<std::endl;
	  --it; // subtract one, because at the end of the loop the iterator will be increased with one
	}
      }
      // reset count_strips
      count_strips = 1;
    }

      /*
      // This dirty mess should be cleaned up
      // such that is is more clear to the reader
      // what is going on
      // -----------------------------------------
      if(n_roll%numbStrips == 0) {
	if(n_tot > 792500) std::cout<<"Special Got RPCDetId "<<detId<<" aka "<<rpcId<<" with "<<numbStrips<<" strips || n_tot = "<<n_tot<<" n_roll = "<<n_roll<<" => n_roll%numbStrips == 0"<<std::endl;
	if(n_tot > 0 ){
	  _mapDetIdNoise[temp]= vvnoise;
	  _mapDetIdEff[temp] = vveff;
	  _bxmap[RPCDetId(it->dpid)] = it->time;
	  
	  vveff.clear();
	  vvnoise.clear();
	  vvnoise.push_back((it->noise));
	  vveff.push_back((it->eff));
	}
	else if(n_tot == 0 ){ // first element
	  vvnoise.push_back((it->noise));
	  vveff.push_back((it->eff));
	  _bxmap[RPCDetId(it->dpid)] = it->time;
	}
      } 
      else if (n_tot == vnoise.size()-1 ){ // last element 
	temp = it->dpid;
	vvnoise.push_back((it->noise));
	vveff.push_back((it->eff));
	_mapDetIdNoise[temp]= vvnoise;
	_mapDetIdEff[temp] = vveff;
      } 
      else {
	temp = it->dpid;
	vvnoise.push_back((it->noise));
	vveff.push_back((it->eff));
      }
      ++n_tot;
      if(n_roll<numbStrips-1) ++n_roll;
      else n_roll = 0;
      // -----------------------------------------
      */
  }
  // ###########################################################################
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
    std::cout<< "Exception comming from RPCSimSetUp - efficiency information in a wrong format for DetId\t"<<id<<" aka "<<RPCDetId(id)<<std::endl;
    std::cout<<" number of strips in Conditions\t"<<(iter->second).size()<<" number of strips in Geometry\t"<<numbStrips<<std::endl;
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
