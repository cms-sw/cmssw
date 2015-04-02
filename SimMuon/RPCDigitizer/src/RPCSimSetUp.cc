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

  unsigned int n = 0; 
  uint32_t temp = 0; 
  std::vector<float> veff, vvnoise;
  veff.clear();
  vvnoise.clear();

  for(std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end(); ++it){
    if(n%96 == 0) {
      if(n > 0 ){
	_mapDetIdNoise[temp]= vvnoise;
	_mapDetIdEff[temp] = veff;
	_bxmap[RPCDetId(it->dpid)] = it->time;
	
	veff.clear();
	vvnoise.clear();
	vvnoise.push_back((it->noise));
	veff.push_back((it->eff));
      }
      else if(n == 0 ){
	vvnoise.push_back((it->noise));
	veff.push_back((it->eff));
	_bxmap[RPCDetId(it->dpid)] = it->time;
      }
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

// Previous version with hardcoded limit of max 96 strips per roll
/*
void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<RPCClusterSize::ClusterSizeItem>& vClusterSize){

  std::vector<RPCClusterSize::ClusterSizeItem>::const_iterator itCls;
  uint32_t  detId;
  int clsCounter(1);
  std::vector<double> clsVect;

  for(itCls = vClusterSize.begin(); itCls != vClusterSize.end(); ++itCls){
    clsVect.push_back(((double)(itCls->clusterSize)));
    if((!(clsCounter%120)) && (clsCounter!=0)){
      detId=itCls->dpid;
      _mapDetClsMap[detId]=clsVect;
      clsVect.clear();
      clsCounter=0;
    }
    ++clsCounter;
  }

  // the same loop (but till 100) to allow old format to be used
  for(itCls = vClusterSize.begin(); itCls != vClusterSize.end(); ++itCls){
    clsVect.push_back(((double)(itCls->clusterSize)));
    if((!(clsCounter%100)) && (clsCounter!=0)){
      detId=itCls->dpid;
      _mapDetClsMapLegacy[detId]=clsVect;
      clsVect.clear();
      clsCounter=0;
    }
    ++clsCounter;
  }

  unsigned int n = 0; 
  uint32_t temp = 0; 
  std::vector<float> veff, vvnoise;
  veff.clear();
  vvnoise.clear();

  for(std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end(); ++it){
    if(n%96 == 0) {
      if(n > 0 ){
	_mapDetIdNoise[temp]= vvnoise;
	_mapDetIdEff[temp] = veff;
	_bxmap[RPCDetId(it->dpid)] = it->time;
	
	veff.clear();
	vvnoise.clear();
	vvnoise.push_back((it->noise));
	veff.push_back((it->eff));
      }
      else if(n == 0 ){
	vvnoise.push_back((it->noise));
	veff.push_back((it->eff));
	_bxmap[RPCDetId(it->dpid)] = it->time;
      }
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
*/

// New version that can deal with rolls with an arbitrary number of strips, no hardcoded value 96 here
void RPCSimSetUp::setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<RPCClusterSize::ClusterSizeItem>& vClusterSize){

  LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp(vector<NoiseItem>, vector<ClusterSizeItem>)"<<std::endl;

  uint32_t detId, current_detId, this_detId;
  RPCDetId rpcId, current_rpcId, this_rpcId;
  const RPCRoll * current_roll,* this_roll;
  unsigned int current_nStrips;

  LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: ClusterSizeItem :: begin"<<std::endl;
  // ### ClusterSizeItem #######################################################                        
  std::vector<RPCClusterSize::ClusterSizeItem>::const_iterator itCls;
  int clsCounter(1);
  std::vector<double> clsVect;
  // ### loop for New Format (120 entries)
  for(itCls = vClusterSize.begin(); itCls != vClusterSize.end(); ++itCls){
    // LogDebug ("rpssimsetup")<<" Push back clustersize = "<<itCls->clusterSize<<std::endl;
    clsVect.push_back(((double)(itCls->clusterSize)));
    // LogDebug ("rpssimsetup")<<"Filling cls in _mapDetCls[detId,clsVect] :: detId = "<<detId;
    // LogDebug ("rpssimsetup")<<" --> will it be accepted? clsCounter = "<<clsCounter<<" accepted?";
    // LogDebug ("rpssimsetup")<<" New Format ::"<<((!(clsCounter%120)) && (clsCounter!=0)); // <<std::endl;
    // LogDebug ("rpssimsetup")<<" Old Format ::"<<((!(clsCounter%100)) && (clsCounter!=0)); // <<std::endl;
    // LogDebug ("rpssimsetup")<<std::endl;

    // New Format :: loop until 120
    if((!(clsCounter%120)) && (clsCounter!=0)){
      detId=itCls->dpid;
      _mapDetClsMap[detId]=clsVect;
      clsVect.clear();
      // LogDebug ("rpssimsetup")<<" --> New Method ";
      // LogDebug ("rpssimsetup")<<" --> saved in map "<<std::endl;
      // LogDebug ("rpssimsetup")<<"Filling cls in _mapDetClsMap[detId,clsVect] :: detId = "<<detId;
      // LogDebug ("rpssimsetup")<<" --> will it be accepted? clsCounter = "<<clsCounter<<" accepted? "<<((!(clsCounter%120)) && (clsCounter!=0))<<std::endl;
      clsCounter=0;
    }
    /*else{
      LogDebug ("rpssimsetup")<<" --> not saved in map "<<std::endl;
    }*/
    ++clsCounter;
  }
  // ### loop for Old Format (100 entries)
  for(itCls = vClusterSize.begin(); itCls != vClusterSize.end(); ++itCls){
    // LogDebug ("rpssimsetup")<<" Push back clustersize = "<<itCls->clusterSize<<std::endl;
    clsVect.push_back(((double)(itCls->clusterSize)));
    // LogDebug ("rpssimsetup")<<"Filling cls in _mapDetClsMapLegacy[detId,clsVect] :: detId = "<<detId;
    // LogDebug ("rpssimsetup")<<" --> will it be accepted? clsCounter = "<<clsCounter<<" accepted?";
    // LogDebug ("rpssimsetup")<<" New Format ::"<<((!(clsCounter%120)) && (clsCounter!=0)); // <<std::endl;
    // LogDebug ("rpssimsetup")<<" Old Format ::"<<((!(clsCounter%100)) && (clsCounter!=0)); // <<std::endl;
    // LogDebug ("rpssimsetup")<<std::endl;

    // Old Format :: same until 100
    if((!(clsCounter%100)) && (clsCounter!=0)){
      detId=itCls->dpid;
      _mapDetClsMapLegacy[detId]=clsVect;
      clsVect.clear();
      // LogDebug ("rpssimsetup")<<" --> Old Method ";
      // LogDebug ("rpssimsetup")<<" --> saved in map "<<std::endl;
      // LogDebug ("rpssimsetup")<<"Filling cls in _mapDetClsMapLegacy[detId,clsVect] :: detId = "<<detId;
      // LogDebug ("rpssimsetup")<<" --> will it be accepted? clsCounter = "<<clsCounter<<" accepted? "<<((!(clsCounter%120)) && (clsCounter!=0))<<std::endl;
      clsCounter=0;
    }
    /*else{
      LogDebug ("rpssimsetup")<<" --> not saved in map "<<std::endl;
      } */
    ++clsCounter;
  }
  // ###########################################################################
  LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: ClusterSizeItem :: end"<<std::endl;

  LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: NoiseItem :: begin"<<std::endl;
  // ### NoiseItem #############################################################
  unsigned int count_strips = 1;
  unsigned int count_all    = 1;
  std::vector<float> vveff, vvnoise;
  // vveff.clear();
  // vvnoise.clear(); 

  // DetId to start with needs to be a DetId inside the Geometry used
  // Therefore loop on the NoiseItems and search for the first valid roll in the Geometry
  // Assign this as the DetId to start with (so called current_roll) and quit the loop
  bool quitLoop = false;
  current_detId = 0; current_nStrips = 0; // current_rpcId = 0; current_roll = 0;
  for(std::vector<RPCStripNoises::NoiseItem>::const_iterator it = vnoise.begin(); it != vnoise.end() && !quitLoop; ++it) {
    // roll associated to the conditions of this strip (iterator)
    current_detId = it->dpid;
    current_rpcId = RPCDetId(current_detId);
    // Test whether this roll (picked up from the conditions) is inside the RPC Geometry
    const RPCRoll* roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(current_rpcId));
    if(roll==0) {
      LogDebug ("rpssimsetup") <<"Searching for first valid detid :: current_detId = "<<current_detId;
      LogDebug ("rpssimsetup") <<" aka "<<current_rpcId<<" is not in current Geometry --> Skip "<<std::endl;
      continue;
    }
    else {
      LogDebug ("rpssimsetup") <<"Searching for first valid detid :: current_detId = "<<current_detId;
      LogDebug ("rpssimsetup") <<" aka "<<current_rpcId<<" is the first (valid) roll in the current Geometry --> Accept, Assign & Quit Loop"<<std::endl;
      current_roll    = dynamic_cast<const RPCRoll* >(theGeometry->roll(current_rpcId));
      current_nStrips = current_roll->nstrips();
      quitLoop = true;
    }
  }

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

    LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: NoiseItem :: case 1"<<std::endl;
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

    LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: NoiseItem :: case 2"<<std::endl;
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

    LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: NoiseItem :: case 3"<<std::endl;
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
	  // in fact the treatment for roll N stops when we find the first occurence of roll N+1
	  // however we want to start the treatment for roll N+1 with the first occurence of roll N+1
	  // so the first entry of each new roll N+1 is manipulated twice in the loop (once as a stop, once as a start)
	  // therefore we have to manipulate the iterator here, subtracting one, to treat again this entry
        }
      }
      // reset count_strips
      count_strips = 1;
    }
  }
  // ###########################################################################
  LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: NoiseItem :: end"<<std::endl;

  LogDebug ("rpssimsetup")<<"RPCSimSetUp::setRPCSetUp :: end"<<std::endl;
}


const std::vector<float>& RPCSimSetUp::getNoise(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdNoise.find(id);
  if(iter == _mapDetIdNoise.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no noise information for DetId\t"<<id<< std::endl;
  }
  LogDebug ("rpssimsetup")<< "All OK coming from RPCSimSetUp - noise information for DetId\t"<<id<< std::endl;
  return iter->second;
}

const std::vector<float>& RPCSimSetUp::getEff(uint32_t id)
{
  map<uint32_t,std::vector<float> >::iterator iter = _mapDetIdEff.find(id);
  if(iter == _mapDetIdEff.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no efficiency information for DetId\t"<<id<< std::endl;
  }
  /*
  if((iter->second).size() != 96){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - efficiency information in a wrong format for DetId\t"<<id<< std::endl;
  }
  */
  RPCDetId rpcId = RPCDetId(id);
  const RPCRoll* roll = dynamic_cast<const RPCRoll* >(theGeometry->roll(rpcId));
  unsigned int numbStrips = roll->nstrips();

  if((iter->second).size() < numbStrips){
    // LogDebug ("rpssimsetup")<< "Exception comming from RPCSimSetUp - efficiency information in a wrong format for DetId\t"<<id<<" aka "<<RPCDetId(id)<<std::endl;
    // LogDebug ("rpssimsetup")<<" number of strips in Conditions\t"<<(iter->second).size()<<" number of strips in Geometry\t"<<numbStrips<<std::endl;
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
const std::vector<double>& RPCSimSetUp::getCls(uint32_t id) //legacy member function
{

  LogDebug ("rpssimsetup")<<"RPCSimSetUp::getCls"<<std::endl;

  map<uint32_t,std::vector<double> >::iterator iter = _mapDetClsMapLegacy.find(id);
  if(iter == _mapDetClsMapLegacy.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 100){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - _mapDetClsMapLegacy - cluster size information in a wrong format for DetId\t"<<id<< std::endl;
  }
  LogDebug ("rpssimsetup")<< "All OK coming from RPCSimSetUp - _mapDetClsMapLegacy - cluster size information for DetId\t"<<id<< std::endl;
  return iter->second;
}

const std::vector<double> & RPCSimSetUp::getAsymmetricClsDistribution(uint32_t id, uint32_t slice){

  LogDebug ("rpssimsetup")<<"RPCSimSetUp::getAsymmetricClsDistribution"<<std::endl;

  map<uint32_t,std::vector<double> >::const_iterator iter = _mapDetClsMap.find(id);
  if(iter == _mapDetClsMap.end()){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - _mapDetClsMap - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 120){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - _mapDetClsMap - cluster size information in a wrong format for DetId\t"<<id<< std::endl;
  }
  //  return iter->second;

  std::vector<double> dataForAsymmCls = iter->second;
  if(slice>4){ 
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - slice variable not in the range"<< std::endl;
  }     

  _DetClsAsymmetric.clear();
   

  vector<double> clsFewStripsDistribution;
  vector<double> clsDistribution;
  vector<double> clsAccumulativeDistribution;
  
  std::map< int, std::vector<double> > mapSliceVsDistribution;

  const int slices=5;
  const int distributionFewStrips=24;

  double sliceVsFewStripsDistribution[slices][distributionFewStrips];
    
  for(int j = 0; j < distributionFewStrips; j++){
    for(int i = 0; i < slices; i++){
      sliceVsFewStripsDistribution[i][j]=dataForAsymmCls[j*slices+i];
    }
  }

double control=0;
for(int j = 0 ; j < distributionFewStrips; j++){
control+=sliceVsFewStripsDistribution[0][j];
} 

double control1=0;
  for(int j = 0; j < distributionFewStrips; j++){
    for(int i = 0; i < slices; i++){
      control1+=dataForAsymmCls[j*slices+i];
    }
  }

  int i = slice;
  double sum=0;
  int counter = 0 ;
  for(int j = 0; j < distributionFewStrips; j++){
    counter++;
    sum+=sliceVsFewStripsDistribution[i][j];
    if(counter%4==0){
      _DetClsAsymmetric.push_back(sum);
    }
  }
  return _DetClsAsymmetric;
}

const std::vector<double> & RPCSimSetUp::getAsymmetryForCls(uint32_t id, uint32_t slice, uint32_t cls){

  LogDebug ("rpssimsetup")<<"RPCSimSetUp::getAsymmetryForCls"<<std::endl;

 map<uint32_t,std::vector<double> >::const_iterator iter = _mapDetClsMap.find(id);
  if(iter == _mapDetClsMap.end()){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - _mapDetClsMap - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 120){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - _mapDetClsMap - cluster size information in a wrong format for DetId\t"<<id<<'\t'<<(iter->second).size()<< std::endl;
  }

std::vector<double> dataForAsymmCls = iter->second;

if(slice>4){ 
     throw cms::Exception("DataCorrupt") 
       << "Exception comming from RPCSimSetUp - slice variable not in the range"<< std::endl;
   }   

  _DetAsymmetryForCls.clear();

 vector<double> clsFewStripsDistribution;
  vector<double> clsDistribution;
  vector<double> clsAccumulativeDistribution;
vector<double> clsDetAsymmetryForCls;
 clsDetAsymmetryForCls.clear();
  
  std::map< int, std::vector<double> > mapSliceVsDistribution;

  const int slices=5;
  const int distributionFewStrips=24;

  double sliceVsFewStripsDistribution[slices][distributionFewStrips];
    
  for(int j = 0; j < distributionFewStrips; j++){
    for(int i = 0; i < slices; i++){
      sliceVsFewStripsDistribution[i][j]=dataForAsymmCls[j*slices+i];
    }
  }

  int vector_lenght;
  switch(cls){
  case 1:  case 3:  case 5:   vector_lenght =3; break;
  case 2:  case 4:            vector_lenght =4; break;
  case 6:  default:           vector_lenght = 1; break;
  }
  
  float sum=0;
  float value;
  for(int i = 0; i < vector_lenght ; i ++){
    value = sliceVsFewStripsDistribution[slice][(cls-1)*4+i];
    clsDetAsymmetryForCls.push_back(value);
    sum +=value; 
    //     LogDebug ("rpssimsetup")<<"value\t"<<value<<std::endl;
    //    LogDebug ("rpssimsetup")<<"sum\t"<<sum<<std::endl;
  }
  
  float accum=0;
  for(int i = clsDetAsymmetryForCls.size()-1; i>-1; i--){
    accum += clsDetAsymmetryForCls[i];
    _DetAsymmetryForCls.push_back(accum/sum);
  }
return  _DetAsymmetryForCls;
}

RPCSimSetUp::~RPCSimSetUp(){}
