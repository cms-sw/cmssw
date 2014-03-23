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
  if((iter->second).size() != 96){
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

  map<uint32_t,std::vector<double> >::iterator iter = _mapDetClsMapLegacy.find(id);
  if(iter == _mapDetClsMapLegacy.end()){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 100){
    throw cms::Exception("DataCorrupt") 
      << "Exception comming from RPCSimSetUp - cluster size information in a wrong format for DetId\t"<<id<< std::endl;
  }
  return iter->second;
}

const std::vector<double> & RPCSimSetUp::getAsymmetricClsDistribution(uint32_t id, uint32_t slice){

 map<uint32_t,std::vector<double> >::const_iterator iter = _mapDetClsMap.find(id);
  if(iter == _mapDetClsMap.end()){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 120){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - cluster size information in a wrong format for DetId\t"<<id<< std::endl;
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

 map<uint32_t,std::vector<double> >::const_iterator iter = _mapDetClsMap.find(id);
  if(iter == _mapDetClsMap.end()){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - no cluster size information for DetId\t"<<id<< std::endl;
  }
  if((iter->second).size() != 120){
    throw cms::Exception("DataCorrupt")
      << "Exception comming from RPCSimSetUp - cluster size information in a wrong format for DetId\t"<<id<<'\t'<<(iter->second).size()<< std::endl;
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
    //     std::cout<<"value\t"<<value<<std::endl;
     //    std::cout<<"sum\t"<<sum<<std::endl;
  }
  
  float accum=0;
  for(int i = clsDetAsymmetryForCls.size()-1; i>-1; i--){
    accum += clsDetAsymmetryForCls[i];
    _DetAsymmetryForCls.push_back(accum/sum);
  }
return  _DetAsymmetryForCls;
}

RPCSimSetUp::~RPCSimSetUp(){}
