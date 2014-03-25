#ifndef RPCSimSetUp_h
#define RPCSimSetUp_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CondFormats/DataRecord/interface/RPCClusterSizeRcd.h"

#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include<cstring>
#include<string>
#include<stdlib.h>
#include <utility>

class RPCDigitizer;
class RPCGeometry;
class RPCDetId;

class RPCSimSetUp
{
public:

  explicit RPCSimSetUp(const edm::ParameterSet& ps);
  virtual ~RPCSimSetUp();

  void setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<float>& vcls);

  void setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>& vnoise, const std::vector<RPCClusterSize::ClusterSizeItem>& vClusterSize);

  const std::vector<float>& getNoise(uint32_t id);
  const std::vector<float>& getEff(uint32_t id);
  float getTime(uint32_t id);
  const std::map< int, std::vector<double> >& getClsMap();
  //  const std::map<int, std::vector<double> >& getClsMap(uint32_t id);
  const std::vector<double>& getCls(uint32_t id);
 const std::vector<double> & getAsymmetricClsDistribution(uint32_t id, uint32_t slice);
 const std::vector<double> & getAsymmetryForCls(uint32_t id, uint32_t slice, uint32_t cls);
  /// sets geometry
  void setGeometry(const RPCGeometry * geom) {theGeometry = geom;}

  const RPCGeometry * getGeometry() { return theGeometry; }

  const RPCGeometry * theGeometry;

  std::map<uint32_t, std::vector<float> > _mapDetIdNoise;
  std::map<uint32_t, std::vector<float> > _mapDetIdEff;
  std::map<RPCDetId, float> _bxmap;
  std::map< int, std::vector<double> > _clsMap;
  std::map<uint32_t, std::vector<double> > _mapDetClsMap;
  std::map<uint32_t, std::vector<double> > _mapDetClsMapLegacy; //allowing old formad to by used by old digitizers
  std::vector<double>  _DetClsAsymmetric;
  std::vector<double> _DetAsymmetryForCls;
private:
};

#endif
