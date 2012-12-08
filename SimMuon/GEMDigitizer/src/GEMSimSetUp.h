#ifndef GEMDigitizer_GEMSimSetUp_h
#define GEMDigitizer_GEMSimSetUp_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"

#include <map>
#include <vector>

class GEMGeometry;
class GEMDetId;

class GEMSimSetUp
{
public:

  explicit GEMSimSetUp(const edm::ParameterSet& ps) {}
  
  virtual ~GEMSimSetUp() {}

  void setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
             std::vector<float> &vcls);

  void setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
             std::vector<RPCClusterSize::ClusterSizeItem> &vClusterSize);

  const std::vector<float>& getNoise(uint32_t id);
  
  const std::vector<float>& getEff(uint32_t id);
  
  float getTime(uint32_t id);
  
  const std::map< int, std::vector<float> >& getClsMap();
  
  const std::vector<float>& getCls(uint32_t id);
  
  /// sets geometry
  void setGeometry(const GEMGeometry * geom) {geometry_ = geom;}

  const GEMGeometry * getGeometry() { return geometry_; }

private:

  void setupNoise(std::vector<RPCStripNoises::NoiseItem> &vnoise);

  const GEMGeometry * geometry_;

  std::map< uint32_t, std::vector<float> > mapDetIdNoise_;
  std::map< uint32_t, std::vector<float> > mapDetIdEff_;
  std::map< GEMDetId, float> bxmap_;
  std::map< int, std::vector<float> > clsMap_;
  std::map< uint32_t, std::vector<float> > mapDetClsMap_;
};

#endif
