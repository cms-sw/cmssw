#ifndef GEMDigitizer_GEMSimSetUp_h
#define GEMDigitizer_GEMSimSetUp_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include <map>
#include <vector>

class GEMGeometry;
class GEMDetId;

class GEMSimSetUp
{
 public:
  
  GEMSimSetUp(const edm::ParameterSet& ps);
  
  virtual ~GEMSimSetUp() {}

  void setup();

  void setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
             std::vector<float> &vcluster);

  void setup(std::vector<RPCStripNoises::NoiseItem> &vnoise,
             std::vector<RPCClusterSize::ClusterSizeItem> &vClusterSize);

  const std::vector<float>& getNoise(uint32_t id);
  
  const std::vector<float>& getEfficiency(uint32_t id);
  
  float getTime(uint32_t id);
  
  const std::map< int, std::vector<float> >& getClusterMap();
  
  const std::vector<float>& getCluster(uint32_t id);
  
  void setGeometry(const GEMGeometry * geom) {geometry_ = geom;}

  const GEMGeometry * getGeometry() { return geometry_; }

private:

  void setupNoiseAndEfficiency(std::vector<RPCStripNoises::NoiseItem> &vnoise);
  void setupTimeCalibration(GEMDetId id, float timing);

  const GEMGeometry * geometry_;

  std::map< uint32_t, std::vector<float> > mapDetIdNoise_;
  std::map< uint32_t, std::vector<float> > mapDetIdEfficiency_;
  std::map< GEMDetId, float> bxmap_;
  std::map< int, std::vector<float> > clusterMap_;
  std::map< uint32_t, std::vector<float> > mapDetClusterMap_;

  double averageEfficiency_;
  double averageNoiseRate_;
  double timeCalibrationOffset_;
  int numberOfStripsPerPartition_;
};

#endif
