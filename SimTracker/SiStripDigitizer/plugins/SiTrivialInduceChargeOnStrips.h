#ifndef Tracker_SiTrivialInduceChargeOnStrips_H
#define Tracker_SiTrivialInduceChargeOnStrips_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SiInduceChargeOnStrips.h"

class TrackerTopology;

class SiTrivialInduceChargeOnStrips: public SiInduceChargeOnStrips {
 public:
  SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g);
  virtual ~SiTrivialInduceChargeOnStrips() {}
  void  induce(const SiChargeCollectionDrifter::collection_type& collection_points,
	       const StripGeomDetUnit& det,
	       std::vector<float>& localAmplitudes,
	       size_t& recordMinAffectedStrip,
	       size_t& recordMaxAffectedStrip,
	       const TrackerTopology *tTopo) const;
  
 private:
  const std::vector<std::vector<float> > signalCoupling; 
  
  const float Nsigma;
  const float geVperElectron;
};

#endif
