#ifndef Tracker_SiTrivialInduceChargeOnStrips_H
#define Tracker_SiTrivialInduceChargeOnStrips_H
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/SiStripDigitizer/interface/SiInduceChargeOnStrips.h"
//
// first + 2* second +3* third .... = 1
//

//extern "C"   float freq_(const float& x);   

/**
 * Concrete implementation of SiInduceChargeOnStrips.
 * Charges are induces only in the first neighbours.
 */
class SiTrivialInduceChargeOnStrips: public SiInduceChargeOnStrips{
 public:
  
  SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g);
  virtual ~SiTrivialInduceChargeOnStrips() {}
  void induce(SiChargeCollectionDrifter::collection_type, const StripGeomDetUnit&, SiPileUpSignals::signal_map_type &, SiPileUpSignals::signal_map_type &);

 private:
  edm::ParameterSet conf_;
  bool peak;
  double clusterWidth;
  std::vector<double> signalCoupling; 
  std::vector<double> coupling_costant;
  double geVperElectron;
};


#endif
