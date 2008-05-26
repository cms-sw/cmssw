#ifndef Tracker_SiTrivialInduceChargeOnStrips_H
#define Tracker_SiTrivialInduceChargeOnStrips_H
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/SiStripDigitizer/interface/SiInduceChargeOnStrips.h"
/**
 * Concrete implementation of SiInduceChargeOnStrips.
 */

class SiTrivialInduceChargeOnStrips: public SiInduceChargeOnStrips{
public:
  
  SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g);
  virtual ~SiTrivialInduceChargeOnStrips() {}
  void induce(SiChargeCollectionDrifter::collection_type, const StripGeomDetUnit&,
	      std::vector<double>&, unsigned int&, unsigned int&);
  
private:
  edm::ParameterSet conf_;
  bool peak;
  double clusterWidth;
  std::vector<double> signalCoupling_TIB; 
  std::vector<double> signalCoupling_TID; 
  std::vector<double> signalCoupling_TOB; 
  std::vector<double> signalCoupling_TEC; 
  double geVperElectron;
};


#endif
