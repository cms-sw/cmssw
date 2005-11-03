#ifndef Tracker_SiTrivialInduceChargeOnStrips_H
#define Tracker_SiTrivialInduceChargeOnStrips_H

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
  
  SiTrivialInduceChargeOnStrips(double g){clusterWidth=3.; geVperElectron = g;}
  SiInduceChargeOnStrips::hit_map_type induce(SiChargeCollectionDrifter::collection_type, const StripGeomDetUnit&);

 private:
  double clusterWidth;
  //   static ConfigurableVector<float> signalCoupling; 
  vector<float> signalCoupling; // AG
  double geVperElectron;
};


#endif
