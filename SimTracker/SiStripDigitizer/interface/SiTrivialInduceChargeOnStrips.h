#ifndef Tracker_SiTrivialInduceChargeOnStrips_H
#define Tracker_SiTrivialInduceChargeOnStrips_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimTracker/SiStripDigitizer/interface/SiInduceChargeOnStrips.h"

class SiTrivialInduceChargeOnStrips: public SiInduceChargeOnStrips {
 public:
  SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g);
  virtual ~SiTrivialInduceChargeOnStrips() {}
  void  induce(SiChargeCollectionDrifter::collection_type collection_points,
	       const StripGeomDetUnit& det,
	       std::vector<double>& localAmplitudes,
	       unsigned int& recordMinAffectedStrip,
	       unsigned int& recordMaxAffectedStrip);
  
 private:
  double chargeDeposited(uint16_t strip, 
			 uint16_t Nstrips, 
			 double amplitude, 
			 double chargeSpread, 
			 double chargePosition) const;
  static uint16_t typeOf(const StripGeomDetUnit&);
  static uint16_t indexOf(const std::string&);
  static const std::string type[];
  static const int Ntypes;
  std::vector<std::vector<double> > signalCoupling; 
  
  const double Nsigma;
  const double geVperElectron;
};

#endif
