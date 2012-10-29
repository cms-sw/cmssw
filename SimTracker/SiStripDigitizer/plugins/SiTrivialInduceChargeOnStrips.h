#ifndef Tracker_SiTrivialInduceChargeOnStrips_H
#define Tracker_SiTrivialInduceChargeOnStrips_H

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SiInduceChargeOnStrips.h"

class SiTrivialInduceChargeOnStrips: public SiInduceChargeOnStrips {
 public:
  SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g);
  virtual ~SiTrivialInduceChargeOnStrips() {}
  void  induce(const SiChargeCollectionDrifter::collection_type& collection_points,
	       const StripGeomDetUnit& det,
	       std::vector<double>& localAmplitudes,
	       size_t& recordMinAffectedStrip,
	       size_t& recordMaxAffectedStrip) const;
  
 private:
  double chargeDeposited(size_t strip, 
			 size_t Nstrips, 
			 double amplitude, 
			 double chargeSpread, 
			 double chargePosition) const;
  static unsigned int typeOf(const StripGeomDetUnit&);
  static unsigned int indexOf(const std::string&);
  static const std::string type[];
  static const int Ntypes;
  const std::vector<std::vector<double> > signalCoupling; 
  
  const double Nsigma;
  const double geVperElectron;
};

#endif
