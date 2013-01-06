#include "SiTrivialInduceChargeOnStrips.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include <Math/ProbFuncMathCore.h>
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <iostream>

const int
SiTrivialInduceChargeOnStrips::
Ntypes = 14;

const  std::string 
SiTrivialInduceChargeOnStrips::
type[Ntypes] = { "IB1", "IB2","OB1","OB2","W1a","W2a","W3a","W1b","W2b","W3b","W4","W5","W6","W7"};

static 
std::vector<std::vector<double> >
fillSignalCoupling(const edm::ParameterSet& conf, int nTypes, const std::string* typeArray) {
  std::vector<std::vector<double> > signalCoupling;
  signalCoupling.reserve(nTypes);
  std::string mode = conf.getParameter<bool>("APVpeakmode") ? "Peak" : "Dec";
  for(int i=0; i<nTypes; ++i) {
    signalCoupling.push_back(conf.getParameter<std::vector<double> >("CouplingConstant"+mode+typeArray[i]));
  }
  return signalCoupling;
}

inline unsigned int 
SiTrivialInduceChargeOnStrips::
indexOf(const std::string& t) { return std::find( type, type + Ntypes, t) - type;}

inline unsigned int
SiTrivialInduceChargeOnStrips::
typeOf(const StripGeomDetUnit& det, const TrackerTopology *tTopo) {
  DetId id = det.geographicalId();
  switch (det.specificType().subDetector()) {
  case GeomDetEnumerators::TIB: {return (tTopo->tibLayer(id) < 3) ? indexOf("IB1") : indexOf("IB2");}
  case GeomDetEnumerators::TOB: {return (tTopo->tobLayer(id) > 4) ? indexOf("OB1") : indexOf("OB2");}
  case GeomDetEnumerators::TID: {return indexOf("W1a") -1 + tTopo->tidRing(id);} //fragile: relies on ordering of 'type'
  case GeomDetEnumerators::TEC: {return indexOf("W1b") -1 + tTopo->tecRing(id);} //fragile: relies on ordering of 'type'
  default: throw cms::Exception("Invalid subdetector") << id();
  }
}

SiTrivialInduceChargeOnStrips::
SiTrivialInduceChargeOnStrips(const edm::ParameterSet& conf,double g) 
  : signalCoupling(fillSignalCoupling(conf, Ntypes, type)), Nsigma(3.), geVperElectron(g)  {
}

void 
SiTrivialInduceChargeOnStrips::
induce(const SiChargeCollectionDrifter::collection_type& collection_points, 
       const StripGeomDetUnit& det, 
       std::vector<double>& localAmplitudes, 
       size_t& recordMinAffectedStrip, 
       size_t& recordMaxAffectedStrip,
       const TrackerTopology *tTopo) const {

  const std::vector<double>& coupling = signalCoupling.at(typeOf(det,tTopo));
  const StripTopology& topology = dynamic_cast<const StripTopology&>(det.specificTopology());
  size_t Nstrips =  topology.nstrips();

  for (SiChargeCollectionDrifter::collection_type::const_iterator 
	 signalpoint = collection_points.begin();  signalpoint != collection_points.end();  signalpoint++ ) {
    
    //In strip coordinates:
    double chargePosition = topology.strip(signalpoint->position());
    double chargeSpread = signalpoint->sigma() / topology.localPitch(signalpoint->position());
    
    size_t fromStrip  = size_t(std::max( 0,          int(std::floor( chargePosition - Nsigma*chargeSpread))));
    size_t untilStrip = size_t(std::min( Nstrips, size_t(std::ceil( chargePosition + Nsigma*chargeSpread) )));
    for (size_t strip = fromStrip;  strip < untilStrip; strip++) {

      double chargeDepositedOnStrip = chargeDeposited( strip, Nstrips, signalpoint->amplitude(), chargeSpread, chargePosition);

      size_t affectedFromStrip  = size_t(std::max( 0, int(strip - coupling.size() + 1)));
      size_t affectedUntilStrip = size_t(std::min( Nstrips, strip + coupling.size())   );  
      for (size_t affectedStrip = affectedFromStrip;  affectedStrip < affectedUntilStrip;  affectedStrip++) {
	localAmplitudes.at( affectedStrip ) += chargeDepositedOnStrip * coupling.at(abs( affectedStrip - strip )) ;
      }

      if( affectedFromStrip  < recordMinAffectedStrip ) recordMinAffectedStrip = affectedFromStrip;
      if( affectedUntilStrip > recordMaxAffectedStrip ) recordMaxAffectedStrip = affectedUntilStrip;
    }
  }
  return;
}

inline double
SiTrivialInduceChargeOnStrips::
chargeDeposited(size_t strip, size_t Nstrips, double amplitude, double chargeSpread, double chargePosition) const {
  double integralUpToStrip = (strip == 0)         ? 0. : ( ROOT::Math::normal_cdf(   strip, chargeSpread, chargePosition) );
  double integralUpToNext  = (strip+1 == Nstrips) ? 1. : ( ROOT::Math::normal_cdf( strip+1, chargeSpread, chargePosition) );
  double percentOfSignal = integralUpToNext - integralUpToStrip;
  
  return percentOfSignal * amplitude / geVperElectron;
}
