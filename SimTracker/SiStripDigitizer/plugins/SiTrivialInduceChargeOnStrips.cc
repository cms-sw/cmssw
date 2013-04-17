#include "SiTrivialInduceChargeOnStrips.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetType.h"
#include "DataFormats/Math/interface/approx_erf.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <iostream>

namespace {
  struct Count {
   double ncall=0;
   double ndep=0, ndep2=0;
   double nstr=0, nstr2=0;
   void dep(int d) { ncall++; ndep+=d; ndep2+=d*d;}
   void	str(int d) { nstr+=d; nstr2+=d*d;}

   ~Count() {
     std::cout << "deposits " << ncall << " " << ndep/ncall << " " << (ndep2*ncall -ndep*ndep)/(ncall*ncall) << std::endl;
     std::cout << "strips  " << nstr/ndep << " " << (nstr2*ndep -nstr*nstr)/(ndep*ndep) << std::endl;
   }
  };

 Count count;
}


namespace {
  constexpr int Ntypes = 14;

  const std::string type[Ntypes] = { "IB1", "IB2","OB1","OB2","W1a","W2a","W3a","W1b","W2b","W3b","W4","W5","W6","W7"};

  inline
  std::vector<std::vector<float> >
  fillSignalCoupling(const edm::ParameterSet& conf, int nTypes, const std::string* typeArray) {
    std::vector<std::vector<float> > signalCoupling;
    signalCoupling.reserve(nTypes);
    std::string mode = conf.getParameter<bool>("APVpeakmode") ? "Peak" : "Dec";
    for(int i=0; i<nTypes; ++i) {
      auto dc = conf.getParameter<std::vector<double> >("CouplingConstant"+mode+typeArray[i]);
      signalCoupling.emplace_back(dc.begin(),dc.end());
    }
    return signalCoupling;
  }

  inline unsigned int indexOf(const std::string& t) { return std::find( type, type + Ntypes, t) - type;}


  inline unsigned int typeOf(const StripGeomDetUnit& det, const TrackerTopology *tTopo) {
    DetId id = det.geographicalId();
    switch (det.specificType().subDetector()) {
    case GeomDetEnumerators::TIB: {return (tTopo->tibLayer(id) < 3) ? indexOf("IB1") : indexOf("IB2");}
    case GeomDetEnumerators::TOB: {return (tTopo->tobLayer(id) > 4) ? indexOf("OB1") : indexOf("OB2");}
    case GeomDetEnumerators::TID: {return indexOf("W1a") -1 + tTopo->tidRing(id);} //fragile: relies on ordering of 'type'
    case GeomDetEnumerators::TEC: {return indexOf("W1b") -1 + tTopo->tecRing(id);} //fragile: relies on ordering of 'type'
    default: throw cms::Exception("Invalid subdetector") << id();
    }
  }

  inline double
  chargeDeposited(size_t strip, size_t Nstrips, double amplitude, double chargeSpread, double chargePosition)  {
    double integralUpToStrip = (strip == 0)         ? 0. : ( approx_erf(   (strip-chargePosition)/chargeSpread/1.41421356237309515 ) );
    double integralUpToNext  = (strip+1 == Nstrips) ? 1. : ( approx_erf(   (strip+1-chargePosition)/chargeSpread/1.41421356237309515 ) );
    
    return  0.5 * (integralUpToNext - integralUpToStrip) * amplitude;
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
       std::vector<float>& localAmplitudes, 
       size_t& recordMinAffectedStrip, 
       size_t& recordMaxAffectedStrip,
       const TrackerTopology *tTopo) const {

  auto const & coupling = signalCoupling[typeOf(det,tTopo)];
  const StripTopology& topology = dynamic_cast<const StripTopology&>(det.specificTopology());
  size_t Nstrips =  topology.nstrips();

  if (!collection_points.empty()) count.dep(collection_points.size());

  for (SiChargeCollectionDrifter::collection_type::const_iterator 
	 signalpoint = collection_points.begin();  signalpoint != collection_points.end();  signalpoint++ ) {
    
    //In strip coordinates:
    double chargePosition = topology.strip(signalpoint->position());
    double chargeSpread = signalpoint->sigma() / topology.localPitch(signalpoint->position());
    
    size_t fromStrip  = size_t(std::max( 0,          int(std::floor( chargePosition - Nsigma*chargeSpread))));
    size_t untilStrip = size_t(std::min( Nstrips, size_t(std::ceil( chargePosition + Nsigma*chargeSpread) )));

    count.str(std::max(0,int(untilStrip)-int(fromStrip)));
    for (size_t strip = fromStrip;  strip < untilStrip; strip++) {

      double chargeDepositedOnStrip = chargeDeposited( strip, Nstrips, signalpoint->amplitude() / geVperElectron, chargeSpread, chargePosition);

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

