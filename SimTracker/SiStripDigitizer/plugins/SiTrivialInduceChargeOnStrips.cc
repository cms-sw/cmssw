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
    double ndep=0, ndep2=0, maxdep=0;
    double nstr=0, nstr2=0;
    double ncv=0, nval=0, nval2=0, maxv=0;
    double dzero=0;
    void dep(double d) { ncall++; ndep+=d; ndep2+=d*d;maxdep=std::max(d,maxdep);}
    void str(double d) { nstr+=d; nstr2+=d*d;}
    void val(double d) { ncv++; nval+=d; nval2+=d*d; maxv=std::max(d,maxv);}
    void zero() { dzero++;}    
    ~Count() {
#ifdef SISTRIP_COUNT
      std::cout << "deposits " << ncall << " " << maxdep << " " << ndep/ncall << " " << std::sqrt(ndep2*ncall -ndep*ndep)/ncall << std::endl;
      std::cout << "zeros " << dzero << std::endl;
      std::cout << "strips  " << nstr/ndep << " " << std::sqrt(nstr2*ndep -nstr*nstr)/ndep << std::endl;
      std::cout << "vaules  " << ncv << " " << maxv << " " << nval/ncv << " " << std::sqrt(nval2*ncv -nval*nval)/ncv << std::endl;
#endif
    }
  };
  
 Count count;
}


namespace {
  constexpr int Ntypes = 14;
  //                                   0     1      2     3     4    5      6     7
  const std::string type[Ntypes] = { "IB1", "IB2","OB1","OB2","W1a","W2a","W3a","W1b","W2b","W3b","W4","W5","W6","W7"};
  enum { indexOfIB1=0, indexOfIB2=1,  indexOfOB1=2, indexOfOB2=3, indexOfW1a=4, indexOfW1b=7}; 

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
    case GeomDetEnumerators::TIB: {return (tTopo->tibLayer(id) < 3) ? indexOfIB1 : indexOfIB2;}
    case GeomDetEnumerators::TOB: {return (tTopo->tobLayer(id) > 4) ? indexOfOB1 : indexOfOB2;}
    case GeomDetEnumerators::TID: {return indexOfW1a -1 + tTopo->tidRing(id);} //fragile: relies on ordering of 'type'
    case GeomDetEnumerators::TEC: {return indexOfW1b -1 + tTopo->tecRing(id);} //fragile: relies on ordering of 'type'
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


   induceVector(collection_points, det, localAmplitudes, recordMinAffectedStrip, recordMaxAffectedStrip, tTopo);

  /*
  auto ominA=recordMinAffectedStrip, omaxA=recordMaxAffectedStrip;
  std::vector<float> oampl(localAmplitudes);	
  induceOriginal(collection_points, det, oampl, ominA, omaxA, tTopo);

  //  std::cout << "orig " << ominA << " " << omaxA << " ";
  //for (auto a : oampl) std::cout << a << ",";
  //std::cout << std::endl;

  auto minA=recordMinAffectedStrip, maxA=recordMaxAffectedStrip;
  std::vector<float> ampl(localAmplitudes);
  induceVector(collection_points, det, ampl, minA, maxA, tTopo);

  // std::cout << "vect " << minA << " " << maxA << " ";          
  //for (auto a :	ampl) std::cout	<< a <<	",";
  //std::cout << std::endl;
 
  float diff=0;
  for (size_t i=0; i!=ampl.size(); ++i) { diff = std::max(diff,ampl[i]>0 ? std::abs(ampl[i]-oampl[i])/ampl[i] : 0);}
  if (diff> 1.e-4) {
    std::cout << diff << std::endl;
    std::cout << "orig " << ominA << " " << omaxA << " ";
//    for (auto a : oampl) std::cout << a << ",";
    std::cout << std::endl;
    std::cout << "vect " << minA << " " << maxA << " ";
//    for (auto a : ampl) std::cout << a << ",";
    std::cout << std::endl;
  }

  localAmplitudes.swap(ampl);
  recordMinAffectedStrip=minA;
  recordMaxAffectedStrip=maxA;
  */

}

void 
SiTrivialInduceChargeOnStrips::
induceVector(const SiChargeCollectionDrifter::collection_type& collection_points, 
	       const StripGeomDetUnit& det, 
	       std::vector<float>& localAmplitudes, 
	       size_t& recordMinAffectedStrip, 
	       size_t& recordMaxAffectedStrip,
	       const TrackerTopology *tTopo) const {


  auto const & coupling = signalCoupling[typeOf(det,tTopo)];
  const StripTopology& topology = dynamic_cast<const StripTopology&>(det.specificTopology());
  const int Nstrips =  topology.nstrips();

  if (Nstrips == 0) return;

  const int NP = collection_points.size();
  if(0==NP) return;

  constexpr int MaxN = 512;
  // if NP too large split...

  for (int ip=0; ip<NP; ip+=MaxN) {
    auto N = std::min(NP-ip,MaxN);

    count.dep(N);
    float amplitude[N];
    float chargePosition[N];
    float chargeSpread[N];
    int fromStrip[N];
    int nStrip[N];

    // load not vectorize
    //In strip coordinates:
    for (int i=0; i!=N;++i) {
      auto j = ip+i;
      if (0==collection_points[j].amplitude()) count.zero();
      chargePosition[i]=topology.strip(collection_points[j].position());
      chargeSpread[i]= collection_points[j].sigma() / topology.localPitch(collection_points[j].position());
      amplitude[i]=0.5f*collection_points[j].amplitude() / geVperElectron;
    }
    
    // this vectorize
    for (int i=0; i!=N;++i) {
      fromStrip[i]  = std::max( 0,  int(std::floor( chargePosition[i] - Nsigma*chargeSpread[i])) );
      nStrip[i] = std::min( Nstrips, int(std::ceil( chargePosition[i] + Nsigma*chargeSpread[i])) ) - fromStrip[i];
    }
    int tot=0;
    for (int i=0; i!=N;++i) tot += nStrip[i];
    tot+=N; // add last strip 
    count.val(tot);
    float value[tot];
    
    // assign relative position (lower bound of strip) in value;
    int kk=0;
    for (int i=0; i!=N;++i) {
      auto delta = 1.f/(std::sqrt(2.f)*chargeSpread[i]);
      auto pos = delta*(float(fromStrip[i])-chargePosition[i]);

      // VI: before value[0] was not defined and value[tot] was filled
      //     to fix this the loop below was changed
      for (int j=0;j<=nStrip[i]; ++j) { /// include last strip
	value[kk] = pos+float(j)*delta;
        ++kk;  
      }
    }
    assert(kk==tot);
    
    // main loop fully vectorized
    for (int k=0;k!=tot; ++k)
      value[k] = approx_erf(value[k]);
    
    // saturate 0 & NStrips strip to 0 and 1???
    kk=0;
    for (int i=0; i!=N;++i) {
      if (0 == fromStrip[i])  value[kk]=0;
      kk+=nStrip[i];
      if (Nstrips == fromStrip[i]+nStrip[i]) value[kk]=1.f;
      ++kk;
    }
    assert(kk==tot);
    
    // compute integral over strip (lower bound becomes the value)
    for (int k=0;k!=tot-1; ++k)
      value[k]-=value[k+1];  // this is negative!
    
    
    float charge[Nstrips]; for (int i=0;i!=Nstrips; ++i) charge[i]=0;
    kk=0;
    for (int i=0; i!=N;++i){ 
      for (int j=0;j!=nStrip[i]; ++j)
	charge[fromStrip[i]+j]-= amplitude[i]*value[kk++];
      ++kk; // skip last "strip"
    }
    assert(kk==tot);
    
    
    /// do crosstalk... (can be done better, most probably not worth)
    int minA=recordMinAffectedStrip, maxA=recordMaxAffectedStrip;
    int sc = coupling.size();
    for (int i=0;i!=Nstrips; ++i) {
      int strip = i;
      if (0==charge[i]) continue;
      auto affectedFromStrip  = std::max( 0, strip - sc + 1);
      auto affectedUntilStrip = std::min(Nstrips, strip + sc);  
      for (auto affectedStrip=affectedFromStrip;  affectedStrip < affectedUntilStrip;  ++affectedStrip)
	localAmplitudes[affectedStrip] += charge[i] * coupling[std::abs(affectedStrip - strip)] ;
      
      if( affectedFromStrip  < minA ) minA = affectedFromStrip;
      if( affectedUntilStrip > maxA ) maxA = affectedUntilStrip;
    }
    recordMinAffectedStrip=minA;
    recordMaxAffectedStrip=maxA;
  }  // end loop ip

}

void 
SiTrivialInduceChargeOnStrips::
induceOriginal(const SiChargeCollectionDrifter::collection_type& collection_points, 
	       const StripGeomDetUnit& det, 
	       std::vector<float>& localAmplitudes, 
	       size_t& recordMinAffectedStrip, 
	       size_t& recordMaxAffectedStrip,
	       const TrackerTopology *tTopo) const {


  auto const & coupling = signalCoupling[typeOf(det,tTopo)];
  const StripTopology& topology = dynamic_cast<const StripTopology&>(det.specificTopology());
  size_t Nstrips =  topology.nstrips();

  if (!collection_points.empty()) count.dep(collection_points.size());

  for (auto signalpoint = collection_points.begin();  signalpoint != collection_points.end();  signalpoint++ ) {
    
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

}

