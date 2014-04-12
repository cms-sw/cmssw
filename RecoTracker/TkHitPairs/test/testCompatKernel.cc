#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitEtaCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitZCheck.h"

#include "RecoTracker/TkHitPairs/interface/RecHitsSortedInPhi.h"

typedef PixelRecoRange<float> Range;


// devirtualizer
#include<tuple>
namespace {

  template<typename Algo>
  struct Kernel {
    using  Base = HitRZCompatibility;
    void set(Base const * a) {
      assert( a->algo()==Algo::me);
      checkRZ=reinterpret_cast<Algo const *>(a);
    }
    
    void operator()(int b, int e, const RecHitsSortedInPhi & innerHitsMap, bool * ok) const {
      constexpr float nSigmaRZ = std::sqrt(12.f);
      for (int i=b; i!=e; ++i) {
	Range allowed = checkRZ->range(innerHitsMap.u[i]);
	float vErr = nSigmaRZ * innerHitsMap.dv[i];
	Range hitRZ(innerHitsMap.v[i]-vErr, innerHitsMap.v[i]+vErr);
	Range crossRange = allowed.intersection(hitRZ);
	ok[i-b] = ! crossRange.empty() ;
      }
    }
    Algo const * checkRZ;
    
  };


  template<typename ... Args> using Kernels = std::tuple<Kernel<Args>...>;

}

void testR(HitRZCompatibility const * algo, int b, int e, const RecHitsSortedInPhi & innerHitsMap, bool * ok) {
  Kernel<HitRCheck> k; k.set(algo);
  k(b,e, innerHitsMap,ok);
}

void testZ(HitRZCompatibility const * algo, int b, int e, const RecHitsSortedInPhi & innerHitsMap, bool * ok) {
  Kernel<HitZCheck> k; k.set(algo);
  k(b,e, innerHitsMap,ok);
}
