#ifndef RecoTracker_TkSeedGenerator_FastCircleFit_h
#define RecoTracker_TkSeedGenerator_FastCircleFit_h

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "CommonTools/Utils/interface/DynArray.h"

#include <vector>

/**
 * Same (almost) as FastCircle but with arbitrary number of hits. 
 *
 * Strandlie, Wroldsen, Frühwirth NIM A 488 (2002) 332-341.
 * Frühwirth, Strandlie, Waltenberger NIM A 490 (2002) 366-378.
 */
class FastCircleFit {
public:
  // TODO: try to make the interface more generic than just vectors
  FastCircleFit(const std::vector<GlobalPoint>& points, const std::vector<GlobalError>& errors):
    FastCircleFit(points.data(), errors.data(), points.size()) {}
  FastCircleFit(const DynArray<GlobalPoint>& points, const DynArray<GlobalError>& errors):
    FastCircleFit(points.begin(), errors.begin(), points.size()) {}
  FastCircleFit(const GlobalPoint *points, const GlobalError *errors, size_t size);
  ~FastCircleFit() = default;

  float x0() const { return x0_; }
  float y0() const { return y0_; }
  float rho() const { return rho_; }

  // TODO: I'm not sure if the minimized square sum is chi2 distributed
  float chi2() const { return chi2_; }

private:
  float x0_;
  float y0_;
  float rho_;
  float chi2_;
};


#endif
