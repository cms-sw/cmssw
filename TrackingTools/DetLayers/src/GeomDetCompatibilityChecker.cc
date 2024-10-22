// #define STAT_TSB

#ifdef STAT_TSB
#include <iostream>
#endif

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "FWCore/Utilities/interface/Likely.h"

namespace {

  struct Stat {
    struct Nop {
      Nop(int = 0) {}
      Nop& operator=(int) { return *this; }
      void operator++(int) {}
    };
#ifdef STAT_TSB
    using VAR = long long;
#else
    using VAR = Nop;
#endif
    VAR ntot = 0;
    VAR nf1 = 0;
    VAR nf2 = 0;

    VAR ns1 = 0;
    VAR ns2 = 0;
    VAR ns11 = 0;
    VAR ns21 = 0;

    VAR nth = 0;
    VAR nle = 0;

    //Geom checker     1337311   84696       634946      20369      259701        241266       18435         614128
    //    Geom checker 124,567,704  3,862,821 36,055,605 3,799,127 29,825,229    4,573,316     320,063         75,420,840
    //    Geom checker 119,618,014  2,307,939 31,142,922 2,903,245 34,673,978    5,139,741     539,152         86,847,116 18,196,497
    //    Geom checker 125,554,439  3,431,348 31,900,589 3,706,531 37,272,039    5,160,257     1,670,236       90,573,031 19,505,412
    //    Geom checker 119,583,440  2,379,307 28,357,175 2,960,173 38,977,837    6,239,242       620,636       86,726,732  9,574,902
    //    Geom checker 214,884,027  6,756,424 54,479,049  7,059,696  78,135,883 18443999 1124058 158,174,933 17153503
    //    Geom checker 453,155,905 14,054,554 79,733,432 14,837,002 163,414,609 0 0 324,629,999 0
#ifdef STAT_TSB
    ~Stat() {
      std::cout << "Geom checker " << ntot << ' ' << nf1 << ' ' << nf2 << ' ' << ns1 << ' ' << ns2 << ' ' << ns11 << ' '
                << ns21 << ' ' << nth << ' ' << nle << std::endl;
    }
#endif
  };

  CMS_THREAD_SAFE Stat stat;  // for production purpose it is thread safe

}  // namespace

std::pair<bool, TrajectoryStateOnSurface> GeomDetCompatibilityChecker::isCompatible(const GeomDet* theDet,
                                                                                    const TrajectoryStateOnSurface& tsos,
                                                                                    const Propagator& prop,
                                                                                    const MeasurementEstimator& est) {
  stat.ntot++;

  auto const sagCut = est.maxSagitta();
  auto const minTol2 = est.minTolerance2();

  // std::cout << "param " << sagCut << ' ' << std::sqrt(minTol2) << std::endl;

  /*
  auto err2 = tsos.curvilinearError().matrix()(3,3);
  auto largeErr = err2> 0.1*tolerance2;
  if (largeErr) stat.nle++; 
  */

  bool isIn = false;
  float sagitta = 99999999.0f;
  bool close = false;
  if LIKELY (sagCut > 0) {
    // linear approximation
    auto const& plane = theDet->specificSurface();
    StraightLinePlaneCrossing crossing(
        tsos.globalPosition().basicVector(), tsos.globalMomentum().basicVector(), prop.propagationDirection());
    auto path = crossing.pathLength(plane);
    isIn = path.first;
    if UNLIKELY (!path.first)
      stat.ns1++;
    else {
      auto gpos = GlobalPoint(crossing.position(path.second));
      auto tpath2 = (gpos - tsos.globalPosition()).perp2();
      // sagitta = d^2*c/2
      sagitta = 0.5f * std::abs(tpath2 * tsos.globalParameters().transverseCurvature());
      close = sagitta < sagCut;
      LogDebug("TkDetLayer") << "GeomDetCompatibilityChecker: sagitta " << sagitta << std::endl;
      if (close) {
        stat.nth++;
        auto pos = plane.toLocal(GlobalPoint(crossing.position(path.second)));
        // auto toll = LocalError(tolerance2,0,tolerance2);
        auto tollL2 = std::max(sagitta * sagitta, minTol2);
        auto toll = LocalError(tollL2, 0, tollL2);
        isIn = plane.bounds().inside(pos, toll);
        if (!isIn) {
          stat.ns2++;
          LogDebug("TkDetLayer") << "GeomDetCompatibilityChecker: not in " << pos << std::endl;
          return std::make_pair(false, TrajectoryStateOnSurface());
        }
      }
    }
  }

  // precise propagation
  TrajectoryStateOnSurface&& propSt = prop.propagate(tsos, theDet->specificSurface());
  if UNLIKELY (!propSt.isValid()) {
    stat.nf1++;
    return std::make_pair(false, std::move(propSt));
  }

  auto es = est.estimate(propSt, theDet->specificSurface());
  if (!es)
    stat.nf2++;
  if (close && (!isIn) && (!es))
    stat.ns11++;
  if (close && es && (!isIn)) {
    stat.ns21++;
  }  // std::cout << sagitta << std::endl;}
  return std::make_pair(es, std::move(propSt));
}
