#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h" 

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"


#include<iostream>

namespace{

  struct Stat {

    long long ntot=0;
    long long nf1=0;
    long long nf2=0;

    long long ns1=0;
    long long ns2=0;
    long long ns11=0;
    long long ns21=0;

    long long nth=0;

                            //Geom checker     1337311   84696       634946      20369      259701        241266       18435         614128
    ~Stat() { std::cout << "Geom checker " << ntot<<' '<< nf1<<' '<< nf2 <<' '<< ns1<< ' '<< ns2 << ' ' << ns11 << ' '<< ns21 << ' ' << nth << std::endl;}

  };

  Stat stat;

}

std::pair<bool, TrajectoryStateOnSurface>  
GeomDetCompatibilityChecker::isCompatible(const GeomDet* theDet,
					  const TrajectoryStateOnSurface& tsos,
					  const Propagator& prop, 
					  const MeasurementEstimator& est) {
  stat.ntot++;

  constexpr float sagCut = 2;

  auto const & plane = theDet->specificSurface();
  StraightLinePlaneCrossing crossing(tsos.globalPosition().basicVector(),tsos.globalMomentum().basicVector(), prop.propagationDirection());
  auto path = crossing.pathLength(plane);

  TrajectoryStateOnSurface && propSt = prop.propagate( tsos, theDet->specificSurface());
  if unlikely ( !propSt.isValid()) { stat.nf1++; return std::make_pair( false, std::move(propSt));}
  auto isIn = path.first;
  float thresh=99999999;
  bool close = false;
  if(!path.first) stat.ns1++;
  else {
    auto gpos =  GlobalPoint(crossing.position(path.second));
    auto tpath = (gpos-tsos.globalPosition()).perp();
    thresh = std::abs(tpath*tpath*tsos.globalParameters().transverseCurvature());
    close = thresh<sagCut;
    if (close) stat.nth++;
    auto pos = plane.toLocal(GlobalPoint(crossing.position(path.second)));
    auto toll = LocalError(1.,0,1.);
    isIn = plane.bounds().inside(pos,toll);
    if (close && !isIn) { stat.ns2++;}// return std::make_pair( false,TrajectoryStateOnSurface()); }
  }
  auto es = est.estimate( propSt, theDet->specificSurface());
  if (!es) stat.nf2++;
  if (close && (!isIn) && (!es) ) stat.ns11++;
  if (close && es &&(!isIn)) { stat.ns21++; std::cout << thresh << std::endl;}
  return std::make_pair( es, std::move(propSt));

}
 
