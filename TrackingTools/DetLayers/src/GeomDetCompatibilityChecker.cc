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

  constexpr float tollerance = 1.; // one cm
  constexpr float sagCut = 2*tollerance;  

  auto const & plane = theDet->specificSurface();
  StraightLinePlaneCrossing crossing(tsos.globalPosition().basicVector(),tsos.globalMomentum().basicVector(), prop.propagationDirection());
  auto path = crossing.pathLength(plane);

  auto isIn = path.first;
  float thresh=99999999;
  bool close = false;
  if  unlikely(!path.first) stat.ns1++;
  else {
    auto gpos =  GlobalPoint(crossing.position(path.second));
    auto tpath2 = (gpos-tsos.globalPosition()).perp2();
    // sagitta = d^2*c/2
    thresh = std::abs(tpath2*tsos.globalParameters().transverseCurvature());
    close = thresh<sagCut;
    if (close) { 
       stat.nth++;
       auto pos = plane.toLocal(GlobalPoint(crossing.position(path.second)));
       auto toll = LocalError(tollerance,0,tollerance);
       isIn = plane.bounds().inside(pos,toll);
       if (!isIn) { stat.ns2++; if (prop.propagationDirection()==alongMomentum) return std::make_pair( false,TrajectoryStateOnSurface()); }
    }
  }

  TrajectoryStateOnSurface && propSt = prop.propagate( tsos, theDet->specificSurface());
  if unlikely ( !propSt.isValid()) { stat.nf1++; return std::make_pair( false, std::move(propSt));}


  auto es = est.estimate( propSt, theDet->specificSurface());
  if (!es) stat.nf2++;
  if (close && (!isIn) && (!es) ) stat.ns11++;
  if (close && es &&(!isIn)) { stat.ns21++; } // std::cout << thresh << std::endl;}
  return std::make_pair( es, std::move(propSt));

}
 
