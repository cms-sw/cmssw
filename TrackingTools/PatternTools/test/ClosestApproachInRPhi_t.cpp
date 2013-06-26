#define private public
#include "TrackingTools/PatternTools/interface/ClosestApproachInRPhi.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistanceHelixHelix.h"
#undef private

// #include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// #include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"


#include "TrackingTools/PatternTools/src/ClosestApproachInRPhi.cc"


#include <iostream>

class ConstMagneticField : public MagneticField {
public:

  virtual GlobalVector inTesla ( const GlobalPoint& ) const {
    return GlobalVector(0,0,4);
  }

};


namespace {
  inline GlobalPoint mean ( std::pair<GlobalPoint, GlobalPoint> pr ) {
    return GlobalPoint ( 0.5*(pr.first.basicVector() + pr.second.basicVector()) );
  }

  inline double dist ( std::pair<GlobalPoint, GlobalPoint> pr ) {
    return ( pr.first - pr.second ).mag();
  }
}


void compute(GlobalTrajectoryParameters const & gtp1, GlobalTrajectoryParameters  const & gtp2) {
  ClosestApproachInRPhi ca;
  TwoTrackMinimumDistanceHelixHelix TTMDhh;

  std::cout << "CAIR" << std::endl;
  bool ok = ca.calculate(gtp1,gtp2);
  if(!ok) 
    std::cout << "no intercept!" << std::endl;
  else {
    std::cout << "distance, xpoint " << ca.distance() << ca.crossingPoint() << std::endl;
    std::pair <GlobalTrajectoryParameters, GlobalTrajectoryParameters > tca = ca.trajectoryParameters();
    std::cout << tca.first << std::endl;
    std::cout << tca.second << std::endl;
  }

  std::cout << "TTMDhh" << std::endl;
  bool nok = TTMDhh.calculate(gtp1,gtp2,.0001);
  if(nok) 
    std::cout << "no intercept!" << std::endl;
  else {
     std::pair<GlobalPoint, GlobalPoint> pr = TTMDhh.points();
     std::cout << "distance, xpoint " << dist(pr) << mean(pr) << std::endl;
    // std::pair <GlobalTrajectoryParameters, GlobalTrajectoryParameters > thh = TTMDhh.trajectoryParameters();
    // std::cout << thh.first << std::endl;
    // std::cout << thh.second << std::endl;
  }
}


int main() {

  MagneticField * field = new ConstMagneticField;

  {
    // going back and forth gtp2 should be identical to gpt1....
    GlobalPoint gp1(1,0,0);
    GlobalVector gv1(1,1,-1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,1,field);
    double bz = field->inTesla(gp1).z() * 2.99792458e-3;
    GlobalPoint np(0.504471,    -0.498494,     0.497014);
    GlobalTrajectoryParameters gtpN = ClosestApproachInRPhi::newTrajectory(np,gtp1,bz);
    GlobalTrajectoryParameters gtp2 = ClosestApproachInRPhi::newTrajectory(gp1,gtpN,bz);
    std::cout << gtp1 << std::endl;
    std::cout << gtpN << std::endl;
    std::cout << gtp2 << std::endl;
    std::cout << std::endl;
  }


  {
    std::cout <<"opposite sign, same direction, same origin: the two circles are tangent to each other at gp1\n" << std::endl;
    GlobalPoint gp1(0,0,0);
    GlobalVector gv1(1,1,1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,1,field);
    
    GlobalPoint gp2(0,0,0);
    GlobalVector gv2(1,1,-1);
    GlobalTrajectoryParameters gtp2(gp2,gv2,-1,field);
    
    compute(gtp1,gtp2);
    std::cout << std::endl;

  }
  {
     std::cout <<" not crossing: the pcas are on the line connecting the two centers\n"
	       <<"the momenta at the respective pcas shall be parallel as they are perpendicular to the same line\n"
	       <<"(the one connecting the two centers)\n" << std::endl;
    GlobalPoint gp1(-1,0,0);
    GlobalVector gv1(1,1,1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,-1,field);
    
    GlobalPoint gp2(1,0,0);
    GlobalVector gv2(1,1,-1);
    GlobalTrajectoryParameters gtp2(gp2,gv2,1,field);
    
    compute(gtp1,gtp2);
   std::cout << std::endl;
  }
  {
    std::cout <<"crossing (only opposite changes w.r.t. previous)\n" << std::endl;
    GlobalPoint gp1(-1,0,0);
    GlobalVector gv1(1,1,1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,1,field);
   
    GlobalPoint gp2(1,0,0);
    GlobalVector gv2(1,1,-1);
    GlobalTrajectoryParameters gtp2(gp2,gv2,-1,field);

    compute(gtp1,gtp2);
    std::cout << std::endl;
  }

  {
    std::cout <<"crossing\n" << std::endl;
    GlobalPoint gp1(-1,0,0);
    GlobalVector gv1(1,1,1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,-1,field);
    
    GlobalPoint gp2(1,0,0);
    GlobalVector gv2(-1,1,-1);
    GlobalTrajectoryParameters gtp2(gp2,gv2,1,field);
    
    compute(gtp1,gtp2);
   std::cout << std::endl;
  }


  return 0;

}
