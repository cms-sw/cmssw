#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"


#include <iostream>

class ConstMagneticField : public MagneticField {
public:

  virtual GlobalVector inTesla ( const GlobalPoint& ) const {
    return GlobalVector(0,0,4);
  }

};

void compute(GlobalTrajectoryParameters const & gtp1, GlobalTrajectoryParameters  const & gtp2) {
  ClosestApproachInRPhi ca;
  
  bool ok = ca.compute(gtp1,gtp2);
  
  if(!ok) 
    std::cout << "no intercept!" std::endl;
  else {
    std::cout << "distance, xpoint" << ca.distance() << ca.crossingPoint() << std::endl;
    std::pair <GlobalTrajectoryParameters, GlobalTrajectoryParameters > tca = ca.trajectoryParameters ();
    std::cout << tca.first << std::endl;
    std::cout << tca.second << std::endl;
  }
}


int main() {

  MagneticField * field = new ConstMagneticField;

  {
    GlobalPoint gp1(0,0,0);
    GlobalVector gv1(1,1,1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,1,field);
    
    GlobalPoint gp2(0,0,0);
    GlobalVector gv2(1,1,-1);
    GlobalTrajectoryParameters gtp2(gp2,gv2,-1,field);
    
    compute(gtp1,gtp2);
  }
  {
    GlobalPoint gp1(-1,0,0);
    GlobalVector gv1(1,1,1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,1,field);
    
    GlobalPoint gp2(1,0,0);
    GlobalVector gv2(1,1,-1);
    GlobalTrajectoryParameters gtp2(gp2,gv2,-1,field);
    
    compute(gtp1,gtp2);
  }
 {
    GlobalPoint gp1(-1,0,0);
    GlobalVector gv1(1,1,1);
    GlobalTrajectoryParameters gtp1(gp1,gv1,1,field);
    
    GlobalPoint gp2(1,0,0);
    GlobalVector gv2(-1,1,-1);
    GlobalTrajectoryParameters gtp2(gp2,gv2,-1,field);
    
    compute(gtp1,gtp2);
  }


  return 0;

}
