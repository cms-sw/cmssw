#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 
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

int main() {

  std::cout << "sizes tsos, bsts, fts" << std::endl;
  std::cout << sizeof( TrajectoryStateOnSurface) << std::endl;
  std::cout << sizeof(BasicSingleTrajectoryState) << std::endl;
  std::cout << sizeof(FreeTrajectoryState) << std::endl;


  using namespace std;

  MagneticField * field = new ConstMagneticField;
  GlobalPoint gp(0,0,0);
  GlobalVector gv(1,1,1);
  GlobalTrajectoryParameters gtp(gp,gv,1,field);
  double v[15] = {0.01,-0.01,0.  ,0.,0.,
                        0.01,0.  ,0.,0.,
                             0.01,0.,0.,
                                  1.,0.,
                                     1.};
  AlgebraicSymMatrix55 gerr(v,15);
  BoundPlane* plane = new BoundPlane( gp, Surface::RotationType());

  TrajectoryStateOnSurface ts(gtp,gerr,*plane);

  cout << "ts.globalMomentum() " << ts.globalMomentum() << endl;
  cout << "ts.localMomentum()  " << ts.localMomentum() << endl;
  cout << "ts.transverseCurvature()  " << ts.transverseCurvature() << endl;
  cout << "ts inversePtErr " << TrajectoryStateAccessor(*ts.freeState()).inversePtError() << std::endl;
							   

  LocalPoint lp(0,0,0);
  LocalVector lv(1,1,1);
  LocalTrajectoryParameters ltp(lp,lv,1);
  LocalTrajectoryError lerr(1.,1.,0.1,0.1,0.1);
  TrajectoryStateOnSurface ts2(ltp,lerr, *plane, field);
  cout << "ts2.globalMomentum() " << ts2.globalMomentum() << endl;
  cout << "ts2.localMomentum()  " << ts2.localMomentum() << endl;
  cout << "ts2.transverseCurvature()  " << ts2.transverseCurvature() << endl;
  cout << "ts2 inversePtErr " << TrajectoryStateAccessor(*ts2.freeState()).inversePtError() << std::endl; 
}
