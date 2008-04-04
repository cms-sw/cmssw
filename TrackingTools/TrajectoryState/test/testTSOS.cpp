#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>

class ConstMagneticField : public MagneticField {
public:

  virtual GlobalVector inTesla ( const GlobalPoint& ) const {
    return GlobalVector(0,0,4);
  }

};

int main() {
  using namespace std;

  MagneticField * field = new ConstMagneticField;
  GlobalPoint gp(0,0,0);
  GlobalVector gv(1,1,1);
  GlobalTrajectoryParameters gtp(gp,gv,1,field);

  BoundPlane* plane = new BoundPlane( gp, Surface::RotationType());

  TrajectoryStateOnSurface ts(gtp,*plane);

  cout << "ts.globalMomentum() " << ts.globalMomentum() << endl;
  cout << "ts.localMomentum()  " << ts.localMomentum() << endl;
  cout << "ts.transverseCurvature()  " << ts.transverseCurvature() << endl;
							   

  LocalPoint lp(0,0,0);
  LocalVector lv(1,1,1);
  LocalTrajectoryParameters ltp(lp,lv,1);

  TrajectoryStateOnSurface ts2(ltp,*plane, field);
  cout << "ts2.globalMomentum() " << ts2.globalMomentum() << endl;
  cout << "ts2.localMomentum()  " << ts2.localMomentum() << endl;
  cout << "ts2.transverseCurvature()  " << ts2.transverseCurvature() << endl;
}
