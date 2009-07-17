#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include <Geometry/CommonDetUnit/interface/GeomDet.h>

#include "MagneticField/Engine/interface/MagneticField.h"

#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"


#include <iostream>

class ConstMagneticField : public MagneticField {
public:

  virtual GlobalVector inTesla ( const GlobalPoint& ) const {
    return GlobalVector(0,0,4);
  }

};


// A fake Det class

class MyDet : public GeomDet {
 public:
  MyDet(BoundPlane * bp) :
    GeomDet(bp){}
  
  virtual DetId geographicalId() const {return DetId();}
  virtual std::vector< const GeomDet*> components() const {
    return std::vector< const GeomDet*>();
  }
  
  /// Which subdetector
  virtual SubDetector subDetector() const {return GeomDetEnumerators::DT;}
  
};  


// a 2d hit

#include "TrackingTools/TransientTrackingRecHit/interface/GenericTransientTrackingRecHit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/HelpertRecHit2DLocalPos.h"


class My2DHit : public GenericTransientTrackingRecHit {
public:

  My2DHit(const GeomDet * geom, TrackingRecHit* rh) :
    GenericTransientTrackingRecHit(geom,  rh) {}

  virtual void getKfComponents( KfComponentsHolder & holder ) const {
      HelpertRecHit2DLocalPos().getKfComponents(holder, *hit(), *det()); 
  }
};


void print(TrajectoryStateOnSurface const & ts) {
  using namespace std;

  cout << "ts.globalMomentum() " << ts.globalMomentum() << endl;
  cout << "ts.localMomentum()  " << ts.localMomentum() << endl;
  cout << "ts.transverseCurvature()  " << ts.transverseCurvature() << endl;
							   
}

int main() {

  MagneticField * field = new ConstMagneticField;
  GlobalPoint gp(0,0,0);
  GlobalVector gv(1,1,1);
  GlobalTrajectoryParameters gtp(gp,gv,1,field);

  BoundPlane* plane = new BoundPlane( gp, Surface::RotationType());
  GeomDet *  det =  new MyDet(plane);

  

  TrajectoryStateOnSurface ts(gtp,*plane);
  print (ts);

  LocalPoint lp(0,0,0);
  LocalVector lv(1,1,1);
  LocalTrajectoryParameters ltp(lp,lv,1);

  TrajectoryStateOnSurface ts2(ltp,*plane, field);
  print (ts2);


  LocalPoint m(0.1,0.1,0);
  LocalError e(0.2,0.1,-0.05);
    
  SiStripRecHit2D dummy;
  TrackingRecHit * hit = new SiStripMatchedRecHit2D(m, e, DetId() , &dummy, &dummy);
  TransientTrackingRecHit * thit = new  My2DHit(det, hit);

  TrajectoryStateUpdator * tsu = new KFUpdator();

  {
    TrajectoryStateOnSurface tsn =  (*tsu).update(ts, *thit);
    print (tsn);
  }

  {
   TrajectoryStateOnSurface tsn =  (*tsu).update(ts2, *thit);
   print(tsn);
    
  }

  return 0;

}
