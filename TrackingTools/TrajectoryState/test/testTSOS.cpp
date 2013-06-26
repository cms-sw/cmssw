#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"
#include "DataFormats/GeometrySurface/interface/Surface.h" 
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/TrajectoryState/interface/PerigeeConversions.h"

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
  cout << "ts curv err\n" << ts.curvilinearError().matrix() << std::endl;					   
  cout << "ts cart err\n" << ts.cartesianError().matrix() << std::endl;					   
  {
    JacobianCartesianToCurvilinear cart2Curv(ts.globalParameters());
    const AlgebraicMatrix56& jac = cart2Curv.jacobian();
    
    CurvilinearTrajectoryError theCurvilinearError = 
      ROOT::Math::Similarity(jac, ts.cartesianError().matrix());
    cout << "curv from cart \n" <<  theCurvilinearError.matrix() << std::endl;

    auto a55 = 
      PerigeeConversions::jacobianCurvilinear2Perigee(*ts.freeState());
    std::cout << " curv 2 per " << a55 << std::endl;
    a55 = PerigeeConversions::jacobianPerigee2Curvilinear(gtp);
    std::cout << " per 2 cuv " << a55 << std::endl;
    auto a66 = PerigeeConversions::jacobianParameters2Cartesian({1.,1.,1.}, gp,1,field);
    std::cout << " per 2 cart " << a66 << std::endl;
  }

  LocalPoint lp(0,0,0);
  LocalVector lv(1,1,1);
  LocalTrajectoryParameters ltp(lp,lv,1);
  LocalTrajectoryError lerr(1.,1.,0.1,0.1,0.1);
  TrajectoryStateOnSurface ts2(ltp,lerr, *plane, field);
  cout << "ts2.globalMomentum() " << ts2.globalMomentum() << endl;
  cout << "ts2.localMomentum()  " << ts2.localMomentum() << endl;
  cout << "ts2.transverseCurvature()  " << ts2.transverseCurvature() << endl;
  cout << "ts2 inversePtErr " << TrajectoryStateAccessor(*ts2.freeState()).inversePtError() << std::endl; 
  cout << "ts2 curv err\n" << ts2.curvilinearError().matrix() << std::endl;					   
  cout << "ts2 cart err\n" << ts2.cartesianError().matrix() << std::endl;					   
  {
    JacobianCartesianToCurvilinear cart2Curv(ts2.globalParameters());
    const AlgebraicMatrix56& jac = cart2Curv.jacobian();
    
    CurvilinearTrajectoryError theCurvilinearError = 
      ROOT::Math::Similarity(jac, ts2.cartesianError().matrix());
    cout << "curv from cart \n" <<  theCurvilinearError.matrix() << std::endl;

    auto a55 = 
      PerigeeConversions::jacobianCurvilinear2Perigee(*ts2.freeState());
    std::cout  << " curv 2 per "<< a55 << std::endl;
  }


}
