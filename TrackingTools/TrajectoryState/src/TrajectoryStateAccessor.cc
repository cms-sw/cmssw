#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"

float TrajectoryStateAccessor::inversePtError() const
{
  GlobalVector momentum = theFts.momentum();
  AlgebraicSymMatrix55 const & errMatrix = theFts.curvilinearError().matrix();
  
  float ptRec2= momentum.perp2();
  float pzRec = momentum.z();
  float pzRec2 = pzRec*pzRec;
  float CosTheta2 = (pzRec2)/(ptRec2+pzRec2);
  float SinTheta2 = 1.f-CosTheta2;
 
  float par2 = CosTheta2/ptRec2;

  float InvpErr=errMatrix(0,0);
  float thetaErr=errMatrix(1,1);
  float corr=errMatrix(0,1);

  float invPtErr2 =
    ( InvpErr + par2*thetaErr - 
      2.f*std::sqrt(par2)*corr
    )/(SinTheta2);
  return std::sqrt(invPtErr2);
}
  
