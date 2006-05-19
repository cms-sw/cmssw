#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"

double TrajectoryStateAccessor::inversePtError() const
{
  GlobalVector momentum = theFts.momentum();
  AlgebraicSymMatrix errMatrix = theFts.curvilinearError().matrix();
  
  float SinTheta=sin(momentum.theta());
  float CosTheta=cos(momentum.theta());
  float ptRec=momentum.perp();
  float InvpErr=errMatrix(1,1);
  float thetaErr=errMatrix(2,2);
  float corr=errMatrix(1,2);
  float invPtErr2 = pow(1/SinTheta,2)*
    (InvpErr + 
     (pow(CosTheta,2)/pow(ptRec,2))*thetaErr - 
     2*(CosTheta/ptRec)*corr);
  return sqrt(invPtErr2);
}
  
