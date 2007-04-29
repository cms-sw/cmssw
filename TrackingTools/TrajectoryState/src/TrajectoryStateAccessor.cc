#include "TrackingTools/TrajectoryState/interface/TrajectoryStateAccessor.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"

double TrajectoryStateAccessor::inversePtError() const
{
  GlobalVector momentum = theFts.momentum();
  AlgebraicSymMatrix55 errMatrix = theFts.curvilinearError().matrix();
  
  float SinTheta=sin(momentum.theta());
  float CosTheta=cos(momentum.theta());
  float ptRec=momentum.perp();
  float InvpErr=errMatrix(0,0);
  float thetaErr=errMatrix(1,1);
  float corr=errMatrix(0,1);
  float invPtErr2 = 1/(SinTheta*SinTheta)*
    (InvpErr + 
     ((CosTheta*CosTheta)/(ptRec*ptRec))*thetaErr - 
     2*(CosTheta/ptRec)*corr);
  return sqrt(invPtErr2);
}
  
