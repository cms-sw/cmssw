#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryParameters.h"
#include<cmath>


CurvilinearTrajectoryParameters::CurvilinearTrajectoryParameters(const GlobalPoint& aX,const GlobalVector& aP,TrackCharge aCharge)
{
 
  theQbp=aCharge/aP.mag();

  double pT2= aP.x()*aP.x()+aP.y()*aP.y();
  double pT =sqrt(pT2);
  thelambda= atan(aP.z()/pT);
  thephi=atan2(aP.y(),aP.x());
  thexT= (-aP.y()*aX.x()+ aP.x()*aX.y()) / pT;
  theyT= (-aX.x()*aP.x()*aP.z() - aX.y()*aP.z()*aP.y() + aX.z()*(pT2))  / (aP.mag()*pT);
}


bool CurvilinearTrajectoryParameters::updateP(double dP) {
  //FIXME. something is very likely to be missing here
  double p = 1./std::abs(Qbp());
  if ((p += dP) <= 0.) return false;
  double newQbp = Qbp() > 0 ? 1./p : -1./p;
  theQbp = newQbp;
  return true;
}
