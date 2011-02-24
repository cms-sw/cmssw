#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryParameters.h"


CurvilinearTrajectoryParameters::CurvilinearTrajectoryParameters(const AlgebraicVector5& v, bool charged) {
  theQbp    = v[0];
  thelambda = v[1];
  thephi    = v[2];
  thexT     = v[3];
  theyT     = v[4];
  
  if ( charged )
    theCharge = theQbp>0 ? 1 : -1;
  else
    theCharge = 0;
  
}


CurvilinearTrajectoryParameters::CurvilinearTrajectoryParameters(double aQbp, double alambda, double aphi, double axT, double ayT, bool charged): theQbp(aQbp), thelambda(alambda), thephi(aphi), thexT(axT), theyT(ayT) {
    
    if ( charged ) {
      theQbp = aQbp;
      theCharge = theQbp>0 ? 1 : -1;
    }
    else {
      theQbp = aQbp;
      theCharge = 0;
    }
}
  

CurvilinearTrajectoryParameters::CurvilinearTrajectoryParameters(const GlobalPoint& aX,const GlobalVector& aP,TrackCharge aCharge)
{
  if(aCharge==0) 
    theQbp = 1./aP.mag();
  else
    theQbp=aCharge/aP.mag();

  double pT2= aP.x()*aP.x()+aP.y()*aP.y();
  double pT =sqrt(pT2);
  thelambda= atan(aP.z()/pT);
  thephi=atan2(aP.y(),aP.x());
  thexT= (-aP.y()*aX.x()+ aP.x()*aX.y()) / pT;
  theyT= (-aX.x()*aP.x()*aP.z() - aX.y()*aP.z()*aP.y() + aX.z()*(pT2))  / (aP.mag()*pT);
  theCharge=aCharge;
}




AlgebraicVector5 CurvilinearTrajectoryParameters::vector() const {
  return AlgebraicVector5(signedInverseMomentum(),
			  thelambda,
			  thephi,
			  thexT,
			  theyT);
}



bool CurvilinearTrajectoryParameters::updateP(double dP) {
  //FIXME. something is very likely to be missing here
  double p = 1./fabs(theQbp);
  if ((p += dP) <= 0.) return false;
  double newQbp = theQbp > 0. ? 1./p : -1./p;
  theQbp = newQbp;
  return true;
}
