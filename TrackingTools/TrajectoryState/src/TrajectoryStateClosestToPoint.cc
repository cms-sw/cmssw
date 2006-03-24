#include "TrackingTools/TrajectoryState/interface/TrajectoryStateClosestToPoint.h"
#include "MagneticField/Engine/interface/MagneticField.h"

// Private constructor

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const FTS& originalFTS, const GlobalPoint& referencePoint) :
  theFTS(originalFTS), theFTSavailable(true), theRefPoint(referencePoint)
{
  theParameters = perigeeConversions.ftsToPerigeeParameters(originalFTS, referencePoint);
  if (theFTS.hasError()) {
    AlgebraicSymMatrix errorMatrix = theFTS.curvilinearError().matrix();
    AlgebraicMatrix curv2perigee = jacobianCurvilinear2Perigee(theFTS);
    thePerigeeError = PerigeeTrajectoryError(errorMatrix.similarity(curv2perigee));
    errorIsAvailable = true;
  } 
  else {
    errorIsAvailable = false;
  }
}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * no error is provided.
   */

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
  const GlobalPoint& referencePoint, const MagneticField* magField) :
    theFTSavailable(true), theRefPoint(referencePoint), 
    theParameters(perigeeParameters), errorIsAvailable(false)
{
  theFTS = FTS(GlobalTrajectoryParameters(
	    perigeeConversions.positionFromPerigee(theParameters, theRefPoint),
	    perigeeConversions.momentumFromPerigee(theParameters, theRefPoint, *magField), 
	    perigeeConversions.chargeFromPerigee(theParameters, theRefPoint), 
	    magField));
}

  /**
   * Public constructor, which is used to convert perigee 
   * parameters to a FreeTrajectoryState. For the case where
   * an error is provided.
   */

TrajectoryStateClosestToPoint::
TrajectoryStateClosestToPoint(const PerigeeTrajectoryParameters& perigeeParameters,
  const PerigeeTrajectoryError& perigeeError, const GlobalPoint& referencePoint,
  const MagneticField* magField):
    theFTSavailable(true), theRefPoint(referencePoint), theParameters(perigeeParameters),
    thePerigeeError(perigeeError), errorIsAvailable(true)
    
{
  FTS incompleteFTS = FTS(GlobalTrajectoryParameters(
	    perigeeConversions.positionFromPerigee(theParameters, theRefPoint),
	    perigeeConversions.momentumFromPerigee(theParameters, theRefPoint, *magField), 
	    perigeeConversions.chargeFromPerigee(theParameters, theRefPoint),
	    magField));
  AlgebraicMatrix perigee2curv = jacobianPerigee2Curvilinear(incompleteFTS);
  AlgebraicSymMatrix curvilinearError = thePerigeeError.covarianceMatrix().similarity(perigee2curv);
  theFTS = FTS(GlobalTrajectoryParameters(
	    perigeeConversions.positionFromPerigee(theParameters, theRefPoint),
	    perigeeConversions.momentumFromPerigee(theParameters, theRefPoint, *magField), 
	    perigeeConversions.chargeFromPerigee(theParameters, theRefPoint), magField),
	    CurvilinearTrajectoryError(curvilinearError));
}


AlgebraicMatrix 
TrajectoryStateClosestToPoint::jacobianCurvilinear2Perigee(const FreeTrajectoryState& fts) const {

  GlobalVector p = fts.momentum();

  GlobalVector Z = GlobalVector(0.,0.,1.);
  GlobalVector T = p.unit();
  GlobalVector U = Z.cross(T).unit();; 
  GlobalVector V = T.cross(U);

  GlobalVector I = GlobalVector(-p.x(), -p.y(), 0.); //opposite to track dir.
  I = I.unit();
  GlobalVector J(-I.y(), I.x(),0.); //counterclockwise rotation
  GlobalVector K(Z);
  GlobalPoint x = fts.position();
//   GlobalVector B = MagneticField::inInverseGeV(x);
  GlobalVector B  = fts.parameters().magneticField().inTesla(x) * 2.99792458e-3;
  GlobalVector H = B.unit();
  GlobalVector HxT = H.cross(T);
  GlobalVector N = HxT.unit();
  double alpha = HxT.mag();
  double qbp = fts.signedInverseMomentum();
  double Q = -B.mag() * qbp;
  double alphaQ = alpha * Q;

  double lambda = 0.5 * M_PI - p.theta();
  double coslambda = cos(lambda), tanlambda = tan(lambda);

  double TI = T.dot(I);
  double NU = N.dot(U);
  double NV = N.dot(V);
  double UI = U.dot(I);
  double VI = V.dot(I);
  double UJ = U.dot(J);
  double VJ = V.dot(J);
  double UK = U.dot(K);
  double VK = V.dot(K);

  AlgebraicMatrix jac(5,5,0);

  if( fabs(fts.transverseCurvature())<1.e-10 ) {
    jac(1,1) = 1/coslambda;
    jac(1,2) = tanlambda/coslambda/fts.parameters().momentum().mag();
  }else{
    double Bz = B.z();
    jac(1,1) = -Bz/coslambda;
    jac(1,2) = -Bz * tanlambda/coslambda*qbp;
    jac(2,4) = alphaQ * NV * UI/TI;
    jac(2,5) = alphaQ * NV * VI/TI;
    jac(1,4) = -jac(1,2) * jac(2,4);
    jac(1,5) = -jac(1,2) * jac(2,5);
    jac(3,4) = -alphaQ/coslambda * NU * UI/TI;
    jac(3,5) = -alphaQ/coslambda * NU * VI/TI;
  }
  jac(2,2) = -1.;
  jac(3,3) = 1.;
  jac(4,4) = VK/TI;
  jac(4,5) = -UK/TI;
  jac(5,4) = -VJ/TI;
  jac(5,5) = UJ/TI;
  
  return jac;
  
}


AlgebraicMatrix 
TrajectoryStateClosestToPoint::jacobianPerigee2Curvilinear(const FreeTrajectoryState& fts) const {

  GlobalVector p = fts.momentum();

  GlobalVector Z = GlobalVector(0.,0.,1.);
  GlobalVector T = p.unit();
  GlobalVector U = Z.cross(T).unit();; 
  GlobalVector V = T.cross(U);

  GlobalVector I = GlobalVector(-p.x(), -p.y(), 0.); //opposite to track dir.
  I = I.unit();
  GlobalVector J(-I.y(), I.x(),0.); //counterclockwise rotation
  GlobalVector K(Z);
  GlobalPoint x = fts.position();
//   GlobalVector B = MagneticField::inInverseGeV(x);
  GlobalVector B  = fts.parameters().magneticField().inTesla(x) * 2.99792458e-3;
  GlobalVector H = B.unit();
  GlobalVector HxT = H.cross(T);
  GlobalVector N = HxT.unit();
  double alpha = HxT.mag();
  double qbp = fts.signedInverseMomentum();
  double Q = -B.mag() * qbp;
  double alphaQ = alpha * Q;

  double lambda = 0.5 * M_PI - p.theta();
  double coslambda = cos(lambda), sinlambda = sin(lambda);
  double mqbpt = -1./coslambda * qbp;

  double TJ = T.dot(J);
  double TK = T.dot(K);
  double NU = N.dot(U);
  double NV = N.dot(V);
  double UJ = U.dot(J);
  double VJ = V.dot(J);
  double UK = U.dot(K);
  double VK = V.dot(K);

  AlgebraicMatrix jac(5,5,0);

  if( fabs(fts.transverseCurvature())<1.e-10 ) {
    jac(1,1) = coslambda;
    jac(1,2) = sinlambda/coslambda/fts.parameters().momentum().mag();
  }else{
    jac(1,1) = -coslambda/B.z();
    jac(1,2) = -sinlambda * mqbpt;
    jac(2,4) = -alphaQ * NV * TJ;
    jac(2,5) = -alphaQ * NV * TK;
    jac(3,4) = -alphaQ/coslambda * NU * TJ;
    jac(3,5) = -alphaQ/coslambda * NU * TK;
  }
  jac(2,2) = -1.;
  jac(3,3) = 1.;
  jac(4,4) = UJ;
  jac(4,5) = UK;
  jac(5,4) = VJ;
  jac(5,5) = VK;
  
  return jac;
}

