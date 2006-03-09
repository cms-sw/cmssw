#include "RecoVertex/KinematicFitPrimitives/interface/TrackKinematicStatePropagator.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "Geometry/Surface/interface/BoundPlane.h"
#include "TrackingTools/TrajectoryState/interface/FakeField.h"

using namespace std;

KinematicState TrackKinematicStatePropagator::propagateToTheTransversePCA(const KinematicState& state,
                                                                              const GlobalPoint& referencePoint) const
{
 if( state.particleCharge() == 0. ) {
    return propagateToTheTransversePCANeutral(state, referencePoint);
  } else {
    return propagateToTheTransversePCACharged(state, referencePoint);
  }
}

pair<HelixBarrelPlaneCrossingByCircle,BoundPlane *> TrackKinematicStatePropagator::planeCrossing(
                                                                         const FreeTrajectoryState& state,
                                                                         const GlobalPoint& point) const
{
 GlobalPoint inPos = state.position();
 GlobalVector inMom = state.momentum();
 double kappa = state.transverseCurvature();
 double fac = 1./state.charge()/TrackingTools::FakeField::Field::inInverseGeV(point).z();
 
 GlobalVectorDouble xOrig2Centre = GlobalVectorDouble(fac * inMom.y(), -fac * inMom.x(), 0.);
 GlobalVectorDouble xOrigProj = GlobalVectorDouble(inPos.x(), inPos.y(), 0.);
 GlobalVectorDouble xRefProj = GlobalVectorDouble(point.x(), point.y(), 0.);
 GlobalVectorDouble deltax = xRefProj-xOrigProj-xOrig2Centre;
 GlobalVectorDouble ndeltax = deltax.unit();
  
 PropagationDirection direction = anyDirection;
 Surface::PositionType pos(point);
  
// Need to define plane with orientation as the ImpactPointSurface
 GlobalVector X(ndeltax.x(), ndeltax.y(), ndeltax.z());
 GlobalVector Y(0.,0.,1.);
 Surface::RotationType rot(X,Y);
 BoundPlane* plane = new BoundPlane(pos,rot);
 HelixBarrelPlaneCrossingByCircle 
   planeCrossing(HelixPlaneCrossing::PositionType(inPos.x(), inPos.y(), inPos.z()),
		 HelixPlaneCrossing::DirectionType(inMom.x(), inMom.y(), inMom.z()), 
		 kappa, direction);
 pair<bool,double> propResult = planeCrossing.pathLength(*plane);
 if ( !propResult.first ) throw VertexException("KinematicStatePropagator without material::propagation failed!");
 return pair<HelixBarrelPlaneCrossingByCircle,BoundPlane *>(planeCrossing,plane);
}


KinematicState TrackKinematicStatePropagator::propagateToTheTransversePCACharged(const KinematicState& state, 
                                                                              const GlobalPoint& referencePoint) const
{
//first use the existing FTS propagator to obtain parameters at PCA
//in transverse plane to the given point

//final parameters and covariance
  AlgebraicVector par(7,0);
  AlgebraicSymMatrix cov(7,0);
    
//initial parameters as class and vectors:  
  GlobalTrajectoryParameters inPar(state.globalPosition(),state.globalMomentum(), 
		state.particleCharge(), TrackingTools::FakeField::Field::field());
  ParticleMass mass = state.mass();							  
  GlobalVector inMom = state.globalMomentum();							  
  
//making a free trajectory state and looking 
//for helix barrel plane crossing  
  FreeTrajectoryState fState = state.freeTrajectoryState();		
  GlobalPoint iP = referencePoint;					  
  pair<HelixBarrelPlaneCrossingByCircle, BoundPlane *> cros = planeCrossing(fState,iP);	   
  
  HelixBarrelPlaneCrossingByCircle planeCrossing = cros.first; 
  BoundPlane * plane = cros.second;
  pair<bool,double> propResult = planeCrossing.pathLength(*plane);
  double s = propResult.second;
  
  HelixPlaneCrossing::PositionType xGen = planeCrossing.position(s);
  GlobalPoint nPosition(xGen.x(),xGen.y(),xGen.z());
  HelixPlaneCrossing::DirectionType pGen = planeCrossing.direction(s);
  pGen *= inMom.mag()/pGen.mag();
  GlobalVector nMomentum(pGen.x(),pGen.y(),pGen.z()); 
  par(1) = nPosition.x();
  par(2) = nPosition.y();
  par(3) = nPosition.z();
  par(4) = nMomentum.x();
  par(5) = nMomentum.y();
  par(6) = nMomentum.z();
  par(7) = mass;
  
//covariance matrix business  
//elements of 7x7 covariance matrix responcible for the mass and
//mass - momentum projections corellations do change under such a transformation:
//special Jacobian needed  
  GlobalTrajectoryParameters fPar(nPosition, nMomentum, state.particleCharge(),
  				TrackingTools::FakeField::Field::field());
					       							  
  JacobianCartesianToCurvilinear cart2curv(inPar);
  JacobianCurvilinearToCartesian curv2cart(fPar);
  
  AlgebraicMatrix ca2cu(6,7,0);
  AlgebraicMatrix cu2ca(7,6,0);
  ca2cu.sub(1,1,cart2curv.jacobian());
  cu2ca.sub(1,1,curv2cart.jacobian());
  ca2cu(6,7) = 1;  
  cu2ca(7,6) = 1;

//now both transformation jacobians: cartesian to curvilinear and back are done
//We transform matrix to curv frame, then propagate it and translate it back to
//cartesian frame.  
  cov = state.kinematicParametersError().matrix().similarity(ca2cu);

//propagation jacobian
  AnalyticalCurvilinearJacobian prop(inPar,nPosition,nMomentum,s);
  AlgebraicMatrix pr(6,6,0);
  pr(6,6) = 1;
  pr.sub(1,1,prop.jacobian());
  
//transportation
  cov = cov.similarity(pr);
  
//now geting back to 7-parametrization from curvilinear
  cov = cov.similarity(cu2ca);
  
//return parameters as a kiematic state  
  KinematicParameters resPar(par);
  KinematicParametersError resEr(cov);
  return  KinematicState(resPar,resEr,state.particleCharge()); 
 }
  
KinematicState TrackKinematicStatePropagator::propagateToTheTransversePCANeutral(const KinematicState& state,
                                                                              const GlobalPoint& referencePoint) const
{
//new parameters vector and covariance:
 AlgebraicVector par(7,0);
 AlgebraicSymMatrix cov(7,0);
 
 AlgebraicVector inStatePar = state.kinematicParameters().vector();
 GlobalTrajectoryParameters inPar(state.globalPosition(),state.globalMomentum(), 
		state.particleCharge(), TrackingTools::FakeField::Field::field());
 
//first making a free trajectory state and propagating it according
//to the algorithm provided by Thomas Speer and Wolfgang Adam 
 FreeTrajectoryState fState = state.freeTrajectoryState();
  
 GlobalPoint xvecOrig = fState.position();
 double phi = fState.momentum().phi();
 double theta = fState.momentum().theta();
 double xOrig = xvecOrig.x();
 double yOrig = xvecOrig.y();
 double zOrig = xvecOrig.z();
 double xR = referencePoint.x();
 double yR = referencePoint.y();

 double s2D = (xR - xOrig)  * cos(phi) + (yR - yOrig)  * sin(phi);
 double s = s2D / sin(theta);
 double xGen = xOrig + s2D*cos(phi);
 double yGen = yOrig + s2D*sin(phi);
 double zGen = zOrig + s2D/tan(theta);
 GlobalPoint xPerigee = GlobalPoint(xGen, yGen, zGen);

//new parameters
 GlobalVector pPerigee = fState.momentum();
 par(1) = xPerigee.x();
 par(2) = xPerigee.y();	
 par(3) = xPerigee.z();	
 par(4) = pPerigee.x();	
 par(5) = pPerigee.y();	
 par(6) = pPerigee.z();	
 par(7) = inStatePar(7);

//covariance matrix business:
//everything lake it was before: jacobains are smart enouhg to 
//distinguish between neutral and charged states themselves

 GlobalTrajectoryParameters fPar(xPerigee, pPerigee, state.particleCharge(),
				TrackingTools::FakeField::Field::field());
 JacobianCartesianToCurvilinear cart2curv(inPar);
 JacobianCurvilinearToCartesian curv2cart(fPar);
  
 AlgebraicMatrix ca2cu(6,7,0);
 AlgebraicMatrix cu2ca(7,6,0);
 ca2cu.sub(1,1,cart2curv.jacobian());
 cu2ca.sub(1,1,curv2cart.jacobian());
 ca2cu(6,7) = 1;  
 cu2ca(7,6) = 1;

//now both transformation jacobians: cartesian to curvilinear and back are done
//We transform matrix to curv frame, then propagate it and translate it back to
//cartesian frame.  
  cov = state.kinematicParametersError().matrix().similarity(ca2cu);

//propagation jacobian
  AnalyticalCurvilinearJacobian prop(inPar,xPerigee,pPerigee,s);
  AlgebraicMatrix pr(6,6,0);
  pr(6,6) = 1;
  pr.sub(1,1,prop.jacobian());
  
//transportation
  cov = cov.similarity(pr);
  
//now geting back to 7-parametrization from curvilinear
  cov = cov.similarity(cu2ca);
 
//return parameters as a kiematic state  
 KinematicParameters resPar(par);
 KinematicParametersError resEr(cov);
 return  KinematicState(resPar,resEr,state.particleCharge());  	
}


