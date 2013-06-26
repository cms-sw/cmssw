#include "RecoVertex/KinematicFitPrimitives/interface/TrackKinematicStatePropagator.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "DataFormats/GeometrySurface/interface/OpenBounds.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

KinematicState
TrackKinematicStatePropagator::propagateToTheTransversePCA
	(const KinematicState& state, const GlobalPoint& referencePoint) const
{
 if( state.particleCharge() == 0. ) {
    return propagateToTheTransversePCANeutral(state, referencePoint);
  } else {
    return propagateToTheTransversePCACharged(state, referencePoint);
  }
}

pair<HelixBarrelPlaneCrossingByCircle, BoundPlane::BoundPlanePointer>
TrackKinematicStatePropagator::planeCrossing(const FreeTrajectoryState& state,
	const GlobalPoint& point) const
{
 GlobalPoint inPos = state.position();
 GlobalVector inMom = state.momentum();
 double kappa = state.transverseCurvature();
 double fac = 1./state.charge()/state.parameters().magneticField().inInverseGeV(point).z();
 
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
 Plane::PlanePointer plane = Plane::build(pos,rot);
 HelixBarrelPlaneCrossingByCircle 
   planeCrossing(HelixPlaneCrossing::PositionType(inPos.x(), inPos.y(), inPos.z()),
		 HelixPlaneCrossing::DirectionType(inMom.x(), inMom.y(), inMom.z()), 
		 kappa, direction);
 return std::pair<HelixBarrelPlaneCrossingByCircle,Plane::PlanePointer>(planeCrossing,plane);
}


KinematicState
TrackKinematicStatePropagator::propagateToTheTransversePCACharged
	(const KinematicState& state, const GlobalPoint& referencePoint) const
{
//first use the existing FTS propagator to obtain parameters at PCA
//in transverse plane to the given point

//final parameters and covariance
  AlgebraicVector7 par;
  AlgebraicSymMatrix77 cov;
    
//initial parameters as class and vectors:  
  GlobalTrajectoryParameters inPar(state.globalPosition(),state.globalMomentum(), 
		state.particleCharge(), state.magneticField());
  ParticleMass mass = state.mass();							  
  GlobalVector inMom = state.globalMomentum();							  
  
//making a free trajectory state and looking 
//for helix barrel plane crossing  
  FreeTrajectoryState fState = state.freeTrajectoryState();		
  GlobalPoint iP = referencePoint;					  
  std::pair<HelixBarrelPlaneCrossingByCircle, BoundPlane::BoundPlanePointer> cros = planeCrossing(fState,iP);	   
  
  HelixBarrelPlaneCrossingByCircle planeCrossing = cros.first; 
  BoundPlane::BoundPlanePointer plane = cros.second;
  std::pair<bool,double> propResult = planeCrossing.pathLength(*plane);
  if ( !propResult.first ) {
    LogDebug("RecoVertex/TrackKinematicStatePropagator") 
	 << "Propagation failed! State is invalid\n";
    return  KinematicState();
  }
  double s = propResult.second;
  
  HelixPlaneCrossing::PositionType xGen = planeCrossing.position(s);
  GlobalPoint nPosition(xGen.x(),xGen.y(),xGen.z());
  HelixPlaneCrossing::DirectionType pGen = planeCrossing.direction(s);
  pGen *= inMom.mag()/pGen.mag();
  GlobalVector nMomentum(pGen.x(),pGen.y(),pGen.z()); 
  par(0) = nPosition.x();
  par(1) = nPosition.y();
  par(2) = nPosition.z();
  par(3) = nMomentum.x();
  par(4) = nMomentum.y();
  par(5) = nMomentum.z();
  par(6) = mass;
  
//covariance matrix business  
//elements of 7x7 covariance matrix responcible for the mass and
//mass - momentum projections corellations do change under such a transformation:
//special Jacobian needed  
  GlobalTrajectoryParameters fPar(nPosition, nMomentum, state.particleCharge(),
  				state.magneticField());
					       							  
  JacobianCartesianToCurvilinear cart2curv(inPar);
  JacobianCurvilinearToCartesian curv2cart(fPar);
  
  AlgebraicMatrix67 ca2cu;
  AlgebraicMatrix76 cu2ca;
  ca2cu.Place_at(cart2curv.jacobian(),0,0);
  cu2ca.Place_at(curv2cart.jacobian(),0,0);
  ca2cu(5,6) = 1;  
  cu2ca(6,5) = 1;

//now both transformation jacobians: cartesian to curvilinear and back are done
//We transform matrix to curv frame, then propagate it and translate it back to
//cartesian frame.  
  AlgebraicSymMatrix66 cov1 = ROOT::Math::Similarity(ca2cu, state.kinematicParametersError().matrix());

//propagation jacobian
  AnalyticalCurvilinearJacobian prop(inPar,nPosition,nMomentum,s);
  AlgebraicMatrix66 pr;
  pr(5,5) = 1;
  pr.Place_at(prop.jacobian(),0,0);

//transportation
  AlgebraicSymMatrix66 cov2 = ROOT::Math::Similarity(pr, cov1);
  
//now geting back to 7-parametrization from curvilinear
  cov = ROOT::Math::Similarity(cu2ca, cov2);
  
//return parameters as a kiematic state  
  KinematicParameters resPar(par);
  KinematicParametersError resEr(cov);
  return  KinematicState(resPar,resEr,state.particleCharge(), state.magneticField()); 
 }
  
KinematicState TrackKinematicStatePropagator::propagateToTheTransversePCANeutral
	(const KinematicState& state, const GlobalPoint& referencePoint) const
{
//new parameters vector and covariance:
 AlgebraicVector7 par;
 AlgebraicSymMatrix77 cov;
 
 //AlgebraicVector7 inStatePar = state.kinematicParameters().vector();
 GlobalTrajectoryParameters inPar(state.globalPosition(),state.globalMomentum(), 
		state.particleCharge(), state.magneticField());
 
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
 par(0) = xPerigee.x();
 par(1) = xPerigee.y(); 
 par(2) = xPerigee.z(); 
 par(3) = pPerigee.x(); 
 par(4) = pPerigee.y(); 
 par(5) = pPerigee.z(); 
 // par(6) = inStatePar(7);
 par(6) = state.mass();

//covariance matrix business:
//everything lake it was before: jacobains are smart enouhg to 
//distinguish between neutral and charged states themselves

 GlobalTrajectoryParameters fPar(xPerigee, pPerigee, state.particleCharge(),
				state.magneticField());

 JacobianCartesianToCurvilinear cart2curv(inPar);
 JacobianCurvilinearToCartesian curv2cart(fPar);
  
  AlgebraicMatrix67 ca2cu;
  AlgebraicMatrix76 cu2ca;
  ca2cu.Place_at(cart2curv.jacobian(),0,0);
  cu2ca.Place_at(curv2cart.jacobian(),0,0);
  ca2cu(5,6) = 1;  
  cu2ca(6,5) = 1;

//now both transformation jacobians: cartesian to curvilinear and back are done
//We transform matrix to curv frame, then propagate it and translate it back to
//cartesian frame.  
  AlgebraicSymMatrix66 cov1 = ROOT::Math::Similarity(ca2cu, state.kinematicParametersError().matrix());

//propagation jacobian
  AnalyticalCurvilinearJacobian prop(inPar,xPerigee,pPerigee,s);
  AlgebraicMatrix66 pr;
  pr(5,5) = 1;
  pr.Place_at(prop.jacobian(),0,0);
  
//transportation
  AlgebraicSymMatrix66 cov2 = ROOT::Math::Similarity(pr, cov1);
  
//now geting back to 7-parametrization from curvilinear
  cov = ROOT::Math::Similarity(cu2ca, cov2);
 
//return parameters as a kiematic state  
 KinematicParameters resPar(par);
 KinematicParametersError resEr(cov);
 return  KinematicState(resPar,resEr,state.particleCharge(), state.magneticField());  	
}


