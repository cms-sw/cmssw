#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"

#include <cmath>

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const FreeTrajectoryState& fts,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(fts)),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(fts.hasCartesianError()),
  theCurvilinErrorUp2Date(fts.hasCurvilinearError()),
  theLocalParameters(),
  theLocalError(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theSurfaceP( &aSurface), theSurfaceSide(side), theWeight(1.),
  theField( &fts.parameters().magneticField())
{}    

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(par)),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theLocalParameters(),
  theLocalError(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theSurfaceP( &aSurface), theSurfaceSide(side), theWeight(1.),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CartesianTrajectoryError& err,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(true),
  theCurvilinErrorUp2Date(false),
  theLocalParameters(),
  theLocalError(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theSurfaceP( &aSurface), theSurfaceSide(side), theWeight(1.),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CurvilinearTrajectoryError& err,
			    const Surface& aSurface,
			    const SurfaceSide side,
			    double weight) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(true),
  theLocalParameters(),
  theLocalError(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theSurfaceP( &aSurface), theSurfaceSide(side), theWeight(weight),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CurvilinearTrajectoryError& err,
			    const Surface& aSurface,
			    double weight) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(true),
  theLocalParameters(),
  theLocalError(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theSurfaceP( &aSurface), theSurfaceSide(atCenterOfSurface), 
  theWeight(weight),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const LocalTrajectoryParameters& par,
			    const Surface& aSurface,
			    const MagneticField* field,
			    const SurfaceSide side) :
  theFreeState(0),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theLocalParameters(par),
  theLocalError(),
  theLocalParametersValid(true),
  theLocalErrorValid(false),
  theSurfaceP( &aSurface), theSurfaceSide(side),
  theWeight(1.),
  theField(field) 
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const LocalTrajectoryParameters& par,
			    const LocalTrajectoryError& err,
			    const Surface& aSurface,
			    const MagneticField* field,
			    const SurfaceSide side,
			    double weight) :
  theFreeState(0),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theLocalParameters(par),
  theLocalError(err),
  theLocalParametersValid(true),
  theLocalErrorValid(true),
  theSurfaceP( &aSurface),
  theSurfaceSide(side), 
  theWeight(weight),
  theField(field)
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const LocalTrajectoryParameters& par,
			    const LocalTrajectoryError& err,
			    const Surface& aSurface,
			    const MagneticField* field,
			    double weight) :
  theFreeState(0),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theLocalParameters(par),
  theLocalError(err),
  theLocalParametersValid(true),
  theLocalErrorValid(true),
  theSurfaceP( &aSurface), theSurfaceSide(atCenterOfSurface),
  theWeight(weight),
  theField(field)
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState(const Surface& aSurface) :
  theFreeState(0),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theLocalParameters(),
  theLocalError(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theSurfaceP( &aSurface), theSurfaceSide(atCenterOfSurface), theWeight(0.),
  theField(0)
{}


BasicSingleTrajectoryState::~BasicSingleTrajectoryState(){}

void BasicSingleTrajectoryState::checkGlobalParameters() const {
  if(!theGlobalParamsUp2Date){
    //    cout<<"!theGlobalParamsUp2Date"<<endl;
    theGlobalParamsUp2Date = true;
    theCurvilinErrorUp2Date = false;
    theCartesianErrorUp2Date = false;
    // calculate global parameters from local
    GlobalPoint  x = surface().toGlobal(theLocalParameters.position());
    GlobalVector p = surface().toGlobal(theLocalParameters.momentum());
    theFreeState = new FreeTrajectoryState(GlobalTrajectoryParameters(x, p, 
								      theLocalParameters.charge(),
								      theField));
  }
}

void BasicSingleTrajectoryState::checkCurvilinError() const {
  if(!theCurvilinErrorUp2Date){
    if(!theLocalParametersValid) createLocalParameters();
    if(!theLocalErrorValid) createLocalError();
    //    cout<<"!theCurvilinErrorUp2Date: create curviError from local"<<endl;
    theCurvilinErrorUp2Date = true;
    
    JacobianLocalToCurvilinear loc2Curv(surface(), localParameters(), *theField);
    const AlgebraicMatrix& jac = loc2Curv.jacobian();
    
    AlgebraicSymMatrix cov = theLocalError.matrix().similarity(jac);
    GlobalTrajectoryParameters params = 
      theFreeState->parameters();
    if(theCartesianErrorUp2Date) {
      CartesianTrajectoryError cate = 
	theFreeState->cartesianError(); 
      theFreeState = new FreeTrajectoryState(params, cate,
					     CurvilinearTrajectoryError(cov));      
    } else {
      theFreeState = new FreeTrajectoryState(params,
					     CurvilinearTrajectoryError(cov));      
    }
  }
}

void BasicSingleTrajectoryState::checkCartesianError() const {
  if(!theCartesianErrorUp2Date){
    if(!theLocalParametersValid) createLocalParameters();
    if(!theLocalErrorValid) createLocalError();
    theCartesianErrorUp2Date = true;
   
    JacobianLocalToCartesian loc2Cart(surface(), localParameters());
    const AlgebraicMatrix& jac = loc2Cart.jacobian();

    AlgebraicSymMatrix cov = theLocalError.matrix().similarity(jac);
    //    cout<<"cov: "<<cov<<endl;
    GlobalTrajectoryParameters params = 
      theFreeState->parameters();
    if(theCurvilinErrorUp2Date) {
      CurvilinearTrajectoryError cute = 
	theFreeState->curvilinearError();
      theFreeState = 
	new FreeTrajectoryState(params, CartesianTrajectoryError(cov), cute);
    } else {
      theFreeState = 
	new FreeTrajectoryState(params, CartesianTrajectoryError(cov));
    }
  }
}
 
// create local parameters from global
void BasicSingleTrajectoryState::createLocalParameters() const {
  LocalPoint  x = surface().toLocal(theFreeState->position());
  LocalVector p = surface().toLocal(theFreeState->momentum());
// believe p.z() never exactly equals 0.
  bool isCharged = theFreeState->charge()!=0;
  theLocalParameters =
    LocalTrajectoryParameters(isCharged?theFreeState->signedInverseMomentum():1./p.mag(),
      p.x()/p.z(), p.y()/p.z(), x.x(), x.y(), p.z()>0. ? 1.:-1., isCharged);
  theLocalParametersValid = true;
}

void BasicSingleTrajectoryState::createLocalError() const {
    if(theFreeState->hasCurvilinearError())
      createLocalErrorFromCurvilinearError();
    else if(theFreeState->hasCartesianError())
      createLocalErrorFromCartesianError();
    else
      theLocalErrorValid = false;
}

void 
BasicSingleTrajectoryState::createLocalErrorFromCurvilinearError() const {
  
  JacobianCurvilinearToLocal curv2Loc(surface(), localParameters(), *theField);
  const AlgebraicMatrix& jac = curv2Loc.jacobian();
  
  AlgebraicSymMatrix cov = 
    theFreeState->curvilinearError().matrix().similarity(jac);
  //    cout<<"Clocal via curvilinear error"<<endl;
  theLocalError = LocalTrajectoryError(cov);
  theLocalErrorValid = true;
}
 
void 
BasicSingleTrajectoryState::createLocalErrorFromCartesianError() const {

  JacobianCartesianToLocal cart2Loc(surface(), localParameters());
  const AlgebraicMatrix& jac = cart2Loc.jacobian();
    

  AlgebraicSymMatrix C = 
    theFreeState->cartesianError().matrix().similarity(jac);
  theLocalError = LocalTrajectoryError(C);
  theLocalErrorValid = true;
}
