#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"

#include <cmath>
#include<sstream>

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const FreeTrajectoryState& fts,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(fts)),
  theLocalError(),
  theLocalParameters(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(fts.hasCartesianError()),
  theCurvilinErrorUp2Date(fts.hasCurvilinearError()),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(1.),
  theField( &fts.parameters().magneticField())
{}    

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(par)),
  theLocalError(),
  theLocalParameters(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(1.),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CartesianTrajectoryError& err,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theLocalError(),
  theLocalParameters(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(true),
  theCurvilinErrorUp2Date(false),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(1.),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CurvilinearTrajectoryError& err,
			    const Surface& aSurface,
			    const SurfaceSide side,
			    double weight) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theLocalError(),
  theLocalParameters(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(true),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(weight),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CurvilinearTrajectoryError& err,
			    const Surface& aSurface,
			    double weight) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theLocalError(),
  theLocalParameters(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theGlobalParamsUp2Date(true),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(true),
  theSurfaceSide(SurfaceSideDefinition::atCenterOfSurface), 
  theSurfaceP( &aSurface), 
  theWeight(weight),
  theField( &par.magneticField())
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState( const LocalTrajectoryParameters& par,
			    const Surface& aSurface,
			    const MagneticField* field,
			    const SurfaceSide side) :
  theFreeState(0),
  theLocalError(),
  theLocalParameters(par),
  theLocalParametersValid(true),
  theLocalErrorValid(false),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theSurfaceSide(side),
  theSurfaceP( &aSurface), 
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
  theLocalError(err),
  theLocalParameters(par),
  theLocalParametersValid(true),
  theLocalErrorValid(true),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface),
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
  theLocalError(err),
  theLocalParameters(par),
  theLocalParametersValid(true),
  theLocalErrorValid(true),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theSurfaceSide(SurfaceSideDefinition::atCenterOfSurface),
  theSurfaceP( &aSurface), 
  theWeight(weight),
  theField(field)
{}

BasicSingleTrajectoryState::
BasicSingleTrajectoryState(const Surface& aSurface) :
  theFreeState(0),
  theLocalError(),
  theLocalParameters(),
  theLocalParametersValid(false),
  theLocalErrorValid(false),
  theGlobalParamsUp2Date(false),
  theCartesianErrorUp2Date(false),
  theCurvilinErrorUp2Date(false),
  theSurfaceSide(SurfaceSideDefinition::atCenterOfSurface), 
  theSurfaceP( &aSurface), 
  theWeight(0.),
  theField(0)
{}


BasicSingleTrajectoryState::~BasicSingleTrajectoryState(){}

void BasicSingleTrajectoryState::notValid() {
  throw TrajectoryStateException("TrajectoryStateOnSurface is invalid and cannot return any parameters");
}


void BasicSingleTrajectoryState::missingError(char const * where) const{
  std::stringstream form;
  form<<"TrajectoryStateOnSurface: attempt to access errors when none available "
      <<where<<".\nfreestate pointer: "
      <<theFreeState<<"\nlocal error valid :"<<theLocalErrorValid ;
  throw TrajectoryStateException(form.str());
}




void BasicSingleTrajectoryState::checkGlobalParameters() const {
  if(!theGlobalParamsUp2Date){
    //    cout<<"!theGlobalParamsUp2Date"<<endl;
    theGlobalParamsUp2Date = true;
    theCurvilinErrorUp2Date = false;
    theCartesianErrorUp2Date = false;
    // calculate global parameters from local
    GlobalPoint  x = surface().toGlobal(theLocalParameters.position());
    GlobalVector p = surface().toGlobal(theLocalParameters.momentum());
    // replace in place
    FreeTrajectoryState * fts = &(*theFreeState);
    if (fts) { 
        fts->~FreeTrajectoryState();
        new(fts) FreeTrajectoryState(x, p, theLocalParameters.charge(), theField);
    }else {
      theFreeState.replaceWith(new FreeTrajectoryState(x, p,
                                                       theLocalParameters.charge(),
                                                       theField));
    } 
  }
}

void BasicSingleTrajectoryState::checkCurvilinError() const {
  if(!theCurvilinErrorUp2Date){
    if(!theLocalParametersValid) createLocalParameters();
    // after createLocalParameters we can be sure theFreeState is not null
    if(!theLocalErrorValid) createLocalError();
    //    cout<<"!theCurvilinErrorUp2Date: create curviError from local"<<endl;
    theCurvilinErrorUp2Date = true;
    
    JacobianLocalToCurvilinear loc2Curv(surface(), localParameters(), *theField);
    const AlgebraicMatrix55& jac = loc2Curv.jacobian();
    
    const AlgebraicSymMatrix55 &cov = ROOT::Math::Similarity(jac, theLocalError.matrix());

    //theFreeState->setCurvilinearError( CurvilinearTrajectoryError(cov) );
    theFreeState->setCurvilinearError( cov );
  }
}

void BasicSingleTrajectoryState::checkCartesianError() const {
  if(!theCartesianErrorUp2Date){
    if(!theLocalParametersValid) createLocalParameters();
    if(!theLocalErrorValid) createLocalError();
    theCartesianErrorUp2Date = true;
   
    JacobianLocalToCartesian loc2Cart(surface(), localParameters());
    const AlgebraicMatrix65& jac = loc2Cart.jacobian();

    const AlgebraicSymMatrix66 &cov = ROOT::Math::Similarity(jac, theLocalError.matrix());
    
    //theFreeState->setCartesianError( CartesianTrajectoryError(cov) );
    theFreeState->setCartesianError( cov );
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
  if(theFreeState->hasCartesianError())
    createLocalErrorFromCartesianError();
  else if(theFreeState->hasCurvilinearError())
    createLocalErrorFromCurvilinearError();
  else
    theLocalErrorValid = false;
}

void 
BasicSingleTrajectoryState::createLocalErrorFromCurvilinearError() const {
  
  JacobianCurvilinearToLocal curv2Loc(surface(), localParameters(), *theField);
  const AlgebraicMatrix55& jac = curv2Loc.jacobian();
  
  const AlgebraicSymMatrix55 &cov = 
    ROOT::Math::Similarity(jac, theFreeState->curvilinearError().matrix());
  //    cout<<"Clocal via curvilinear error"<<endl;
  theLocalError = LocalTrajectoryError(cov);
  theLocalErrorValid = true;
}
 
void 
BasicSingleTrajectoryState::createLocalErrorFromCartesianError() const {

  JacobianCartesianToLocal cart2Loc(surface(), localParameters());
  const AlgebraicMatrix56& jac = cart2Loc.jacobian();
    

  const AlgebraicSymMatrix55 &C = 
    ROOT::Math::Similarity(jac, theFreeState->cartesianError().matrix());
  theLocalError = LocalTrajectoryError(C);
  theLocalErrorValid = true;
}

void
BasicSingleTrajectoryState::update( const LocalTrajectoryParameters& p,
        const Surface& aSurface,
        const MagneticField* field,
        const SurfaceSide side) 
{
    theLocalParameters = p;
    if (&aSurface != &*theSurfaceP) theSurfaceP.reset(&aSurface);
    theSurfaceSide = side;
    theWeight      = 1.0; 

    theGlobalParamsUp2Date   = false;
    theCartesianErrorUp2Date = false;
    theCurvilinErrorUp2Date  = false; 
    theLocalErrorValid       = false;
    theLocalParametersValid  = true;
}

void
BasicSingleTrajectoryState::update( const LocalTrajectoryParameters& p,
        const LocalTrajectoryError& err,
        const Surface& aSurface,
        const MagneticField* field,
        const SurfaceSide side, 
        double weight) 
{
    theLocalParameters = p;
    theLocalError      = err;
    if (&aSurface != &*theSurfaceP) theSurfaceP.reset(&aSurface);
    theSurfaceSide = side;
    theWeight      = weight; 

    theGlobalParamsUp2Date   = false;
    theCartesianErrorUp2Date = false;
    theCurvilinErrorUp2Date  = false; 
    theLocalErrorValid       = true;
    theLocalParametersValid  = true;

}

void 
BasicSingleTrajectoryState::rescaleError(double factor) {
  if unlikely(!hasError()) missingError(" trying to rescale");    
  if (theFreeState)
    theFreeState->rescaleError(factor);
  
  if (theLocalErrorValid){
    //do it by hand if the free state is not around.
    bool zeroField =theField->inInverseGeV(GlobalPoint(0,0,0)).mag2()==0;
    if unlikely(zeroField){
      AlgebraicSymMatrix55 errors=theLocalError.matrix();
      //scale the 0 indexed covariance by the square root of the factor
      for (unsigned int i=1;i!=5;++i)      errors(i,0)*=factor;
      double factor_squared=factor*factor;
      //scale all others by the scaled factor
      for (unsigned int i=1;i!=5;++i)  for (unsigned int j=i;j!=5;++j) errors(i,j)*=factor_squared;
      //term 0,0 is not scaled at all
      theLocalError = LocalTrajectoryError(errors);
    }
    else theLocalError *= (factor*factor);
  }
}

FreeTrajectoryState* 
BasicSingleTrajectoryState::freeTrajectoryState(bool withErrors) const {
  if(!isValid()) notValid();
  checkGlobalParameters();
  //if(hasError()) { // let's start like this to see if we alloc less
  if(withErrors && hasError()) { // this is the right thing
    checkCartesianError();
    checkCurvilinError();
  }
  return &(*theFreeState);
}


bool 
BasicSingleTrajectoryState::hasError() const {
  return (theFreeState && theFreeState->hasError()) || theLocalErrorValid;
}

