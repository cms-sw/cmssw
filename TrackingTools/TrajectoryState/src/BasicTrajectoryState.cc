#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"

#include <cmath>
#include<sstream>

BasicTrajectoryState::
BasicTrajectoryState( const FreeTrajectoryState& fts,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(fts)),
  theLocalError(InvalidError()),
  theLocalParameters(),
  theLocalParametersValid(false),
  theGlobalParamsUp2Date(true),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(1.),
  theField( &fts.parameters().magneticField())
{}    

BasicTrajectoryState::
BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(par)),
  theLocalError(InvalidError()),
  theLocalParameters(),
  theLocalParametersValid(false),
   theGlobalParamsUp2Date(true),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(1.),
  theField( &par.magneticField())
{}

BasicTrajectoryState::
BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CartesianTrajectoryError& err,
			    const Surface& aSurface,
			    const SurfaceSide side) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theLocalError(InvalidError()),
  theLocalParameters(),
  theLocalParametersValid(false),
  theGlobalParamsUp2Date(true),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(1.),
  theField( &par.magneticField())
{}

BasicTrajectoryState::
BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CurvilinearTrajectoryError& err,
			    const Surface& aSurface,
			    const SurfaceSide side,
			    double weight) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theLocalError(InvalidError()),
  theLocalParameters(),
  theLocalParametersValid(false),
  theGlobalParamsUp2Date(true),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface), 
  theWeight(weight),
  theField( &par.magneticField())
{}

BasicTrajectoryState::
BasicTrajectoryState( const GlobalTrajectoryParameters& par,
			    const CurvilinearTrajectoryError& err,
			    const Surface& aSurface,
			    double weight) :
  theFreeState( new FreeTrajectoryState(par, err)),
  theLocalError(InvalidError()),
  theLocalParameters(),
  theLocalParametersValid(false),
  theGlobalParamsUp2Date(true),
  theSurfaceSide(SurfaceSideDefinition::atCenterOfSurface), 
  theSurfaceP( &aSurface), 
  theWeight(weight),
  theField( &par.magneticField())
{}

BasicTrajectoryState::
BasicTrajectoryState( const LocalTrajectoryParameters& par,
			    const Surface& aSurface,
			    const MagneticField* field,
			    const SurfaceSide side) :
  theFreeState(0),
  theLocalError(InvalidError()),
  theLocalParameters(par),
  theLocalParametersValid(true),
  theGlobalParamsUp2Date(false),
   theSurfaceSide(side),
  theSurfaceP( &aSurface), 
  theWeight(1.),
  theField(field) 
{}

BasicTrajectoryState::
BasicTrajectoryState( const LocalTrajectoryParameters& par,
			    const LocalTrajectoryError& err,
			    const Surface& aSurface,
			    const MagneticField* field,
			    const SurfaceSide side,
			    double weight) :
  theFreeState(0),
  theLocalError(err),
  theLocalParameters(par),
  theLocalParametersValid(true),
  theGlobalParamsUp2Date(false),
  theSurfaceSide(side), 
  theSurfaceP( &aSurface),
  theWeight(weight),
  theField(field)
{}

BasicTrajectoryState::
BasicTrajectoryState( const LocalTrajectoryParameters& par,
			    const LocalTrajectoryError& err,
			    const Surface& aSurface,
			    const MagneticField* field,
			    double weight) :
  theFreeState(0),
  theLocalError(err),
  theLocalParameters(par),
  theLocalParametersValid(true),
  theGlobalParamsUp2Date(false),
  theSurfaceSide(SurfaceSideDefinition::atCenterOfSurface),
  theSurfaceP( &aSurface), 
  theWeight(weight),
  theField(field)
{}

BasicTrajectoryState::
BasicTrajectoryState(const Surface& aSurface) :
  theFreeState(0),
  theLocalError(InvalidError()),
  theLocalParameters(),
  theLocalParametersValid(false),
  theGlobalParamsUp2Date(false),
  theSurfaceSide(SurfaceSideDefinition::atCenterOfSurface), 
  theSurfaceP( &aSurface), 
  theWeight(0.),
  theField(0)
{}


BasicTrajectoryState::~BasicTrajectoryState(){}

void BasicTrajectoryState::notValid() {
  throw TrajectoryStateException("TrajectoryStateOnSurface is invalid and cannot return any parameters");
}


void BasicTrajectoryState::missingError(char const * where) const{
  std::stringstream form;
  form<<"TrajectoryStateOnSurface: attempt to access errors when none available "
      <<where<<".\nfreestate pointer: "
      <<theFreeState<<"\nlocal error valid :"<< theLocalError.valid() ;
  throw TrajectoryStateException(form.str());
}




void BasicTrajectoryState::checkGlobalParameters() const {
  if likely(theGlobalParamsUp2Date) return;
 
  //    cout<<"!theGlobalParamsUp2Date"<<endl;
  theGlobalParamsUp2Date = true;
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

void BasicTrajectoryState::checkCurvilinError() const {
  if likely(theFreeState->hasCurvilinearError()) return;

  if unlikely(!theLocalParametersValid) createLocalParameters();
  
  JacobianLocalToCurvilinear loc2Curv(surface(), localParameters(), *theField);
    const AlgebraicMatrix55& jac = loc2Curv.jacobian();
    
    const AlgebraicSymMatrix55 &cov = ROOT::Math::Similarity(jac, theLocalError.matrix());

    theFreeState->setCurvilinearError( cov );
  
}


 
// create local parameters from global
void BasicTrajectoryState::createLocalParameters() const {
  LocalPoint  x = surface().toLocal(theFreeState->position());
  LocalVector p = surface().toLocal(theFreeState->momentum());
// believe p.z() never exactly equals 0.
  bool isCharged = theFreeState->charge()!=0;
  theLocalParameters =
    LocalTrajectoryParameters(isCharged?theFreeState->signedInverseMomentum():1./p.mag(),
      p.x()/p.z(), p.y()/p.z(), x.x(), x.y(), p.z()>0. ? 1.:-1., isCharged);
  theLocalParametersValid = true;
}

void BasicTrajectoryState::createLocalError() const {
  if likely(theFreeState->hasCurvilinearError())
    createLocalErrorFromCurvilinearError();
  else theLocalError = InvalidError();
}

void 
BasicTrajectoryState::createLocalErrorFromCurvilinearError() const {
  
  JacobianCurvilinearToLocal curv2Loc(surface(), localParameters(), *theField);
  const AlgebraicMatrix55& jac = curv2Loc.jacobian();
  
  const AlgebraicSymMatrix55 &cov = 
    ROOT::Math::Similarity(jac, theFreeState->curvilinearError().matrix());
  //    cout<<"Clocal via curvilinear error"<<endl;
  theLocalError = LocalTrajectoryError(cov);
}
 


void
BasicTrajectoryState::update( const LocalTrajectoryParameters& p,
        const Surface& aSurface,
        const MagneticField* field,
        const SurfaceSide side) 
{
    theLocalParameters = p;
    if (&aSurface != &*theSurfaceP) theSurfaceP.reset(&aSurface);
    theField=field;
    theSurfaceSide = side;
    theWeight      = 1.0; 
    theLocalError = InvalidError();

    theGlobalParamsUp2Date   = false;
    theLocalParametersValid  = true;
}

void
BasicTrajectoryState::update( const LocalTrajectoryParameters& p,
        const LocalTrajectoryError& err,
        const Surface& aSurface,
        const MagneticField* field,
        const SurfaceSide side, 
        double weight) 
{
    theLocalParameters = p;
    theLocalError      = err;
    if (&aSurface != &*theSurfaceP) theSurfaceP.reset(&aSurface);
    theField=field;
    theSurfaceSide = side;
    theWeight      = weight; 

    theGlobalParamsUp2Date   = false;
    theLocalParametersValid  = true;

}

void 
BasicTrajectoryState::rescaleError(double factor) {
  if unlikely(!hasError()) missingError(" trying to rescale");    
  if (theFreeState)
    theFreeState->rescaleError(factor);
  
  if (theLocalError.valid()){
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
BasicTrajectoryState::freeTrajectoryState(bool withErrors) const {
  if(!isValid()) notValid();
  checkGlobalParameters();
  if(withErrors && hasError()) { // this is the right thing
    checkCurvilinError();
  }
  return &(*theFreeState);
}



#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
std::vector<TrajectoryStateOnSurface> 
BasicTrajectoryState::components() const {
  std::vector<TrajectoryStateOnSurface> result; result.reserve(1);
  result.push_back( const_cast<BasicTrajectoryState*>(this));
  return result;
}
