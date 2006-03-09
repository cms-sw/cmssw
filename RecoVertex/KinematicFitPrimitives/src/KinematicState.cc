#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "TrackingTools/TrajectoryState/interface/FakeField.h"

KinematicState::KinematicState(const KinematicParameters& parameters, 
                               const KinematicParametersError& error, 
                               const TrackCharge& charge):
                               param(parameters),err(error), ch(charge), vl(true)
{}


bool KinematicState::operator==(const KinematicState& other) const 
{
 bool res = false;
 if((kinematicParameters().vector() == other.kinematicParameters().vector())&&
    (kinematicParametersError().matrix() == other.kinematicParametersError().matrix())) res = true;
 return res;
}

ParticleMass KinematicState::mass() const
{return param.vector()(7);}

KinematicParameters KinematicState::kinematicParameters() const
{return param;}

KinematicParametersError KinematicState::kinematicParametersError() const
{return err;}

GlobalVector KinematicState::globalMomentum() const
{return param.momentum();}
 
GlobalPoint KinematicState::globalPosition() const
{return param.position();}

TrackCharge KinematicState::particleCharge() const
{return ch;}

FreeTrajectoryState KinematicState::freeTrajectoryState() const
{
 GlobalTrajectoryParameters globalPar(globalPosition(),globalMomentum(),particleCharge(), 
				TrackingTools::FakeField::Field::field());
 AlgebraicSymMatrix cError = kinematicParametersError().matrix().sub(1,6);
 CartesianTrajectoryError cartError(cError);
// cout<<"conversion called"<<endl;
// cout<<"parameters::position"<<globalPosition()<<endl;
// cout<<"parameters::momentum"<<globalMomentum()<<endl;
// cout<<"parameters::error"<<cError<<endl;
 return FreeTrajectoryState(globalPar,cartError);
}
/*
AlgebraicSymMatrix KinematicState::weightMatrix() const
{
 GlobalTrajectoryParameters gtp = freeTrajectoryState().parameters();
 cout<<"curvilinear error is"<<freeTrajectoryState().curvilinearError().matrix()<<endl;
 return err.weightMatrix(gtp);
}
*/
