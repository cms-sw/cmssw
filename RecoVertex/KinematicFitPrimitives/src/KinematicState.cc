#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"

KinematicState::KinematicState(const KinematicParameters& parameters,
	const KinematicParametersError& error, const TrackCharge& charge,
	const MagneticField* field) :
	theField(field), param(parameters),err(error), ch(charge), vl(true)
{}


bool KinematicState::operator==(const KinematicState& other) const
{
 bool res = false;
 if((kinematicParameters().vector() == other.kinematicParameters().vector())&&
    (kinematicParametersError().matrix() == other.kinematicParametersError().matrix())) res = true;
 return res;
}


FreeTrajectoryState KinematicState::freeTrajectoryState() const
{
 GlobalTrajectoryParameters globalPar(globalPosition(), globalMomentum(),
	particleCharge(), theField);
 AlgebraicSymMatrix66 cError =
	kinematicParametersError().matrix().Sub<AlgebraicSymMatrix66>(0,0);
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
