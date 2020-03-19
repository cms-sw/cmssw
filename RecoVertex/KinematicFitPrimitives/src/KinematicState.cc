#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/Matrices.h"

KinematicState::KinematicState(const KinematicParameters& parameters,
                               const KinematicParametersError& error,
                               const TrackCharge& charge,
                               const MagneticField* field)
    : fts(GlobalTrajectoryParameters(parameters.position(), parameters.momentum(), charge, field),
          CartesianTrajectoryError(error.matrix().Sub<AlgebraicSymMatrix66>(0, 0))),
      param(parameters),
      err(error),
      vl(true) {}

bool KinematicState::operator==(const KinematicState& other) const {
  return (kinematicParameters().vector() == other.kinematicParameters().vector()) &&
         (kinematicParametersError().matrix() == other.kinematicParametersError().matrix());
}

/*
void KinematicState::setFreeTrajectoryState() const {
 GlobalTrajectoryParameters globalPar(globalPosition(), globalMomentum(),
	particleCharge(), theField);
 AlgebraicSymMatrix66 cError =
	kinematicParametersError().matrix().Sub<AlgebraicSymMatrix66>(0,0);
 CartesianTrajectoryError cartError(cError);
// cout<<"conversion called"<<endl;
// cout<<"parameters::position"<<globalPosition()<<endl;
// cout<<"parameters::momentum"<<globalMomentum()<<endl;
// cout<<"parameters::error"<<cError<<endl;
 fts = FreeTrajectoryState(globalPar,cartError);
}
*/
