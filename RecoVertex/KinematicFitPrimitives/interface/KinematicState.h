#ifndef KinematicState_H
#define KinematicState_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParameters.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParametersError.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"

/**
 * Class  providing  a state of particle needed
 * for Kinematic Fit. 
 *
 * Kirill Prokofiev, March 2003
 */


class KinematicState{

public:

/**
 * Default constructor for internal
 * KinematicFitPrimitives library needs
 * only
 */
 KinematicState()
 {vl = false;}
 
/**
 * Constructor taking directly KinematicParameters
 * KinematicError and Charge. To be used with
 * proper KinematicStateBuilder.
 */
 KinematicState(const KinematicParameters& parameters, const KinematicParametersError& error, 
                                                       const TrackCharge& charge);
						       
 bool operator==(const KinematicState& other) const;

/**
 * Access methods to parameters
 * and private data
 */
 ParticleMass mass() const;

 KinematicParameters kinematicParameters() const;

 KinematicParametersError kinematicParametersError() const;
 
 GlobalVector globalMomentum() const;
 
 GlobalPoint globalPosition() const;
 
 TrackCharge particleCharge() const;

/**
 * KinematicState -> FreeTrajectoryState 
 * converter
 */
 FreeTrajectoryState freeTrajectoryState() const;
 
 bool isValid() const
 {return vl;}



 
private:

 KinematicParameters param;

 KinematicParametersError err;

 TrackCharge ch;
 
 bool vl;
};
#endif
