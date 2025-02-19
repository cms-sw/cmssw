#ifndef KinematicState_H
#define KinematicState_H

#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParameters.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParametersError.h"
#include "RecoVertex/KinematicFitPrimitives/interface/ParticleMass.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "MagneticField/Engine/interface/MagneticField.h"

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
 KinematicState(const KinematicParameters& parameters,
 	const KinematicParametersError& error, const TrackCharge& charge,
	const MagneticField* field);
						       
 bool operator==(const KinematicState& other) const;

  /**
   * The mass of the particle
   */
  ParticleMass mass() const {return param.vector()[6];}

/**
 * Access methods to parameters
 * and private data
 */

KinematicParameters const & kinematicParameters() const {return param;}

KinematicParametersError const & kinematicParametersError() const {return err;}

GlobalVector globalMomentum() const {return param.momentum();}

GlobalPoint  globalPosition() const {return param.position();}

TrackCharge particleCharge() const {return ch;}



/**
 * KinematicState -> FreeTrajectoryState 
 * converter
 */
 FreeTrajectoryState freeTrajectoryState() const;
 
 bool isValid() const
 {return vl;}

  const MagneticField* magneticField() const {return theField;}

 
private:

  const MagneticField* theField;
  KinematicParameters param;
  KinematicParametersError err;
  TrackCharge ch;
 
  bool vl;
};
#endif
