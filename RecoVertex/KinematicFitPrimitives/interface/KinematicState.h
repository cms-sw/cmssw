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

class KinematicState {
public:
  /**
   * Default constructor for internal
   * KinematicFitPrimitives library needs
   * only
   */
  KinematicState() { vl = false; }

  /**
   * Constructor taking directly KinematicParameters
   * KinematicError and Charge. To be used with
   * proper KinematicStateBuilder.
   */

  KinematicState(const KinematicParameters& parameters,
                 const KinematicParametersError& error,
                 const TrackCharge& charge,
                 const MagneticField* field);

  KinematicState(const FreeTrajectoryState& state, const ParticleMass& mass, float m_sigma)
      : fts(state),
        param(state.position().x(),
              state.position().y(),
              state.position().z(),
              state.momentum().x(),
              state.momentum().y(),
              state.momentum().z(),
              mass),
        err(state.cartesianError(), m_sigma),
        vl(true) {}

  bool operator==(const KinematicState& other) const;

  /**
   * The mass of the particle
   */
  ParticleMass mass() const { return param.vector()[6]; }

  /**
 * Access methods to parameters
 * and private data
 */

  KinematicParameters const& kinematicParameters() const { return param; }

  KinematicParametersError const& kinematicParametersError() const { return err; }

  GlobalTrajectoryParameters const& trajectoryParameters() const { return fts.parameters(); }

  GlobalVector globalMomentum() const { return fts.momentum(); }

  GlobalPoint globalPosition() const { return fts.position(); }

  TrackCharge particleCharge() const { return fts.charge(); }

  /**
 * KinematicState -> FreeTrajectoryState 
 * converter
 */
  FreeTrajectoryState freeTrajectoryState() const { return fts; }

  bool isValid() const { return vl; }

  GlobalVector magneticFieldInInverseGeV(const GlobalPoint& x) const {
    return trajectoryParameters().magneticFieldInInverseGeV(x);
  }
  GlobalVector magneticFieldInInverseGeV() const { return trajectoryParameters().magneticFieldInInverseGeV(); }

  const MagneticField* magneticField() const { return &trajectoryParameters().magneticField(); }

private:
  FreeTrajectoryState fts;
  KinematicParameters param;
  KinematicParametersError err;

  bool vl;
};
#endif
