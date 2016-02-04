#ifndef GsfMaterialEffectsUpdator_h_
#define GsfMaterialEffectsUpdator_h_

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"

#include <vector>

/**
 * Interface for adding material effects during propagation
 *   as a Gaussian mixture. Similar to MaterialEffectsUpdator,
 *   but returns MultiTrajectoryState.
 */
class GsfMaterialEffectsUpdator
{  
public:
  /** Constructor with explicit mass hypothesis
   */
  GsfMaterialEffectsUpdator ( float mass ) :
    theMass(mass) {}

  virtual ~GsfMaterialEffectsUpdator () {}

  /** Updates TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  virtual TrajectoryStateOnSurface updateState (const TrajectoryStateOnSurface& TSoS, 
						const PropagationDirection propDir) const;
  /** Weights of components.
   */
  virtual std::vector<double> weights (const TrajectoryStateOnSurface& TSoS, 
				       const PropagationDirection propDir) const {
    // check for material
    if ( !TSoS.surface().mediumProperties() )  return std::vector<double>();
    // check for change (avoid using compute method if possible)
    if ( newArguments(TSoS,propDir) )  compute(TSoS,propDir);
    return theWeights;
  }
  /** Change in |p| from material effects.
   */
  virtual std::vector<double> deltaPs (const TrajectoryStateOnSurface& TSoS, 
				       const PropagationDirection propDir) const {
    // check for material
    if ( !TSoS.surface().mediumProperties() )  return std::vector<double>();
    // check for change (avoid using compute method if possible)
    if ( newArguments(TSoS,propDir) )  compute(TSoS,propDir);
    return theDeltaPs;
  }
  /** Contribution to covariance matrix (in local co-ordinates) from material effects.
   */
  virtual std::vector<AlgebraicSymMatrix55> deltaLocalErrors (const TrajectoryStateOnSurface& TSoS, 
							    const PropagationDirection propDir) const {
    // check for material
    if ( !TSoS.surface().mediumProperties() )  return std::vector<AlgebraicSymMatrix55>();
    // check for change (avoid using compute method if possible)
    if ( newArguments(TSoS,propDir) )  compute(TSoS,propDir);
    return theDeltaCovs;
  }  
  /** Particle mass assigned at construction.
   */
  inline float mass () const {
    return theMass;
  }

  virtual GsfMaterialEffectsUpdator* clone()  const = 0;

private:
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection) const = 0;

protected:
  // check of arguments for use with cached values
  virtual bool newArguments (const TrajectoryStateOnSurface&, const PropagationDirection) const {
    return true;
  }
  // storage of arguments for later use
  virtual void storeArguments (const TrajectoryStateOnSurface&, const PropagationDirection) const {
  }
  
private:
  float theMass;


protected:  
  mutable std::vector<double> theWeights;
  mutable std::vector<double> theDeltaPs;
  mutable std::vector<AlgebraicSymMatrix55> theDeltaCovs;
};

#endif
