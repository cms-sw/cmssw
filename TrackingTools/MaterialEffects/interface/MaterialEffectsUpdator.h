#ifndef _CR_MATERIALEFFECTSUPDATOR_H_
#define _CR_MATERIALEFFECTSUPDATOR_H_

/** \class MaterialEffectsUpdator
 *  Interface for adding material effects during propagation.
 *  Updates to TrajectoryStateOnSurface are implemented 
 *  in this class.
 *  Ported from ORCA.
 *
 *  $Date: 2007/08/22 23:05:20 $
 *  $Revision: 1.6 $
 *  \author todorov, cerati
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
class MaterialEffectsUpdator
{  
public:
  /** Constructor with explicit mass hypothesis
   */
  MaterialEffectsUpdator ( float mass ) :
    theMass(mass),
    theDeltaP(0.),
    theDeltaCov() {}
  /** Default constructor (mass from configurable)
   */
  //  MaterialEffectsUpdator ();

  virtual ~MaterialEffectsUpdator () {}

  /** Updates TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  virtual TrajectoryStateOnSurface updateState (const TrajectoryStateOnSurface& TSoS, 
						const PropagationDirection propDir) const {
        TrajectoryStateOnSurface shallowCopy = TSoS;
        // A TSOS is a proxy. Its contents will be really copied only if/when the updateStateInPlace attempts to change them
        return updateStateInPlace(shallowCopy, propDir) ? shallowCopy : TrajectoryStateOnSurface();
  }

  /** Updates in place TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected)
   *  Will return 'false' if the 'updateState' would have returned an invalid TSOS
   *  Note that the TSoS might be very well unchanged from this method 
   *  (just like 'updateState' can return the same TSOS)
   */
  virtual bool updateStateInPlace (TrajectoryStateOnSurface& TSoS, 
				   const PropagationDirection propDir) const;

 
  /** Change in |p| from material effects.
   */
  virtual double deltaP (const TrajectoryStateOnSurface& TSoS, const PropagationDirection propDir) const {
    // check for material
    if ( !TSoS.surface().mediumProperties() )  return 0.;
    // check for change (avoid using compute method if possible)
    if ( newArguments(TSoS,propDir) )  compute(TSoS,propDir);
    return theDeltaP;
  }


  /** Contribution to covariance matrix (in local co-ordinates) from material effects.
   */
  virtual const AlgebraicSymMatrix55 &deltaLocalError (const TrajectoryStateOnSurface& TSoS, 
					      const PropagationDirection propDir) const {
    // check for material
    if ( !TSoS.surface().mediumProperties() )  return theNullMatrix;
    // check for change (avoid using compute method if possible)
    if ( newArguments(TSoS,propDir) )  compute(TSoS,propDir);
    return theDeltaCov;
  }  
  /** Particle mass assigned at construction.
   */
  inline float mass () const {
    return theMass;
  }

  virtual MaterialEffectsUpdator* clone()  const = 0;

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
  mutable double theDeltaP;
  mutable AlgebraicSymMatrix55 theDeltaCov;
  static  AlgebraicSymMatrix55  theNullMatrix;
};

#endif
