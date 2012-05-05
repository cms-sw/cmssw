#ifndef _CR_MATERIALEFFECTSUPDATOR_H_
#define _CR_MATERIALEFFECTSUPDATOR_H_

/** \class MaterialEffectsUpdator
 *  Interface for adding material effects during propagation.
 *  Updates to TrajectoryStateOnSurface are implemented 
 *  in this class.
 *  Ported from ORCA.
 *
 *  $Date: 2011/04/16 14:51:22 $
 *  $Revision: 1.13 $
 *  \author todorov, cerati
 */

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"

#include "FWCore/Utilities/interface/GCC11Compatibility.h"
#include "FWCore/Utilities/interface/Visibility.h"

class MaterialEffectsUpdator
{  
public:
  /** Constructor with explicit mass hypothesis
   */
  MaterialEffectsUpdator ( double mass );
  virtual ~MaterialEffectsUpdator ();

  /** Updates TrajectoryStateOnSurface with material effects
   *    (momentum and covariance matrix are potentially affected.
   */
  virtual TrajectoryStateOnSurface updateState (const TrajectoryStateOnSurface& TSoS, 
						const PropagationDirection propDir) const;

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
  virtual double deltaP (const TrajectoryStateOnSurface& TSoS, const PropagationDirection propDir) const;


  /** Contribution to covariance matrix (in local co-ordinates) from material effects.
   */
  virtual const AlgebraicSymMatrix55 &deltaLocalError (const TrajectoryStateOnSurface& TSoS, 
						       const PropagationDirection propDir) const;

  /** Particle mass assigned at construction.
   */
  inline double mass () const {
    return theMass;
  }

  virtual MaterialEffectsUpdator* clone()  const = 0;

 private:
  // here comes the actual computation of the values
  virtual void compute (const TrajectoryStateOnSurface&, const PropagationDirection) const dso_internal = 0;

  // check of arguments for use with cached values
  bool newArguments (const TrajectoryStateOnSurface & TSoS, PropagationDirection  propDir) const dso_internal;
  
 private:
  double theMass;

  // chache previous call state
  mutable double theLastOverP;
  mutable double theLastDxdz;
  mutable float  theLastRL;
  mutable PropagationDirection theLastPropDir;


protected:  
  mutable double theDeltaP;
  mutable AlgebraicSymMatrix55 theDeltaCov;
  static  AlgebraicSymMatrix55  theNullMatrix;
};

#endif
