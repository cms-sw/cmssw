#ifndef CommonDet_Propagator_H
#define CommonDet_Propagator_H

#include "DataFormats/TrajectorySeed/interface/PropagationDirection.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include <utility>
#include <memory>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

class FreeTrajectoryState;
class TrajectoryStateOnSurface;
class Plane;
class Cylinder;
class Surface;
class MagneticField;

namespace reco{class BeamSpot;}

/** Basic tool for "propagation" of trajectory states to surfaces.
 *  If the starting state has an error matrix the errors will be also
 *  propagated. If you want to propagate just the parameters,
 *  construct a starting state that does not have errors.
 *  In case of propagation failure (e.g. when the trajectory does
 *  not cross the destination surface) and invalid state is returned.
 *  Always check the returned state with isValid() before using it!
 *
 *  The propagation can be "alongMomentum" or "oppositeToMomentum"
 *  (see setPropagationDirection() below). The difference between the two
 *  is the sign of energy loss: the trajectory momentum decreases
 *  "alongMomentum" and increases "oppositeToMomentum".
 *  In both directions extrapolation errors and multiple scattering errors
 *  increase. Propagation "oppositeToMomentum" is convenient for
 *  fitting a track "backwards", sterting from the last measurement.
 *
 *  The propagator interface promises to take you to "any surface"
 *  but you should check the concrete propagator you are using for
 *  additional limitations.
 */

class Propagator {
public:

  Propagator (PropagationDirection dir = alongMomentum) :
    theDir(dir) {}
  virtual ~Propagator();

  /** Propagate from a free state (e.g. position and momentum in
   *  in global cartesian coordinates) to a surface.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
  virtual TrajectoryStateOnSurface
  propagate (const FreeTrajectoryState&, const Surface&) const;

  virtual TrajectoryStateOnSurface
  propagate (const FreeTrajectoryState&, const Plane&) const = 0;

  virtual TrajectoryStateOnSurface
  propagate (const FreeTrajectoryState&, const Cylinder&) const = 0;

  /** The following three methods are equivalent to the corresponding
   *  methods above,
   *  but if the starting state is a TrajectoryStateOnSurface, it's better
   *  to use it as such rather than use just the FreeTrajectoryState
   *  part. It may help some concrete propagators.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
  virtual TrajectoryStateOnSurface
  propagate (const TrajectoryStateOnSurface&, const Surface&) const;

  virtual TrajectoryStateOnSurface
  propagate (const TrajectoryStateOnSurface&, const Plane&) const;

  virtual TrajectoryStateOnSurface
  propagate (const TrajectoryStateOnSurface&, const Cylinder&) const;

  virtual FreeTrajectoryState
  propagate(const FreeTrajectoryState&,
	    const reco::BeamSpot&) const;

  /** The methods propagateWithPath() are identical to the corresponding
   *  methods propagate() in what concerns the resulting
   *  TrajectoryStateOnSurface, but they provide in addition the
   *  exact path length along the trajectory.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
  virtual std::pair< TrajectoryStateOnSurface, double>
  propagateWithPath (const FreeTrajectoryState&, const Surface&) const;

  virtual std::pair< TrajectoryStateOnSurface, double>
  propagateWithPath (const FreeTrajectoryState&, const Plane&) const = 0;

  virtual std::pair< TrajectoryStateOnSurface, double>
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const=0;

  /** The following three methods are equivalent to the corresponding
   *  methods above,
   *  but if the starting state is a TrajectoryStateOnSurface, it's better
   *  to use it as such rather than use just the FreeTrajectoryState
   *  part. It may help some concrete propagators.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
  virtual std::pair< TrajectoryStateOnSurface, double>
  propagateWithPath (const TrajectoryStateOnSurface&, const Surface&) const;

  virtual std::pair< TrajectoryStateOnSurface, double>
  propagateWithPath (const TrajectoryStateOnSurface&, const Plane&) const;

  virtual std::pair< TrajectoryStateOnSurface, double>
  propagateWithPath (const TrajectoryStateOnSurface&, const Cylinder&) const;

  virtual std::pair<FreeTrajectoryState, double>
    propagateWithPath(const FreeTrajectoryState&,
                      const GlobalPoint&, const GlobalPoint&) const;

  /** The propagation direction can now be set for every propagator.
   *  There is no more distinction between unidirectional and bidirectional
   *  at class level. The value "anyDiriction" for PropagationDirection
   *  provides the functionality of the ex-BidirectionalPropagator.
   *  The values "alongMomentum" and "oppositeToMomentum" provide the
   *  functionality of the ex-UnidirectionalPropagator.
   */
  virtual void setPropagationDirection(PropagationDirection dir) {
    theDir = dir;
  }

  /** Returns the current value of the propagation direction.
   *  If you need to know the actual direction used for a given propagation
   *  in case "propagationDirection() == anyDirection",
   *  you should use propagateWithPath. A positive sign of
   *  path lengt means "alongMomentum", an egeative sign means
   *  "oppositeToMomentum".
   */
  virtual PropagationDirection propagationDirection() const GCC11_FINAL {
    return theDir;
  }

  /** Set the maximal change of direction (integrated along the path)
   *  for any single propagation.
   *  If reaching of the destination surface requires change of direction that exceeds
   *  this value the Propagator returns an invalid state.
   *  For example, a track may reach a forward plane after many spirals,
   *  which may be undesirable for a track reconstructor. Setting this value
   *  to pi will force the propagation to fail.
   *  The default value is "no limit". The method returnd true if the concrete propagator
   *  respects the limit, false otherwise.
   */
  virtual bool setMaxDirectionChange( float phiMax) { return false;}

  virtual Propagator * clone() const = 0;

  virtual const MagneticField* magneticField() const = 0;

private:

  PropagationDirection theDir;
};

// Put here declaration of helper function, so that it is
// automatically included in all proper places w/o having to add an
// additional include file. Keep implementation separate, to avoid
// multiple definition of the same symbol in all cc inlcuding this
// file.
std::unique_ptr<Propagator> SetPropagationDirection (Propagator const & iprop,
                                                     PropagationDirection dir);

#endif // CommonDet_Propagator_H
