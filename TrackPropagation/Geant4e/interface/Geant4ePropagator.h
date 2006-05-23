#ifndef TrackPropagation_Geant4ePropagator_h
#define TrackPropagation_Geant4ePropagator_h


#include "TrackingTools/GeomPropagators/interface/Propagator.h"


//Geant4e includes
#include "G4eManager.hh"



/** Propagator based on the Geant4e package. Uses the Propagator class
 *  in the TrackingTools/GeomPropagators package to define the interface.
 *  See that class for more details.
 */

class Geant4ePropagator: public Propagator {

 public:
  /** Constructor. Takes as arguments:
   *  * The propagation direction. It may be: alongMomentum, oppositeToMomentum
   *  * The particle name whose properties will be used in the propagation without the charge.
   */
  Geant4ePropagator(PropagationDirection dir = alongMomentum, 
		    const MagneticField* field = 0,
		    const char* particleName = "mu");
  virtual ~Geant4ePropagator() {}


  /** Propagate from a free state (e.g. position and momentum in 
   *  in global cartesian coordinates) to a surface.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
/*   virtual TrajectoryStateOnSurface  */
/*   propagate (const FreeTrajectoryState&, const Surface&) const; */

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Plane&) const;

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState&, const Cylinder&) const;

  /** The following three methods are equivalent to the corresponding
   *  methods above,
   *  but if the starting state is a TrajectoryStateOnSurface, it's better 
   *  to use it as such rather than use just the FreeTrajectoryState
   *  part. It may help some concrete propagators.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
/*   virtual TrajectoryStateOnSurface  */
/*   propagate (const TrajectoryStateOnSurface&, const Surface&) const; */

/*   virtual TrajectoryStateOnSurface  */
/*   propagate (const TrajectoryStateOnSurface&, const Plane&) const; */

/*   virtual TrajectoryStateOnSurface  */
/*   propagate (const TrajectoryStateOnSurface&, const Cylinder&) const; */

  /** The methods propagateWithPath() are identical to the corresponding
   *  methods propagate() in what concerns the resulting 
   *  TrajectoryStateOnSurface, but they provide in addition the
   *  exact path length along the trajectory.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
/*   virtual std::pair< TrajectoryStateOnSurface, double>  */
/*   propagateWithPath (const FreeTrajectoryState&, const Surface&) const; */

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Plane&) const;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const;

  /** The following three methods are equivalent to the corresponding
   *  methods above,
   *  but if the starting state is a TrajectoryStateOnSurface, it's better 
   *  to use it as such rather than use just the FreeTrajectoryState
   *  part. It may help some concrete propagators.
   */

  /** Only use the generic method if the surface type (plane or cylinder)
   *  is not known at the calling point.
   */
/*   virtual std::pair< TrajectoryStateOnSurface, double>  */
/*   propagateWithPath (const TrajectoryStateOnSurface&, const Surface&) const; */

/*   virtual std::pair< TrajectoryStateOnSurface, double>  */
/*   propagateWithPath (const TrajectoryStateOnSurface&, const Plane&) const; */

/*   virtual std::pair< TrajectoryStateOnSurface, double>  */
/*   propagateWithPath (const TrajectoryStateOnSurface&, const Cylinder&) const; */


  virtual Geant4ePropagator* clone() const {return new Geant4ePropagator(*this);}

  virtual const MagneticField* magneticField() const {return theField;}



 protected:
  typedef std::pair<TrajectoryStateOnSurface, double> TsosPP;
  const MagneticField* theField;

  std::string theParticleName;

  G4eManager* theG4eManager;
};


#endif
