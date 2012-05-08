#ifndef TrackPropagation_Geant4ePropagator_h
#define TrackPropagation_Geant4ePropagator_h


//CMS includes
// - Propagator
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

// - Geant4e
#include "G4ErrorPropagatorManager.hh"



class Geant4eSteppingAction;

/** Propagator based on the Geant4e package. Uses the Propagator class
 *  in the TrackingTools/GeomPropagators package to define the interface.
 *  See that class for more details.
 */

class Geant4ePropagator GCC11_FINAL : public Propagator {

 public:
  /** Constructor. Takes as arguments:
   *  * The magnetic field
   *  * The particle name whose properties will be used in the propagation. Without the charge, i.e. "mu", "pi", ...
   *  * The propagation direction. It may be: alongMomentum, oppositeToMomentum
   */
  Geant4ePropagator(const MagneticField* field = 0,
		    const char* particleName = "mu",
		    PropagationDirection dir = alongMomentum);

  virtual ~Geant4ePropagator();


  /** Propagate from a free state (e.g. position and momentum in 
   *  in global cartesian coordinates) to a surface.
   */

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState& ftsStart, const Plane& pDest) const;

  virtual TrajectoryStateOnSurface 
  propagate (const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const;

  /** Propagate from a state on surface (e.g. position and momentum in 
   *  in global cartesian coordinates associated with a layer) to a surface.
   */
  virtual TrajectoryStateOnSurface
    propagate (const TrajectoryStateOnSurface& tsos, const Plane& plane) const;
  
  virtual TrajectoryStateOnSurface
    propagate (const TrajectoryStateOnSurface& tsos, const Cylinder& cyl) const; 

  /** The methods propagateWithPath() are identical to the corresponding
   *  methods propagate() in what concerns the resulting 
   *  TrajectoryStateOnSurface, but they provide in addition the
   *  exact path length along the trajectory.
   */

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Plane&) const;

  virtual std::pair< TrajectoryStateOnSurface, double> 
  propagateWithPath (const FreeTrajectoryState&, const Cylinder&) const;


   virtual std::pair< TrajectoryStateOnSurface, double>  
   propagateWithPath (const TrajectoryStateOnSurface&, const Plane&) const; 

   virtual std::pair< TrajectoryStateOnSurface, double>  
   propagateWithPath (const TrajectoryStateOnSurface&, const Cylinder&) const; 


  virtual Geant4ePropagator* clone() const {return new Geant4ePropagator(*this);}

  virtual const MagneticField* magneticField() const {return theField;}



 protected:

  typedef std::pair<TrajectoryStateOnSurface, double> TsosPP;


  //Magnetic field
  const MagneticField* theField;

  //Name of the particle whose properties will be used in the propagation
  std::string theParticleName; 

  //The Geant4e manager. Does the real propagation
  G4ErrorPropagatorManager* theG4eManager;

  //A G4 stepping action to find out the track length
  mutable Geant4eSteppingAction* theSteppingAction;

};


#endif
