#ifndef TrackPropagation_Geant4ePropagator_h
#define TrackPropagation_Geant4ePropagator_h

#include <memory>

//CMS includes
// - Propagator
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

// - Geant4e
#include "G4ErrorPropagatorManager.hh"
#include "G4ErrorSurfaceTarget.hh"
#include "G4ErrorPropagatorData.hh"

/** Propagator based on the Geant4e package. Uses the Propagator class
 *  in the TrackingTools/GeomPropagators package to define the interface.
 *  See that class for more details.
 */

class Geant4ePropagator: public Propagator {

public:
	/** Constructor. Takes as arguments:
	 *  * The magnetic field
	 *  * The particle name whose properties will be used in the propagation. Without the charge, i.e. "mu", "pi", ...
	 *  * The propagation direction. It may be: alongMomentum, oppositeToMomentum
	 */
	Geant4ePropagator(const MagneticField* field = 0, std::string particleName =
			"mu", PropagationDirection dir = alongMomentum);

	virtual ~Geant4ePropagator() override;

	/** Propagate from a free state (e.g. position and momentum in
	 *  in global cartesian coordinates) to a surface.
	 */
/*
	virtual TrajectoryStateOnSurface
	propagate(const FreeTrajectoryState& ftsStart, const Plane& pDest) const override;

	virtual TrajectoryStateOnSurface
	propagate(const FreeTrajectoryState& ftsStart, const Cylinder& cDest) const override;
*/
	/** Propagate from a state on surface (e.g. position and momentum in
	 *  in global cartesian coordinates associated with a layer) to a surface.
	 */
/*        
	virtual TrajectoryStateOnSurface
	propagate(const TrajectoryStateOnSurface& tsos, const Plane& plane) const override;

	virtual TrajectoryStateOnSurface
	propagate(const TrajectoryStateOnSurface& tsos, const Cylinder& cyl) const override;
*/
	/** The methods propagateWithPath() are identical to the corresponding
	 *  methods propagate() in what concerns the resulting
	 *  TrajectoryStateOnSurface, but they provide in addition the
	 *  exact path length along the trajectory.
	 *  All of these method calls are internally mapped to
	 */

	virtual std::pair<TrajectoryStateOnSurface, double>
	propagateWithPath(const FreeTrajectoryState&, const Plane&) const override;

	virtual std::pair<TrajectoryStateOnSurface, double>
	propagateWithPath(const FreeTrajectoryState&, const Cylinder&) const override;

	virtual std::pair<TrajectoryStateOnSurface, double>
	propagateWithPath(const TrajectoryStateOnSurface&, const Plane&) const override;

	virtual std::pair<TrajectoryStateOnSurface, double>
	propagateWithPath(const TrajectoryStateOnSurface&, const Cylinder&) const override;

	virtual Geant4ePropagator* clone() const override {
		return new Geant4ePropagator(*this);
	}

	virtual const MagneticField* magneticField() const override {
		return theField;
	}

private:

	typedef std::pair<TrajectoryStateOnSurface, double> TsosPP;
	typedef std::pair< bool, std::shared_ptr< G4ErrorTarget > > ErrorTargetPair;

	//Magnetic field
	const MagneticField* theField;

	//Name of the particle whose properties will be used in the propagation
	std::string theParticleName;

	//The Geant4e manager. Does the real propagation
	G4ErrorPropagatorManager* theG4eManager;
	G4ErrorPropagatorData* theG4eData;

	// Transform a CMS Reco detector surface into a Geant4 Target for the error propagation
	template <class SurfaceType>
	ErrorTargetPair transformToG4SurfaceTarget( const SurfaceType& pDest, bool moveTargetToEndOfSurface ) const;

	// generates the Geant4 name for a particle from the
	// string stored in theParticleName ( set via constructor )
	// and the particle charge.
	// 'mu' as a basis for muon becomes 'mu+' or 'mu-', depening on the charge
	// This method only supports neutral and +/- 1e charges so far
	//
	// returns the generated string
	std::string generateParticleName ( int charge ) const;

	// flexible method which performs the actual propagation either for a plane or cylinder
	// surface type
	//
	// returns TSOS after the propagation and the path length
	template <class SurfaceType>
	std::pair<TrajectoryStateOnSurface, double>
	propagateGeneric(const FreeTrajectoryState& ftsStart, const SurfaceType& pDest) const;

	// saves the Geant4 propagation direction (Forward or Backward) in the provided variable
	// reference mode and returns true if the propagation direction could be set
	template <class SurfaceType>
	bool configurePropagation ( G4ErrorMode & mode, SurfaceType const& pDest,
								GlobalPoint const& cmsInitPos,GlobalVector const& cmsInitMom ) const;

	// special case to determine the propagation direction if the CMS propagation direction 'anyDirection'
	// was set.
	// This method is called by configurePropagation and provides specific implementations for Plane
	// and Cylinder classes
	template <class SurfaceType>
	bool configureAnyPropagation ( G4ErrorMode & mode, SurfaceType const& pDest,
								   GlobalPoint const& cmsInitPos,
								   GlobalVector const& cmsInitMom  ) const;

	// Ensure Geant4 Error propagation is initialized, if not done so, yet
	// if the forceInit parameter is set to true, the initialization is performed,
	// even if already done before.
	// This can be necessary, when Geant4 needs to read in a new MagneticField object,
	// which changed during lumi section crossing
	void ensureGeant4eIsInitilized( bool forceInit ) const;

	// returns the name of the SurfaceType. Mostly for debug outputs
	template < class SurfaceType >
	std::string getSurfaceType ( SurfaceType const& surface) const;

	void debugReportPlaneSetup ( GlobalPoint const& posPlane, HepGeom::Point3D<double> const& surfPos,
								 GlobalVector const& normalPlane, HepGeom::Normal3D<double> const& surfNorm,
								 const Plane& pDest ) const;

	template <class SurfaceType>
	void  debugReportTrackState ( std::string const& currentContext, GlobalPoint const& cmsInitPos,
								  CLHEP::Hep3Vector const& g4InitPos,
								  GlobalVector const& cmsInitMom, CLHEP::Hep3Vector const& g4InitMom,
								  const SurfaceType& pDest) const;
};

#endif
