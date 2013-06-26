#ifndef TrajectoryStateOnSurface_H
#define TrajectoryStateOnSurface_H

#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

#include <iosfwd>

/** As the class name suggests, this class encapsulates the state of a
 *  trajectory on a surface.
 *  The class is a reference counting proxy to the actual state, in other words
 *  it takes no more space than a pointer. Therefore it should be used by value.
 */

class TrajectoryStateOnSurface : private  BasicTrajectoryState::Proxy {

  typedef BasicTrajectoryState::SurfaceType SurfaceType;
  typedef BasicTrajectoryState::SurfaceSide SurfaceSide;
  typedef BasicTrajectoryState::Proxy             Base;

public:
  // construct
  TrajectoryStateOnSurface() {}
  /// Constructor from one of the basic states

  // invalid state
  explicit TrajectoryStateOnSurface(const SurfaceType& aSurface);


  TrajectoryStateOnSurface( BasicTrajectoryState* p) : Base(p) {}
  /** Constructor from FTS and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  TrajectoryStateOnSurface( const FreeTrajectoryState& fts,
			    const SurfaceType& aSurface,
			    const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  TrajectoryStateOnSurface( const GlobalTrajectoryParameters& gp,
			    const SurfaceType& aSurface,
			    const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  TrajectoryStateOnSurface( const GlobalTrajectoryParameters& gp,
			    const CartesianTrajectoryError& err,
			    const SurfaceType& aSurface,
			    const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  TrajectoryStateOnSurface( const GlobalTrajectoryParameters& gp,
			    const CurvilinearTrajectoryError& err,
			    const SurfaceType& aSurface,
			    const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface, 
			    double weight = 1.);
  /** Constructor from global parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  TrajectoryStateOnSurface( const GlobalTrajectoryParameters& gp,
			    const CurvilinearTrajectoryError& err,
			    const SurfaceType& aSurface,
			    double weight);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  TrajectoryStateOnSurface( const LocalTrajectoryParameters& p,
			    const SurfaceType& aSurface,
			    const MagneticField* field,
			    const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  TrajectoryStateOnSurface( const LocalTrajectoryParameters& p,
			    const LocalTrajectoryError& err,
			    const SurfaceType& aSurface,
			    const MagneticField* field,
			    const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface, 
			    double weight = 1.);
  /** Constructor from local parameters, errors and surface. For multi-states the
   *  weight should be specified explicitely. For backward compatibility without
   *  specification of the side of the surface.
   */
  TrajectoryStateOnSurface( const LocalTrajectoryParameters& p,
			    const LocalTrajectoryError& err,
			    const SurfaceType& aSurface,
			    const MagneticField* field,
			    double weight);

  ~TrajectoryStateOnSurface() {}

#if defined( __GXX_EXPERIMENTAL_CXX0X__)

 TrajectoryStateOnSurface(TrajectoryStateOnSurface const & rh)  noexcept :
    Base(rh){}


  TrajectoryStateOnSurface(TrajectoryStateOnSurface && rh)  noexcept :
    Base(std::forward<Base>(rh)){}
    
  TrajectoryStateOnSurface & operator=(TrajectoryStateOnSurface && rh)  noexcept {
    Base::swap(rh);
    return *this;
  }

 TrajectoryStateOnSurface & operator=(TrajectoryStateOnSurface const & rh)  noexcept {
    Base::operator=(rh);
    return *this;
  }


#endif

  void swap(TrajectoryStateOnSurface & rh)  noexcept {
    Base::swap(rh);
  }


  bool isValid() const {
    return Base::isValid() && data().isValid();
  }

  bool hasError() const {
    return data().hasError();
  }

  FreeTrajectoryState* freeState(bool withErrors = true) const {
    return data().freeTrajectoryState();
  }

  FreeTrajectoryState* freeTrajectoryState(bool withErrors = true) const {
    return freeState();
  }

  const MagneticField *magneticField() const { return data().magneticField(); }

  const GlobalTrajectoryParameters& globalParameters() const {
    return data().globalParameters();
  }
  GlobalPoint globalPosition() const {
    return data().globalPosition();
  }
  GlobalVector globalMomentum() const {
    return data().globalMomentum();
  }
  GlobalVector globalDirection() const {
    return data().globalDirection();
  }
  TrackCharge charge() const {
    return data().charge();
  }
  double signedInverseMomentum() const {
    return data().signedInverseMomentum();
  }
  double transverseCurvature() const {
    return data().transverseCurvature();
  }
  const CartesianTrajectoryError cartesianError() const {
    return data().cartesianError();
  }
  const CurvilinearTrajectoryError& curvilinearError() const {
    return data().curvilinearError();
  }
  const LocalTrajectoryParameters& localParameters() const {
    return data().localParameters();
  }
  LocalPoint localPosition() const {
    return data().localPosition();
  }
  LocalVector localMomentum() const {
    return data().localMomentum();
  }
  LocalVector localDirection() const {
    return data().localDirection();
  }
  const LocalTrajectoryError& localError() const {
    return data().localError();
  }
  const SurfaceType& surface() const {
    return data().surface();
  }

  double weight() const {return data().weight();} 

  void rescaleError(double factor) {
    unsharedData().rescaleError(factor);
  }

  std::vector<TrajectoryStateOnSurface> components() const {
    return data().components();
  }
  /*
  std::vector<TrajectoryStateOnSurface> components() const {
    std::vector<BasicTrajectoryState::RCPtr> c( data().components());
    std::vector<TrajectoryStateOnSurface> result; 
    result.reserve(c.size());
    for (std::vector<BasicTrajectoryState::RCPtr>::iterator i=c.begin();
	 i != c.end(); i++) result.push_back(&(**i));
    return result;
  }
  */

  /// Position relative to material, defined relative to momentum vector.
  SurfaceSide surfaceSide() const {
    return data().surfaceSide();
  }

  /** Mutator from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   *  If the underlying trajectory state supports updates, it will be updated, otherwise this method will
   *  just behave like creating a new TSOS (which will make a new BasicSingleTrajectoryState)
   */
  void update( const LocalTrajectoryParameters& p,
	       const SurfaceType& aSurface,
	       const MagneticField* field,
	       const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);
  /** Mutator from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   *  If the underlying trajectory state supports updates, it will be updated, otherwise this method will
   *  just behave like creating a new TSOS (which will make a new BasicSingleTrajectoryState)
   */
  void update( const LocalTrajectoryParameters& p,
	       const LocalTrajectoryError& err,
               const SurfaceType& aSurface,
               const MagneticField* field,
               const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface, 
               double weight = 1.);

};

inline void swap( TrajectoryStateOnSurface & rh,  TrajectoryStateOnSurface & lh) {
  // use base swap
  rh.swap(lh);
}

std::ostream& operator<<(std::ostream& os, const TrajectoryStateOnSurface & tsos);
#endif
