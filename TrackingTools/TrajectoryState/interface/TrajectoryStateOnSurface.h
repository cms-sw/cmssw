#ifndef TrajectoryStateOnSurface_H
#define TrajectoryStateOnSurface_H

#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"
#include "TrackingTools/TrajectoryState/interface/BasicSingleTrajectoryState.h"


#include <iosfwd>

/** As the class name suggests, this class encapsulates the state of a
 *  trajectory on a surface.
 *  The class is a reference counting proxy to the actual state, in other words
 *  it takes no more space than a pointer. Therefore it should be used by value.
 */

class TrajectoryStateOnSurface : private  BasicTrajectoryState::Proxy {

  typedef BasicTrajectoryState                    BTSOS;
  typedef BasicTrajectoryState::SurfaceType SurfaceType;
  typedef BasicTrajectoryState::SurfaceSide SurfaceSide;
  typedef BasicTrajectoryState::Proxy              Base;

public:
  // construct
  TrajectoryStateOnSurface() {}
  /// Constructor from one of the basic states
  explicit TrajectoryStateOnSurface( Base::pointer p) : Base(p) {}
  explicit TrajectoryStateOnSurface( BasicTrajectoryState* p) : Base(p) {}
  explicit TrajectoryStateOnSurface( BasicSingleTrajectoryState* p) : Base(p) {}

  ~TrajectoryStateOnSurface() {}

 TrajectoryStateOnSurface(TrajectoryStateOnSurface & rh)  noexcept :
    Base(rh){}

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

  template<typename... Args>
  explicit TrajectoryStateOnSurface(Args && ...args) : Base(BTSOS::churn<BasicSingleTrajectoryState>(std::forward<Args>(args)...)){}

  void swap(TrajectoryStateOnSurface & rh)  noexcept {
    Base::swap(rh);
  }


  bool isValid() const {
    return Base::isValid() && data().isValid();
  }

  bool hasError() const {
    return data().hasError();
  }

  FreeTrajectoryState const* freeState(bool withErrors = true) const {
    return data().freeTrajectoryState();
  }

  FreeTrajectoryState const* freeTrajectoryState(bool withErrors = true) const {
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

  using	Components = BasicTrajectoryState::Components;
  Components const & components() const {
    return data().components();
  }
  bool singleState() const { return data().singleState();}


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
	       SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);

  void update( const LocalTrajectoryParameters& p,
               SurfaceSide side) { unsharedData().update(p, side);}

  void update( const LocalTrajectoryParameters& p,
               const LocalTrajectoryError& err,
               SurfaceSide side) {unsharedData().update(p, err, side);}

  /** Mutator from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   *  If the underlying trajectory state supports updates, it will be updated, otherwise this method will
   *  just behave like creating a new TSOS (which will make a new BasicSingleTrajectoryState)
   */
  void update(const LocalTrajectoryParameters& p,
	      const LocalTrajectoryError& err,
	      const SurfaceType& aSurface,
	      const MagneticField* field,
	      SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);

  CurvilinearTrajectoryError & setCurvilinearError() {
        return sharedData().setCurvilinearError();
  }



};

inline void swap( TrajectoryStateOnSurface & rh,  TrajectoryStateOnSurface & lh) {
  // use base swap
  rh.swap(lh);
}

std::ostream& operator<<(std::ostream& os, const TrajectoryStateOnSurface & tsos);
#endif
