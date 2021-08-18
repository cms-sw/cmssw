#ifndef BasicTrajectoryState_H
#define BasicTrajectoryState_H

#include "TrackingTools/TrajectoryState/interface/ProxyBase11.h"

#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/CartesianTrajectoryError.h"
#include "TrackingTools/TrajectoryParametrization/interface/CurvilinearTrajectoryError.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

#include <vector>

#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryError.h"

#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/DeepCopyPointer.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/TrajectoryStateExceptions.h"
#include "FWCore/Utilities/interface/Likely.h"

/// vvv DEBUG
// #include <iostream>

class MagneticField;
class TrajectoryStateOnSurface;

#ifdef DO_BTSCount
class BTSCount {
public:
  BTSCount() {}
  virtual ~BTSCount();
  BTSCount(BTSCount const&) {}

  static unsigned int maxReferences;
  static unsigned long long aveReferences;
  static unsigned long long toteReferences;

  void addReference() const {
    ++referenceCount_;
    referenceMax_ = std::max(referenceMax_, referenceCount_);
  }
  void removeReference() const {
    if (0 == --referenceCount_) {
      delete const_cast<BTSCount*>(this);
    }
  }

  unsigned int references() const { return referenceCount_; }

private:
  mutable unsigned int referenceCount_ = 0;
  mutable unsigned int referenceMax_ = 0;
};
#endif

/** No so Abstract (anyore) base class for TrajectoryState.
 *  It is ReferenceCounted.
 *
 * VI 8/12/2011   content of BasicSingleTrajectoryState moved here....
 * fully devirtualized
 */
class BasicTrajectoryState {
public:
  typedef BasicTrajectoryState BTSOS;
  typedef ProxyBase11<BTSOS> Proxy;
  typedef Proxy::pointer pointer;
  typedef SurfaceSideDefinition::SurfaceSide SurfaceSide;
  typedef Surface SurfaceType;

public:
  // default constructor : to make root happy
  BasicTrajectoryState() : theValid(false), theWeight(0) {}

  /// construct invalid trajectory state (without parameters)
  explicit BasicTrajectoryState(const SurfaceType& aSurface);

  virtual ~BasicTrajectoryState();

  virtual pointer clone() const = 0;

  template <typename T, typename... Args>
  static std::shared_ptr<BTSOS> build(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  static std::shared_ptr<BTSOS> churn(Args&&... args) {
    return std::allocate_shared<T>(churn_allocator<T>(), std::forward<Args>(args)...);
  }

  /** Constructor from FTS and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState(const FreeTrajectoryState& fts,
                       const SurfaceType& aSurface,
                       const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface)
      : theFreeState(fts),
        theLocalError(InvalidError()),
        theLocalParameters(),
        theLocalParametersValid(false),
        theValid(true),
        theSurfaceSide(side),
        theSurfaceP(&aSurface),
        theWeight(1.) {}

  /** Constructor from FTS: just a wrapper
   */
  explicit BasicTrajectoryState(const FreeTrajectoryState& fts)
      : theFreeState(fts),
        theLocalError(InvalidError()),
        theLocalParameters(),
        theLocalParametersValid(false),
        theValid(true),
        theWeight(1.) {}

  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicTrajectoryState(const LocalTrajectoryParameters& par,
                       const LocalTrajectoryError& err,
                       const SurfaceType& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface);

  /** Constructor from local parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState(const LocalTrajectoryParameters& par,
                       const SurfaceType& aSurface,
                       const MagneticField* field,
                       const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface)
      : BasicTrajectoryState(par, InvalidError(), aSurface, field, side) {}

  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState(const GlobalTrajectoryParameters& par,
                       const CartesianTrajectoryError& err,
                       const SurfaceType& aSurface,
                       const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface)
      : theFreeState(par, err),
        theLocalError(InvalidError()),
        theLocalParameters(),
        theLocalParametersValid(false),
        theValid(true),
        theSurfaceSide(side),
        theSurfaceP(&aSurface),
        theWeight(1.) {}

  /** Constructor from global parameters, errors and surface. For surfaces 
   *  with material the side of the surface should be specified explicitely. 
   *  For multi-states the weight should be specified explicitely.
   */
  BasicTrajectoryState(const GlobalTrajectoryParameters& par,
                       const CurvilinearTrajectoryError& err,
                       const SurfaceType& aSurface,
                       const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface)
      : theFreeState(par, err),
        theLocalError(InvalidError()),
        theLocalParameters(),
        theLocalParametersValid(false),
        theValid(true),
        theSurfaceSide(side),
        theSurfaceP(&aSurface),
        theWeight(1.) {}

  /** Constructor from global parameters and surface. For surfaces with material
   *  the side of the surface should be specified explicitely.
   */
  BasicTrajectoryState(const GlobalTrajectoryParameters& par,
                       const SurfaceType& aSurface,
                       const SurfaceSide side = SurfaceSideDefinition::atCenterOfSurface)
      : BasicTrajectoryState(par, InvalidError(), aSurface, side) {}

  // as above, with explicit weight
  template <typename... Args>
  BasicTrajectoryState(double iweight, Args&&... args) : BasicTrajectoryState(std::forward<Args>(args)...) {
    theWeight = iweight;
  }

  bool isValid() const { return theValid; }

  // access global parameters/errors
  const GlobalTrajectoryParameters& globalParameters() const { return theFreeState.parameters(); }
  GlobalPoint globalPosition() const { return theFreeState.position(); }
  GlobalVector globalMomentum() const { return theFreeState.momentum(); }
  GlobalVector globalDirection() const { return theFreeState.momentum().unit(); }
  TrackCharge charge() const { return theFreeState.charge(); }
  double signedInverseMomentum() const { return theFreeState.signedInverseMomentum(); }
  double transverseCurvature() const { return theFreeState.transverseCurvature(); }

  const CartesianTrajectoryError cartesianError() const {
    if UNLIKELY (!hasError()) {
      missingError(" accesing cartesian error.");
      return CartesianTrajectoryError();
    }
    return freeTrajectoryState(true)->cartesianError();
  }
  const CurvilinearTrajectoryError& curvilinearError() const {
    if UNLIKELY (!hasError()) {
      missingError(" accesing curvilinearerror.");
      static const CurvilinearTrajectoryError crap;
      return crap;
    }
    return freeTrajectoryState(true)->curvilinearError();
  }

  FreeTrajectoryState const* freeTrajectoryState(bool withErrors = true) const {
    if UNLIKELY (!isValid())
      notValid();
    if (withErrors && hasError()) {  // this is the right thing
      checkCurvilinError();
    }
    return &theFreeState;
  }

  const MagneticField* magneticField() const { return &theFreeState.parameters().magneticField(); }

  // access local parameters/errors
  const LocalTrajectoryParameters& localParameters() const {
    if UNLIKELY (!isValid())
      notValid();
    if UNLIKELY (!theLocalParametersValid)
      createLocalParameters();
    return theLocalParameters;
  }
  LocalPoint localPosition() const { return localParameters().position(); }
  LocalVector localMomentum() const { return localParameters().momentum(); }
  LocalVector localDirection() const { return localMomentum().unit(); }

  const LocalTrajectoryError& localError() const {
    if UNLIKELY (!hasError()) {
      missingError(" accessing local error.");
      return theLocalError;
    }
    if UNLIKELY (theLocalError.invalid())
      createLocalError();
    return theLocalError;
  }

  const SurfaceType& surface() const { return *theSurfaceP; }

  double weight() const { return theWeight; }

  void rescaleError(double factor);

  /// Position relative to material, defined relative to momentum vector.
  SurfaceSide surfaceSide() const { return theSurfaceSide; }

  bool hasError() const { return theFreeState.hasError() || theLocalError.valid(); }

  virtual bool canUpdateLocalParameters() const { return true; }

  virtual void update(const LocalTrajectoryParameters& p,
                      const SurfaceType& aSurface,
                      const MagneticField* field,
                      const SurfaceSide side);

  // update in place and in the very same place
  virtual void update(const LocalTrajectoryParameters& p, const SurfaceSide side) final;

  virtual void update(double weight,
                      const LocalTrajectoryParameters& p,
                      const LocalTrajectoryError& err,
                      const SurfaceType& aSurface,
                      const MagneticField* field,
                      const SurfaceSide side);

  // update in place and in the very same place
  virtual void update(const LocalTrajectoryParameters& p,
                      const LocalTrajectoryError& err,
                      const SurfaceSide side) final;

  CurvilinearTrajectoryError& setCurvilinearError() { return theFreeState.setCurvilinearError(); }

public:
  using Components = std::vector<TrajectoryStateOnSurface>;
  virtual Components const& components() const = 0;
  virtual bool singleState() const = 0;

private:
  static void notValid();

  void missingError(char const* where) const;  // dso_internal;

  // create global errors from local
  void checkCurvilinError() const;  //  dso_internal;

  // create local parameters and errors from global
  void createLocalParameters() const;
  // create local errors from global
  void createLocalError() const;
  void createLocalErrorFromCurvilinearError() const dso_internal;

private:
  mutable FreeTrajectoryState theFreeState;

  mutable LocalTrajectoryError theLocalError;
  mutable LocalTrajectoryParameters theLocalParameters;

  mutable bool theLocalParametersValid;
  mutable bool theValid;

  SurfaceSide theSurfaceSide;
  ConstReferenceCountingPointer<SurfaceType> theSurfaceP;

  double theWeight = 0.;
};

#endif
