#include "TrackingTools/TrajectoryState/interface/BasicTrajectoryState.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <cmath>
#include <sstream>

#ifdef DO_BTSCount
unsigned int BTSCount::maxReferences = 0;
unsigned long long BTSCount::aveReferences = 0;
unsigned long long BTSCount::toteReferences = 0;

BTSCount::~BTSCount() {
  maxReferences = std::max(referenceMax_, maxReferences);
  toteReferences++;
  aveReferences += referenceMax_;
  // if (referenceMax_>100) std::cout <<"BST with " << referenceMax_ << std::endl;
}

#include <iostream>
namespace {

  struct Printer {
    ~Printer() {
      std::cout << "maxReferences of BTSCount = " << BTSCount::maxReferences << " "
                << double(BTSCount::aveReferences) / double(BTSCount::toteReferences) << std::endl;
    }
  };
  Printer printer;

}  // namespace
#endif

BasicTrajectoryState::~BasicTrajectoryState() {}
BasicTrajectoryState::BasicTrajectoryState(const SurfaceType& aSurface)
    : theLocalError(InvalidError()),
      theLocalParameters(),
      theLocalParametersValid(false),
      theValid(false),
      theSurfaceSide(SurfaceSideDefinition::atCenterOfSurface),
      theSurfaceP(&aSurface),
      theWeight(1.) {}

namespace {
  inline FreeTrajectoryState makeFTS(const LocalTrajectoryParameters& par,
                                     const BasicTrajectoryState::SurfaceType& surface,
                                     const MagneticField* field) {
    GlobalPoint x = surface.toGlobal(par.position());
    GlobalVector p = surface.toGlobal(par.momentum());
    return FreeTrajectoryState(x, p, par.charge(), field);
  }

  inline FreeTrajectoryState makeFTS(const LocalTrajectoryParameters& par,
                                     const BasicTrajectoryState::SurfaceType& surface,
                                     const MagneticField* field,
                                     GlobalVector fieldValue) {
    GlobalPoint x = surface.toGlobal(par.position());
    GlobalVector p = surface.toGlobal(par.momentum());
    return FreeTrajectoryState(x, p, par.charge(), field, fieldValue);
  }

}  // namespace

BasicTrajectoryState::BasicTrajectoryState(const LocalTrajectoryParameters& par,
                                           const LocalTrajectoryError& err,
                                           const SurfaceType& aSurface,
                                           const MagneticField* field,
                                           const SurfaceSide side)
    : theFreeState(makeFTS(par, aSurface, field)),
      theLocalError(err),
      theLocalParameters(par),
      theLocalParametersValid(true),
      theValid(true),
      theSurfaceSide(side),
      theSurfaceP(&aSurface),
      theWeight(1.) {}

void BasicTrajectoryState::notValid() {
  throw TrajectoryStateException("TrajectoryStateOnSurface is invalid and cannot return any parameters");
}

namespace {
  void verifyLocalErr(LocalTrajectoryError const& err, const FreeTrajectoryState& state) {
    if UNLIKELY (!err.posDef())
      edm::LogWarning("BasicTrajectoryState") << "local error not pos-def\n"
                                              << err.matrix() << "\npos/mom/mf " << state.position() << ' '
                                              << state.momentum() << ' ' << state.parameters().magneticFieldInTesla();
  }
  void verifyCurvErr(CurvilinearTrajectoryError const& err, const FreeTrajectoryState& state) {
    if UNLIKELY (!err.posDef())
      edm::LogWarning("BasicTrajectoryState") << "curv error not pos-def\n"
                                              << err.matrix() << "\npos/mom/mf " << state.position() << ' '
                                              << state.momentum() << ' ' << state.parameters().magneticFieldInTesla();
  }
}  // namespace

void BasicTrajectoryState::missingError(char const* where) const {
  std::stringstream form;
  form << "BasicTrajectoryState: attempt to access errors when none available " << where
       << ".\nfreestate pointer: " << theFreeState << "\nlocal error valid/values :" << theLocalError.valid() << "\n"
       << theLocalError.matrix();

  edm::LogWarning("BasicTrajectoryState") << form.str();

  // throw TrajectoryStateException(form.str());
}

void BasicTrajectoryState::checkCurvilinError() const {
  if LIKELY (theFreeState.hasCurvilinearError())
    return;

  if UNLIKELY (!theLocalParametersValid)
    createLocalParameters();

  JacobianLocalToCurvilinear loc2Curv(surface(), localParameters(), globalParameters(), *magneticField());
  const AlgebraicMatrix55& jac = loc2Curv.jacobian();
  const AlgebraicSymMatrix55& cov = ROOT::Math::Similarity(jac, theLocalError.matrix());

  theFreeState.setCurvilinearError(cov);

  verifyLocalErr(theLocalError, theFreeState);
  verifyCurvErr(cov, theFreeState);
}

// create local parameters from global
void BasicTrajectoryState::createLocalParameters() const {
  LocalPoint x = surface().toLocal(theFreeState.position());
  LocalVector p = surface().toLocal(theFreeState.momentum());
  // believe p.z() never exactly equals 0.
  bool isCharged = theFreeState.charge() != 0;
  theLocalParameters = LocalTrajectoryParameters(isCharged ? theFreeState.signedInverseMomentum() : 1. / p.mag(),
                                                 p.x() / p.z(),
                                                 p.y() / p.z(),
                                                 x.x(),
                                                 x.y(),
                                                 p.z() > 0. ? 1. : -1.,
                                                 isCharged);
  theLocalParametersValid = true;
}

void BasicTrajectoryState::createLocalError() const {
  if LIKELY (theFreeState.hasCurvilinearError())
    createLocalErrorFromCurvilinearError();
  else
    theLocalError = InvalidError();
}

void BasicTrajectoryState::createLocalErrorFromCurvilinearError() const {
  JacobianCurvilinearToLocal curv2Loc(surface(), localParameters(), globalParameters(), *magneticField());
  const AlgebraicMatrix55& jac = curv2Loc.jacobian();

  theLocalError = ROOT::Math::Similarity(jac, theFreeState.curvilinearError().matrix());

  verifyCurvErr(theFreeState.curvilinearError(), theFreeState);
  verifyLocalErr(theLocalError, theFreeState);
}

// update in place and in the	very same place
void BasicTrajectoryState::update(const LocalTrajectoryParameters& p, const SurfaceSide side) {
  theLocalParameters = p;
  theSurfaceSide = side;
  theLocalError = InvalidError();
  theFreeState = makeFTS(p, surface(), magneticField(), theFreeState.parameters().magneticFieldInTesla());

  theValid = true;
  theLocalParametersValid = true;
}

void BasicTrajectoryState::update(const LocalTrajectoryParameters& p,
                                  const SurfaceType& aSurface,
                                  const MagneticField* field,
                                  const SurfaceSide side) {
  theLocalParameters = p;
  if (&aSurface != &*theSurfaceP)
    theSurfaceP.reset(&aSurface);
  theSurfaceSide = side;
  theWeight = 1.0;
  theLocalError = InvalidError();
  theFreeState = makeFTS(p, aSurface, field);

  theValid = true;
  theLocalParametersValid = true;
}

void BasicTrajectoryState::update(double weight,
                                  const LocalTrajectoryParameters& p,
                                  const LocalTrajectoryError& err,
                                  const SurfaceType& aSurface,
                                  const MagneticField* field,
                                  const SurfaceSide side) {
  theLocalParameters = p;
  theLocalError = err;
  if (&aSurface != &*theSurfaceP)
    theSurfaceP.reset(&aSurface);
  theSurfaceSide = side;
  theWeight = weight;
  theFreeState = makeFTS(p, aSurface, field);

  theValid = true;
  theLocalParametersValid = true;
}

void BasicTrajectoryState::update(const LocalTrajectoryParameters& p,
                                  const LocalTrajectoryError& err,
                                  const SurfaceSide side) {
  theLocalParameters = p;
  theLocalError = err;
  theSurfaceSide = side;
  theFreeState = theFreeState =
      makeFTS(p, surface(), magneticField(), theFreeState.parameters().magneticFieldInTesla());

  theValid = true;
  theLocalParametersValid = true;
}

void BasicTrajectoryState::rescaleError(double factor) {
  if UNLIKELY (!hasError())
    missingError(" trying to rescale");
  theFreeState.rescaleError(factor);

  if (theLocalError.valid()) {
    //do it by hand if the free state is not around.
    bool zeroField = (magneticField()->nominalValue() == 0);
    if UNLIKELY (zeroField) {
      AlgebraicSymMatrix55 errors = theLocalError.matrix();
      //scale the 0 indexed covariance by the square root of the factor
      for (unsigned int i = 1; i != 5; ++i)
        errors(i, 0) *= factor;
      double factor_squared = factor * factor;
      //scale all others by the scaled factor
      for (unsigned int i = 1; i != 5; ++i)
        for (unsigned int j = i; j != 5; ++j)
          errors(i, j) *= factor_squared;
      //term 0,0 is not scaled at all
      theLocalError = LocalTrajectoryError(errors);
    } else
      theLocalError *= (factor * factor);
  }
}

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
BasicSingleTrajectoryState::Components const& BasicSingleTrajectoryState::components() const {
  edm::LogError("BasicSingleTrajectoryState") << "asking for componenets to a SingleTrajectoryState" << std::endl;
  assert(false);
}
