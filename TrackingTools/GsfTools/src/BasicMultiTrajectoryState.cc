#include "TrackingTools/GsfTools/interface/BasicMultiTrajectoryState.h"

#include "DataFormats/GeometrySurface/interface/BoundPlane.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace SurfaceSideDefinition;

BasicMultiTrajectoryState::BasicMultiTrajectoryState(const std::vector<TSOS>& tsvec)
    : BasicTrajectoryState(tsvec.front().surface()), theStates(tsvec) {
  // only accept planes!!
  const BoundPlane* bp = dynamic_cast<const BoundPlane*>(&tsvec.begin()->surface());
  if UNLIKELY (bp == nullptr)
    throw cms::Exception("LogicError") << "MultiTrajectoryState constructed on cylinder";

  for (auto i = tsvec.begin(); i != tsvec.end(); i++) {
    if UNLIKELY (!i->isValid()) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState constructed with invalid state";
    }
    if UNLIKELY (i->hasError() != tsvec.front().hasError()) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState mixes states with and without errors";
    }
    if UNLIKELY (&i->surface() != &tsvec.front().surface()) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState mixes states with different surfaces";
    }
    if UNLIKELY (i->surfaceSide() != tsvec.front().surfaceSide()) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState mixes states defined before and after material";
    }
    if UNLIKELY (i->localParameters().pzSign() * tsvec.front().localParameters().pzSign() < 0.) {
      throw cms::Exception("LogicError") << "MultiTrajectoryState mixes states with different signs of local p_z";
    }
  }
  //
  combine();
}

void BasicMultiTrajectoryState::rescaleError(double factor) {
  if UNLIKELY (theStates.empty()) {
    edm::LogError("BasicMultiTrajectoryState") << "Trying to rescale errors of empty MultiTrajectoryState!";
    return;
  }

  for (auto& is : theStates)
    is.rescaleError(factor);
  combine();
}

void BasicMultiTrajectoryState::combine() {
  const std::vector<TrajectoryStateOnSurface>& tsos = theStates;

  if UNLIKELY (tsos.empty()) {
    edm::LogError("MultiTrajectoryStateCombiner") << "Trying to collapse empty set of trajectory states!";
    return;
  }

  double pzSign = tsos.front().localParameters().pzSign();
  for (std::vector<TrajectoryStateOnSurface>::const_iterator it = tsos.begin(); it != tsos.end(); it++) {
    if UNLIKELY (it->localParameters().pzSign() != pzSign) {
      edm::LogError("MultiTrajectoryStateCombiner")
          << "Trying to collapse trajectory states with different signs on p_z!";
      return;
    }
  }

  if UNLIKELY (tsos.size() == 1) {
    BasicTrajectoryState::update(tsos.front().weight(),
                                 tsos.front().localParameters(),
                                 tsos.front().localError(),
                                 tsos.front().surface(),
                                 tsos.front().magneticField(),
                                 tsos.front().surfaceSide());
    return;
  }

  double sumw = 0.;
  //int dim = tsos.front().localParameters().vector().num_row();
  AlgebraicVector5 mean;
  AlgebraicSymMatrix55 covarPart1, covarPart2, covtmp;
  for (auto it1 = tsos.begin(); it1 != tsos.end(); it1++) {
    auto weight = it1->weight();
    auto const& param = it1->localParameters().vector();
    sumw += weight;
    mean += weight * param;
    covarPart1 += weight * it1->localError().matrix();
    for (auto it2 = it1 + 1; it2 != tsos.end(); it2++) {
      AlgebraicVector5 diff = param - it2->localParameters().vector();
      ROOT::Math::AssignSym::Evaluate(covtmp, ROOT::Math::TensorProd(diff, diff));
      covarPart2 += (weight * it2->weight()) * covtmp;
    }
  }
  double sumwI = 1.0 / sumw;
  mean *= sumwI;
  covarPart1 *= sumwI;
  covarPart2 *= (sumwI * sumwI);
  AlgebraicSymMatrix55 covar = covarPart1 + covarPart2;

  BasicTrajectoryState::update(sumw,
                               LocalTrajectoryParameters(mean, pzSign),
                               LocalTrajectoryError(covar),
                               tsos.front().surface(),
                               tsos.front().magneticField(),
                               tsos.front().surfaceSide());
}

void BasicMultiTrajectoryState::update(const LocalTrajectoryParameters& p,
                                       const Surface& aSurface,
                                       const MagneticField* field,
                                       const SurfaceSide side) {
  throw cms::Exception("LogicError",
                       "BasicMultiTrajectoryState::update(LocalTrajectoryParameters, Surface, ...) called even if "
                       "canUpdateLocalParameters() is false");
}

void BasicMultiTrajectoryState::update(double weight,
                                       const LocalTrajectoryParameters& p,
                                       const LocalTrajectoryError& err,
                                       const Surface& aSurface,
                                       const MagneticField* field,
                                       const SurfaceSide side) {
  throw cms::Exception("LogicError",
                       "BasicMultiTrajectoryState::update(LocalTrajectoryParameters, LocalTrajectoryError, ...) called "
                       "even if canUpdateLocalParameters() is false");
}
