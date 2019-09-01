#include "RecoVertex/VertexPrimitives/interface/BasicSingleVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include <limits>

namespace {
  constexpr double dNaN = std::numeric_limits<double>::quiet_NaN();
}

BasicSingleVertexState::BasicSingleVertexState()
    : thePos(GlobalPoint(0, 0, 0)),
      theTime(dNaN),
      theErr(AlgebraicSymMatrix44()),
      theWeight(AlgebraicSymMatrix44()),
      theWeightTimesPos(AlgebraicVector4()),
      theWeightInMix(0.),
      thePosAvailable(false),
      theTimeAvailable(false),
      theErrAvailable(false),
      theWeightAvailable(false),
      theWeightTimesPosAvailable(false),
      valid(false),
      vertexIs4D(false) {}

BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint& pos,
                                               const GlobalError& posErr,
                                               const double& weightInMix)
    : thePos(pos),
      theTime(dNaN),
      theErr(posErr),
      theWeight(AlgebraicSymMatrix44()),
      theWeightTimesPos(AlgebraicVector4()),
      theWeightInMix(weightInMix),
      thePosAvailable(true),
      theTimeAvailable(false),
      theErrAvailable(true),
      theWeightAvailable(false),
      theWeightTimesPosAvailable(false),
      valid(true),
      vertexIs4D(false) {}

BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint& pos,
                                               const GlobalWeight& posWeight,
                                               const double& weightInMix)
    : thePos(pos),
      theTime(dNaN),
      theErr(AlgebraicSymMatrix44()),
      theWeight(posWeight),
      theWeightTimesPos(AlgebraicVector4()),
      theWeightInMix(weightInMix),
      thePosAvailable(true),
      theTimeAvailable(false),
      theErrAvailable(false),
      theWeightAvailable(true),
      theWeightTimesPosAvailable(false),
      valid(true),
      vertexIs4D(false) {}

BasicSingleVertexState::BasicSingleVertexState(const AlgebraicVector3& weightTimesPosition,
                                               const GlobalWeight& posWeight,
                                               const double& weightInMix)
    : thePos(GlobalPoint(0, 0, 0)),
      theTime(dNaN),
      theErr(AlgebraicSymMatrix44()),
      theWeight(posWeight),
      theWeightTimesPos(weightTimesPosition[0], weightTimesPosition[1], weightTimesPosition[2], 0),
      theWeightInMix(weightInMix),
      thePosAvailable(false),
      theTimeAvailable(false),
      theErrAvailable(false),
      theWeightAvailable(true),
      theWeightTimesPosAvailable(true),
      valid(true),
      vertexIs4D(false) {}

// no-offdiags for time
BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint& pos,
                                               const GlobalError& posErr,
                                               const double time,
                                               const double timeErr,
                                               const double& weightInMix)
    : thePos(pos),
      theTime(time),
      theErr(posErr),
      theWeight(AlgebraicSymMatrix44()),
      theWeightTimesPos(AlgebraicVector4()),
      theWeightInMix(weightInMix),
      thePosAvailable(true),
      theTimeAvailable(true),
      theErrAvailable(true),
      theWeightAvailable(false),
      theWeightTimesPosAvailable(false),
      valid(true),
      vertexIs4D(true) {
  // You dumb bastard. It's not a schooner, its a sailboat.
  GlobalError timeErrMat(0., 0., 0., 0., 0., 0., 0., 0., 0., timeErr * timeErr);
  theErr = theErr + timeErrMat;
}

BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint& pos,
                                               const GlobalWeight& posWeight,
                                               const double time,
                                               const double timeWeight,
                                               const double& weightInMix)
    : thePos(pos),
      theTime(time),
      theErr(AlgebraicSymMatrix44()),
      theWeight(posWeight),
      theWeightTimesPos(AlgebraicVector4()),
      theWeightInMix(weightInMix),
      thePosAvailable(true),
      theTimeAvailable(true),
      theErrAvailable(false),
      theWeightAvailable(true),
      theWeightTimesPosAvailable(false),
      valid(true),
      vertexIs4D(true) {
  GlobalWeight timeWeightMat(0., 0., 0., 0., 0., 0., 0., 0., 0., timeWeight);
  theWeight = theWeight + timeWeightMat;
}

BasicSingleVertexState::BasicSingleVertexState(const AlgebraicVector3& weightTimesPosition,
                                               const GlobalWeight& posWeight,
                                               const double weightTimesTime,
                                               const double timeWeight,
                                               const double& weightInMix)
    : thePos(GlobalPoint(0, 0, 0)),
      theTime(dNaN),
      theErr(AlgebraicSymMatrix44()),
      theWeight(posWeight),
      theWeightTimesPos(weightTimesPosition[0], weightTimesPosition[1], weightTimesPosition[2], weightTimesTime),
      theWeightInMix(weightInMix),
      thePosAvailable(false),
      theTimeAvailable(false),
      theErrAvailable(false),
      theWeightAvailable(true),
      theWeightTimesPosAvailable(true),
      valid(true),
      vertexIs4D(true) {
  GlobalWeight timeWeightMat(0., 0., 0., 0., 0., 0., 0., 0., 0., timeWeight);
  theWeight = theWeight + timeWeightMat;
}

// off-diags for time
BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint& pos,
                                               const double time,
                                               const GlobalError& posTimeErr,  // fully filled 4x4 matrix
                                               const double& weightInMix)
    : thePos(pos),
      theTime(time),
      theErr(posTimeErr),
      theWeight(AlgebraicSymMatrix44()),
      theWeightTimesPos(AlgebraicVector4()),
      theWeightInMix(weightInMix),
      thePosAvailable(true),
      theTimeAvailable(true),
      theErrAvailable(true),
      theWeightAvailable(false),
      theWeightTimesPosAvailable(false),
      valid(true),
      vertexIs4D(true) {}

BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint& pos,
                                               const double time,
                                               const GlobalWeight& posTimeWeight,
                                               const double& weightInMix)
    : thePos(pos),
      theTime(time),
      theErr(AlgebraicSymMatrix44()),
      theWeight(posTimeWeight),
      theWeightTimesPos(AlgebraicVector4()),
      theWeightInMix(weightInMix),
      thePosAvailable(true),
      theTimeAvailable(true),
      theErrAvailable(false),
      theWeightAvailable(true),
      theWeightTimesPosAvailable(false),
      valid(true),
      vertexIs4D(true) {}

BasicSingleVertexState::BasicSingleVertexState(const AlgebraicVector4& weightTimesPosition,
                                               const GlobalWeight& posWeight,
                                               const double& weightInMix)
    : thePos(GlobalPoint(0, 0, 0)),
      theTime(dNaN),
      theErr(AlgebraicSymMatrix44()),
      theWeight(posWeight),
      theWeightTimesPos(weightTimesPosition),
      theWeightInMix(weightInMix),
      thePosAvailable(false),
      theTimeAvailable(false),
      theErrAvailable(false),
      theWeightAvailable(true),
      theWeightTimesPosAvailable(true),
      valid(true),
      vertexIs4D(true) {}

GlobalPoint BasicSingleVertexState::position() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::position::invalid");
  if (!thePosAvailable)
    computePosition();
  return thePos;
}

GlobalError BasicSingleVertexState::error() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::error::invalid");
  if (!theErrAvailable)
    computeError();
  return GlobalError(theErr.matrix());
}

GlobalError BasicSingleVertexState::error4D() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::error4D::invalid");
  if (!theErrAvailable)
    computeError();
  return theErr;
}

double BasicSingleVertexState::time() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::time::invalid");
  if (!theTimeAvailable)
    computePosition();  // time computed with position (4-vector)
  return theTime;
}

double BasicSingleVertexState::timeError() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::timeError::invalid");
  if (!theTimeAvailable)
    computeError();
  return std::sqrt(theErr.matrix4D()(3, 3));
}

GlobalWeight BasicSingleVertexState::weight() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::weight::invalid");
  if (!theWeightAvailable)
    computeWeight();
  return GlobalWeight(theWeight.matrix());
}

GlobalWeight BasicSingleVertexState::weight4D() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::weight4D::invalid");
  if (!theWeightAvailable)
    computeWeight();
  return theWeight;
}

AlgebraicVector3 BasicSingleVertexState::weightTimesPosition() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::weightTimesPosition::invalid");
  if (!theWeightTimesPosAvailable)
    computeWeightTimesPos();
  return AlgebraicVector3(theWeightTimesPos[0], theWeightTimesPos[1], theWeightTimesPos[2]);
}

AlgebraicVector4 BasicSingleVertexState::weightTimesPosition4D() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::weightTimesPosition4D::invalid");
  if (!theWeightTimesPosAvailable)
    computeWeightTimesPos();
  return theWeightTimesPos;
}

double BasicSingleVertexState::weightInMixture() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::weightInMixture::invalid");
  return theWeightInMix;
}
// RefCountedVertexSeed BasicSingleVertexState::seedWithoutTracks() const
// {
//   RefCountedVertexSeed v = new VertexSeed(position(), error());
//   return v;
// }

void BasicSingleVertexState::computePosition() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::computePosition::invalid");
  AlgebraicVector4 pos = error4D().matrix4D() * weightTimesPosition4D();
  thePos = GlobalPoint(pos[0], pos[1], pos[2]);
  theTime = pos[3];
  thePosAvailable = true;
  theTimeAvailable = true;
}

void BasicSingleVertexState::computeError() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::computeError::invalid");
  int ifail;
  if (vertexIs4D) {
    theErr = weight4D().matrix4D().Inverse(ifail);
    if (ifail != 0)
      throw VertexException("BasicSingleVertexState::could not invert weight matrix");
  } else {
    theErr = weight4D().matrix().Inverse(ifail);
    if (ifail != 0)
      throw VertexException("BasicSingleVertexState::could not invert weight matrix");
  }
  theErrAvailable = true;
}

void BasicSingleVertexState::computeWeight() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::computeWeight::invalid");
  int ifail;
  if (vertexIs4D) {
    theWeight = error4D().matrix4D().Inverse(ifail);
    if (ifail != 0)
      throw VertexException("BasicSingleVertexState::could not invert error matrix");
  } else {
    theWeight = error4D().matrix().Inverse(ifail);
    if (ifail != 0)
      throw VertexException("BasicSingleVertexState::could not invert error matrix");
  }
  theWeightAvailable = true;
}

void BasicSingleVertexState::computeWeightTimesPos() const {
  if (!valid)
    throw VertexException("BasicSingleVertexState::computeWeightTimesPos::invalid");
  AlgebraicVector4 pos;
  pos(0) = position().x();
  pos(1) = position().y();
  pos(2) = position().z();
  if (vertexIs4D) {
    pos(3) = theTime;
  } else {
    pos(3) = 0.;
  }
  theWeightTimesPos = weight4D().matrix4D() * pos;
  theWeightTimesPosAvailable = true;
}
