#include "RecoVertex/VertexPrimitives/interface/BasicSingleVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"
#include <limits>

namespace {
  constexpr double dNaN = std::numeric_limits<double>::quiet_NaN();
}

BasicSingleVertexState::BasicSingleVertexState()
  : thePos(GlobalPoint(0, 0, 0)), thePosAvailable(false),
    theTime(dNaN), theTimeAvailable(false),
    theErr(AlgebraicSymMatrix44()), theErrAvailable(false),
    theWeight(AlgebraicSymMatrix44()), theWeightAvailable(false),
    theWeightTimesPos(AlgebraicVector4()), theWeightTimesPosAvailable(false),
    valid(false), vertexIs4D(false), theWeightInMix(0.)
{}

BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
			     const GlobalError & posErr,
			     const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theTime(dNaN), theTimeAvailable(false),
    theErr(posErr), theErrAvailable(true),
    theWeight(AlgebraicSymMatrix44()), theWeightAvailable(false),
    theWeightTimesPos(AlgebraicVector4()), theWeightTimesPosAvailable(false),
    valid(true), vertexIs4D(false), theWeightInMix(weightInMix)
{}


BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
			     const GlobalWeight & posWeight,
			     const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theTime(dNaN), theTimeAvailable(false),
    theErr(AlgebraicSymMatrix44()), theErrAvailable(false),
    theWeight(posWeight), theWeightAvailable(true),
    theWeightTimesPos(AlgebraicVector4()), theWeightTimesPosAvailable(false),
    valid(true), vertexIs4D(false), theWeightInMix(weightInMix)
{}


BasicSingleVertexState::BasicSingleVertexState(const AlgebraicVector3 & weightTimesPosition,
			     const GlobalWeight & posWeight,
			     const double & weightInMix)
  : thePos(GlobalPoint(0, 0, 0)), thePosAvailable(false),
    theTime(dNaN), theTimeAvailable(false),
    theErr(AlgebraicSymMatrix44()), theErrAvailable(false),
    theWeight(GlobalWeight(posWeight)), theWeightAvailable(true),
    theWeightTimesPos(AlgebraicVector4(weightTimesPosition[0],
                                       weightTimesPosition[1],
                                       weightTimesPosition[2],0.0)), 
    theWeightTimesPosAvailable(true),
    valid(true), vertexIs4D(false), theWeightInMix(weightInMix)
{}

// no-offdiags for time
BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
                                               const GlobalError & posErr,
                                               const double time,
                                               const double timeErr,
                                               const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theTime(time), theTimeAvailable(true),
    theErr(posErr), theErrAvailable(true),
    theWeight(AlgebraicSymMatrix44()), theWeightAvailable(false),
    theWeightTimesPos(AlgebraicVector4()), theWeightTimesPosAvailable(false),
    valid(true), vertexIs4D(true), theWeightInMix(weightInMix)
{
  // You dumb bastard. It's not a schooner, its a sailboat.
  GlobalError timeErrMat(0.,
                         0.,0.,
                         0.,0.,0.,
                         0.,0.,0.,timeErr*timeErr);
  theErr = theErr + timeErrMat;
}


BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
                                               const GlobalWeight & posWeight,
                                               const double time,
                                               const double timeWeight,
                                               const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theTime(time), theTimeAvailable(true),
    theErr(AlgebraicSymMatrix44()), theErrAvailable(false),
    theWeight(posWeight), theWeightAvailable(true),
    theWeightTimesPos(AlgebraicVector4()), theWeightTimesPosAvailable(false),
    valid(true), vertexIs4D(true), theWeightInMix(weightInMix)
{
  GlobalWeight timeWeightMat(0.,
                             0.,0.,
                             0.,0.,0.,
                             0.,0.,0.,timeWeight);
  theWeight = theWeight + timeWeightMat;
}


BasicSingleVertexState::BasicSingleVertexState(const AlgebraicVector3 & weightTimesPosition,
                                               const GlobalWeight & posWeight,
                                               const double weightTimesTime,
                                               const double timeWeight,
                                               const double & weightInMix)
  : thePos(GlobalPoint(0, 0, 0)), thePosAvailable(false),
    theTime(dNaN), theTimeAvailable(false),
    theErr(AlgebraicSymMatrix44()), theErrAvailable(false),
    theWeight(posWeight), theWeightAvailable(true),
    theWeightTimesPos(AlgebraicVector4(weightTimesPosition[0],weightTimesPosition[1],weightTimesPosition[2],weightTimesTime)), 
    theWeightTimesPosAvailable(true),
    valid(true), vertexIs4D(true), theWeightInMix(weightInMix)
{
  GlobalWeight timeWeightMat(0.,
                             0.,0.,
                             0.,0.,0.,
                             0.,0.,0.,timeWeight);
  theWeight = theWeight + timeWeightMat;
}

// off-diags for time
BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
                                               const double time,
                                               const GlobalError & posTimeErr, // fully filled 4x4 matrix
                                               const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theTime(time), theTimeAvailable(true),
    theErr(posTimeErr), theErrAvailable(true),
    theWeight(AlgebraicSymMatrix44()), theWeightAvailable(false),
    theWeightTimesPos(AlgebraicVector4()), theWeightTimesPosAvailable(false),
    valid(true), vertexIs4D(true), theWeightInMix(weightInMix)
{}


BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
                                               const double time,
                                               const GlobalWeight & posTimeWeight,
                                               const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theTime(time), theTimeAvailable(true),
    theErr(AlgebraicSymMatrix44()), theErrAvailable(false),
    theWeight(posTimeWeight), theWeightAvailable(true),
    theWeightTimesPos(AlgebraicVector4()), theWeightTimesPosAvailable(false),
    valid(true), vertexIs4D(true), theWeightInMix(weightInMix)
{}


BasicSingleVertexState::BasicSingleVertexState(const AlgebraicVector4 & weightTimesPosition,
                                               const GlobalWeight & posWeight,
                                               const double & weightInMix)
  : thePos(GlobalPoint(0, 0, 0)), thePosAvailable(false),
    theTime(dNaN), theTimeAvailable(false),
    theErr(AlgebraicSymMatrix44()), theErrAvailable(false),
    theWeight(posWeight), theWeightAvailable(true),
    theWeightTimesPos(weightTimesPosition), theWeightTimesPosAvailable(true),
    valid(true), vertexIs4D(true), theWeightInMix(weightInMix)
{}

GlobalPoint BasicSingleVertexState::position() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::position::invalid");
  if (!thePosAvailable) computePosition();
  return thePos;
}

GlobalError BasicSingleVertexState::error() const
{
  if (!valid) {
    float *temp = nullptr;
    std::cout << *temp;
    throw VertexException("BasicSingleVertexState::error::invalid");
  }
  if (!theErrAvailable) computeError();
  return GlobalError(theErr.matrix());
}

GlobalError BasicSingleVertexState::error4D() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::error4D::invalid");
  if (!theErrAvailable) computeError();
  return theErr;
}

double BasicSingleVertexState::time() const {
  if ( vertexIs4D ) {
    if (!valid) throw VertexException("BasicSingleVertexState::time::invalid");
    if (!theTimeAvailable) computePosition(); // time computed with position (4-vector)
    return theTime;
  }
  return dNaN;
}

double BasicSingleVertexState::timeError() const {
  if( vertexIs4D ) {
    if (!valid) throw VertexException("BasicSingleVertexState::timeError::invalid");
    if (!theTimeAvailable) computeError();
    return std::sqrt(theErr.matrix4D()(3,3));
  }
  return dNaN;
}

GlobalWeight BasicSingleVertexState::weight() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::weight::invalid");
  if (!theWeightAvailable) computeWeight();
  return GlobalWeight(theWeight.matrix());
}

GlobalWeight BasicSingleVertexState::weight4D() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::weight4D::invalid");
  if (!theWeightAvailable) computeWeight();
  return theWeight;
}

AlgebraicVector3 BasicSingleVertexState::weightTimesPosition() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::weightTimesPosition::invalid");
  if (!theWeightTimesPosAvailable) computeWeightTimesPos();
  return AlgebraicVector3(theWeightTimesPos[0],theWeightTimesPos[1],theWeightTimesPos[2]);
}

AlgebraicVector4 BasicSingleVertexState::weightTimesPosition4D() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::weightTimesPosition4D::invalid");
  if (!theWeightTimesPosAvailable) computeWeightTimesPos();
  return theWeightTimesPos;  
}


double BasicSingleVertexState::weightInMixture() const 
{
  if (!valid) throw VertexException("BasicSingleVertexState::weightInMixture::invalid");
  return theWeightInMix;
}
// RefCountedVertexSeed BasicSingleVertexState::seedWithoutTracks() const
// {
//   RefCountedVertexSeed v = new VertexSeed(position(), error());
//   return v;
// }



void BasicSingleVertexState::computePosition() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::computePosition::invalid");
  AlgebraicVector4 pos = error4D().matrix_new4D()*weightTimesPosition4D();
  thePos = GlobalPoint(pos[0], pos[1], pos[2]);
  theTime = pos[3];
  thePosAvailable  = true;
  theTimeAvailable = true;
}

void BasicSingleVertexState::computeError() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::computeError::invalid");
  int ifail;
  if( vertexIs4D ) {
    theErr = weight4D().matrix4D().Inverse(ifail);
    if (ifail != 0) throw VertexException("BasicSingleVertexState::could not invert weight matrix");
  } else {
    theErr = weight4D().matrix().Inverse(ifail);
    if (ifail != 0) throw VertexException("BasicSingleVertexState::could not invert weight matrix");
  }
  theErrAvailable = true;
}


void BasicSingleVertexState::computeWeight() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::computeWeight::invalid");
  int ifail;
  if( vertexIs4D ) {
    theWeight = error4D().matrix4D().Inverse(ifail);    
    if (ifail != 0) throw VertexException("BasicSingleVertexState::could not invert error matrix");
  } else {
    theWeight = error4D().matrix().Inverse(ifail);
    if (ifail != 0) throw VertexException("BasicSingleVertexState::could not invert error matrix");
  }  
  theWeightAvailable = true;
}


void BasicSingleVertexState::computeWeightTimesPos() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::computeWeightTimesPos::invalid");
  AlgebraicVector4 pos; pos(0) = position().x();
  pos(1) = position().y(); pos(2) = position().z();
  if ( vertexIs4D ) {
    pos(3) = theTime;
  } else {
    pos(3) = 0.;
  }
  theWeightTimesPos = weight4D().matrix_new4D()*pos;
  theWeightTimesPosAvailable = true;
}


