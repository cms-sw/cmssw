#include "RecoVertex/VertexPrimitives/interface/BasicSingleVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexException.h"

BasicSingleVertexState::BasicSingleVertexState()
  : thePos(GlobalPoint(0, 0, 0)), thePosAvailable(false),
    theErr(AlgebraicSymMatrix33()), theErrAvailable(false),
    theWeight(AlgebraicSymMatrix33()), theWeightAvailable(false),
    theWeightTimesPos(AlgebraicVector3()), theWeightTimesPosAvailable(false),
    valid(false), theWeightInMix(0.)
{}


BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
			     const GlobalError & posErr,
			     const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theErr(posErr), theErrAvailable(true),
    theWeight(AlgebraicSymMatrix33()), theWeightAvailable(false),
    theWeightTimesPos(AlgebraicVector3()), theWeightTimesPosAvailable(false),
    valid(true), theWeightInMix(weightInMix)
{}


BasicSingleVertexState::BasicSingleVertexState(const GlobalPoint & pos,
			     const GlobalWeight & posWeight,
			     const double & weightInMix)
  : thePos(pos), thePosAvailable(true),
    theErr(AlgebraicSymMatrix33()), theErrAvailable(false),
    theWeight(posWeight), theWeightAvailable(true),
    theWeightTimesPos(AlgebraicVector3()), theWeightTimesPosAvailable(false),
    valid(true), theWeightInMix(weightInMix)
{}


BasicSingleVertexState::BasicSingleVertexState(const AlgebraicVector3 & weightTimesPosition,
			     const GlobalWeight & posWeight,
			     const double & weightInMix)
  : thePos(GlobalPoint(0, 0, 0)), thePosAvailable(false),
    theErr(AlgebraicSymMatrix33()), theErrAvailable(false),
    theWeight(posWeight), theWeightAvailable(true),
    theWeightTimesPos(weightTimesPosition), theWeightTimesPosAvailable(true),
    valid(true), theWeightInMix(weightInMix)
{//std::cout <<"BasicSingleVertexState ctor\n";
}

GlobalPoint BasicSingleVertexState::position() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  if (!thePosAvailable) computePosition();
  return thePos;
}


GlobalError BasicSingleVertexState::error() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  if (!theErrAvailable) computeError();
  return theErr;
}


GlobalWeight BasicSingleVertexState::weight() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  if (!theWeightAvailable) computeWeight();
  return theWeight;
}


AlgebraicVector3 BasicSingleVertexState::weightTimesPosition() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  if (!theWeightTimesPosAvailable) computeWeightTimesPos();
  return theWeightTimesPos;
}


double BasicSingleVertexState::weightInMixture() const 
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  return theWeightInMix;
}
// RefCountedVertexSeed BasicSingleVertexState::seedWithoutTracks() const
// {
//   RefCountedVertexSeed v = new VertexSeed(position(), error());
//   return v;
// }



void BasicSingleVertexState::computePosition() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  AlgebraicVector3 pos = error().matrix_new()*weightTimesPosition();
  thePos = GlobalPoint(pos[0], pos[1], pos[2]);
  thePosAvailable = true;
}


void BasicSingleVertexState::computeError() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  int ifail;
  theErr = weight().matrix().inverse(ifail);
  if (ifail != 0) throw VertexException("BasicSingleVertexState::could not invert weight matrix");
  theErrAvailable = true;
}


void BasicSingleVertexState::computeWeight() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  int ifail;
  theWeight = error().matrix().inverse(ifail);
  if (ifail != 0) throw VertexException("BasicSingleVertexState::could not invert error matrix");
  theWeightAvailable = true;
}


void BasicSingleVertexState::computeWeightTimesPos() const
{
  if (!valid) throw VertexException("BasicSingleVertexState::invalid");
  AlgebraicVector3 pos; pos(0) = position().x();
  pos(1) = position().y(); pos(2) = position().z();
  theWeightTimesPos = weight().matrix_new()*pos;
  theWeightTimesPosAvailable = true;
}


