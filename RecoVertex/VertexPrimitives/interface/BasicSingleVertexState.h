#ifndef BasicSingleVertexState_H
#define BasicSingleVertexState_H

#include "RecoVertex/VertexPrimitives/interface/BasicVertexState.h"

//#include "CommonReco/CommonVertex/interface/RefCountedVertexSeed.h"


/** Single state measurement of a vertex.
 * Some data is calculated on demand to improve performance.
 */

class BasicSingleVertexState : public BasicVertexState {

public:

  /** Constructors
   */
  BasicSingleVertexState();
  BasicSingleVertexState(const GlobalPoint & pos, const GlobalError & posErr,
  		const double & weightInMix = 1.0);
  BasicSingleVertexState(const GlobalPoint & pos, const GlobalWeight & posWeight,
  		const double & weightInMix = 1.0);
  BasicSingleVertexState(const AlgebraicVector3 & weightTimesPosition,
		const GlobalWeight & posWeight,
  		const double & weightInMix = 1.0);

  /** Access methods
   */
  virtual BasicSingleVertexState* clone() const
  {
    return new BasicSingleVertexState(*this);
  }

  GlobalPoint position() const;
  GlobalError error() const;
  GlobalWeight weight() const;
  AlgebraicVector3 weightTimesPosition() const;
  double weightInMixture() const;

  /**
   * The validity of the vertex
   */
  bool isValid() const {return valid;}

private:

  void computePosition() const;
  void computeError() const;
  void computeWeight() const;
  void computeWeightTimesPos() const;

  mutable GlobalPoint thePos;
  mutable bool thePosAvailable;
  mutable GlobalError theErr;
  mutable bool theErrAvailable;
  mutable GlobalWeight theWeight;
  mutable bool theWeightAvailable;
  mutable AlgebraicVector3 theWeightTimesPos;
  mutable bool theWeightTimesPosAvailable;

  bool valid;
  double theWeightInMix;
};

#endif
