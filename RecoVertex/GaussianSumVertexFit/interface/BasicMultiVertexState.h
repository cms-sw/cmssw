#ifndef BasicMultiVertexState_H
#define BasicMultiVertexState_H

#include "RecoVertex/VertexPrimitives/interface/BasicVertexState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"
#include "RecoVertex/GaussianSumVertexFit/interface/MultiVertexStateCombiner.h"

/** Multi state measurement of a vertex.
 * Some data is calculated on demand to improve performance.
 */

class BasicMultiVertexState : public BasicVertexState {

public:

  /** Constructors
   */
  BasicMultiVertexState() : valid(false){}

  BasicMultiVertexState(const std::vector<VertexState>& vsComp);

  /** Access methods
   */
  virtual BasicMultiVertexState* clone() const
  {
    return new BasicMultiVertexState(*this);
  }

  /**
   * Mean position of the mixture (position of the collapsed state)
   */
  GlobalPoint position() const;

  /**
   * Mean covariance matrix of the mixture
   * (covariance matrix of the collapsed state)
   */
  GlobalError error() const;

  /**
   * Mean weight matrix (inverse of covariance) of the mixture
   * ( weight matrix of the collapsed state)
   */
  GlobalWeight weight() const;

  /**
   * Mean (weight*position) matrix of the mixture
   */
  AlgebraicVector3 weightTimesPosition() const;

  /**
   * The weight of this state. It will be the sum of the weights of the
   * individual components in the mixture.
   */
  double weightInMixture() const;

  /**
   * Vector of individual components in the mixture.
   */
  virtual std::vector<VertexState> components() const {
    return theComponents;
  }

  /**
   * The validity of the vertex
   */
  bool isValid() const {return valid;}

private:

  void checkCombinedState() const;

  bool valid;
  mutable std::vector<VertexState> theComponents;
  mutable VertexState theCombinedState;
  mutable bool theCombinedStateUp2Date;

  MultiVertexStateCombiner theCombiner;

};

#endif
