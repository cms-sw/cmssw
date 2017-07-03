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
  BasicMultiVertexState* clone() const override
  {
    return new BasicMultiVertexState(*this);
  }

  /**
   * Mean position of the mixture (position of the collapsed state)
   */
  GlobalPoint position() const override;

  /**
   * Mean time of the mixture (time of the collapsed state)
   */
  double time() const override;

  /**
   * Mean covariance matrix of the mixture
   * (covariance matrix of the collapsed state)
   */
  GlobalError error() const override;

  /**
   * Mean covariance matrix of the mixture
   * (covariance matrix of the collapsed state)
   */
  double timeError() const override;

  /**
   * Mean covariance matrix of the mixture
   * (covariance matrix of the collapsed state)
   */
  GlobalError error4D() const override;

  /**
   * Mean weight matrix (inverse of covariance) of the mixture
   * ( weight matrix of the collapsed state)
   */
  GlobalWeight weight() const override;

  /**
   * Mean weight matrix (inverse of covariance) of the mixture
   * ( weight matrix of the collapsed state)
   */
  GlobalWeight weight4D() const override;

  /**
   * Mean (weight*position) matrix of the mixture
   */
  AlgebraicVector3 weightTimesPosition() const override;

  /**
   * Mean (weight*position) matrix of the mixture
   */
  AlgebraicVector4 weightTimesPosition4D() const override;

  /**
   * The weight of this state. It will be the sum of the weights of the
   * individual components in the mixture.
   */
  double weightInMixture() const override;

  /**
   * Vector of individual components in the mixture.
   */
  std::vector<VertexState> components() const override {
    return theComponents;
  }

  /**
   * The validity of the vertex
   */
  bool isValid() const override {return valid;}

  bool is4D() const override { checkCombinedState(); return theCombinedState.is4D(); }

private:

  void checkCombinedState() const;

  bool valid;
  mutable std::vector<VertexState> theComponents;
  mutable VertexState theCombinedState;
  mutable bool theCombinedStateUp2Date;

  MultiVertexStateCombiner theCombiner;

};

#endif
