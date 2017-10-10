#ifndef BasicSingleVertexState_H
#define BasicSingleVertexState_H

#include "RecoVertex/VertexPrimitives/interface/BasicVertexState.h"

//#include "CommonReco/CommonVertex/interface/RefCountedVertexSeed.h"


/** Single state measurement of a vertex.
 * Some data is calculated on demand to improve performance.
 */

class BasicSingleVertexState final : public BasicVertexState {

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

  // constructors with time (ignores off-diagonals in fit)
  BasicSingleVertexState(const GlobalPoint & pos, const GlobalError & posErr,
                         const double time, const double timeError,
                         const double & weightInMix = 1.0);
  BasicSingleVertexState(const GlobalPoint & pos, const GlobalWeight & posWeight,
                         const double time, const double timeWeight,
                         const double & weightInMix = 1.0);
  BasicSingleVertexState(const AlgebraicVector3 & weightTimesPosition, 
                         const GlobalWeight & posWeight,
                         const double weightTimesTime, const double timeWeight,
                         const double & weightInMix = 1.0);

  // constructors with time, full cov
  BasicSingleVertexState(const GlobalPoint & pos, const double time, 
                         const GlobalError & posTimeErr, const double & weightInMix = 1.0);
  BasicSingleVertexState(const GlobalPoint & pos, const double time, 
                         const GlobalWeight & posTimeWeight, const double & weightInMix = 1.0);
  BasicSingleVertexState(const AlgebraicVector4 & weightTimesPosition,
                         const GlobalWeight & posTimeWeight,
                         const double & weightInMix = 1.0);

  /** Access methods
   */
  BasicSingleVertexState* clone() const override
  {
    return new BasicSingleVertexState(*this);
  }

  GlobalPoint position() const override;
  GlobalError error() const override;
  GlobalError error4D() const override;
  double time() const override;
  double timeError() const override;
  GlobalWeight weight() const override;
  GlobalWeight weight4D() const override;
  AlgebraicVector3 weightTimesPosition() const override;
  AlgebraicVector4 weightTimesPosition4D() const override;
  double weightInMixture() const override;

  /**
   * The validity of the vertex
   */
  bool isValid() const override {return valid;}
  bool is4D() const override { return vertexIs4D; }

private:

  void computePosition() const;
  void computeError() const;
  void computeWeight() const;
  void computeWeightTimesPos() const;

  mutable GlobalPoint thePos;
  mutable double theTime;

  mutable GlobalError theErr;
  mutable GlobalWeight theWeight;

  mutable AlgebraicVector4 theWeightTimesPos;
  double theWeightInMix;

  mutable bool thePosAvailable;  
  mutable bool theTimeAvailable;
  mutable bool theErrAvailable;
  mutable bool theWeightAvailable;
  mutable bool theWeightTimesPosAvailable;

  bool valid;
  bool vertexIs4D;  
};

#endif
