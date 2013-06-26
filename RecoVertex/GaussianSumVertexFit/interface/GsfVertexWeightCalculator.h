#ifndef GsfVertexWeightCalculator_H
#define GsfVertexWeightCalculator_H

#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

/**
 *  Calulates the (non-normalised) weight of a component the new mixture 
 *  of vertices for the Gaussian Sum vertex filter.
 *  (c.f. Th.Speer & R. Fruewirth, Comp.Phys.Comm 174, 935 (2006) )
 */

class GsfVertexWeightCalculator {

public:

  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

/**
 *  Method to calculate the weight
 *  A negative weight is returned in case of error.
 *
 */

   double calculate(const VertexState & oldVertex,
        const RefCountedLinearizedTrackState track, double cov) const;

private:
  typedef LinearizedTrackState<5>::AlgebraicVectorN AlgebraicVectorN;
  typedef LinearizedTrackState<5>::AlgebraicMatrixN3 AlgebraicMatrixN3;
  typedef LinearizedTrackState<5>::AlgebraicMatrixNM   AlgebraicMatrixNM;
  typedef LinearizedTrackState<5>::AlgebraicSymMatrixNN AlgebraicSymMatrixNN;

};

#endif
