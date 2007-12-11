#ifndef GsfVertexWeightCalculator_H
#define GsfVertexWeightCalculator_H

#include "RecoVertex/VertexPrimitives/interface/LinearizedTrackState.h"
#include "DataFormats/GeometrySurface/interface/ReferenceCounted.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

/**
 *  Calulates the (non-normalised) weight of a component the new mixture 
 *  of vertices for the Gaussian Sum vertex filter.
 *  (c.f. R. Fruewirth et.al., Comp.Phys.Comm 100 (1997) 1
 */

class GsfVertexWeightCalculator {

public:

  typedef ReferenceCountingPointer<LinearizedTrackState<5> > RefCountedLinearizedTrackState;

/**
 *  Method to calculate the weight
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
