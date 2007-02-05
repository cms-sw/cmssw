#ifndef GsfVertexWeightCalculator_H
#define GsfVertexWeightCalculator_H

#include "RecoVertex/VertexPrimitives/interface/RefCountedLinearizedTrackState.h"
#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

/**
 *  Calulates the (non-normalised) weight of a component the new mixture 
 *  of vertices for the Gaussian Sum vertex filter.
 *  (c.f. R. Fruewirth et.al., Comp.Phys.Comm 100 (1997) 1
 */

class GsfVertexWeightCalculator {

public:

/**
 *  Method to calculate the weight
 *
 */

   double calculate(const VertexState & oldVertex,
        const RefCountedLinearizedTrackState track, double cov) const;

};

#endif
