#ifndef MultiVertexStateCombiner_H
#define MultiVertexStateCombiner_H

#include "RecoVertex/VertexPrimitives/interface/VertexState.h"

#include <vector>

/**
   * Class to collapse (combine) a Gaussian mixture of VertexStates
   * into one.
   * (c.f. R. Fruewirth et.al., Comp.Phys.Comm 100 (1997) 1
   */

class MultiVertexStateCombiner {
public:
  typedef std::vector<VertexState> VSC;

  VertexState combine(const VSC& theMixture) const;
};

#endif
