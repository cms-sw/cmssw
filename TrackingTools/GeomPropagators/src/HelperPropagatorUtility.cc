#include <memory>

#include "TrackingTools/GeomPropagators/interface/Propagator.h"

std::unique_ptr<Propagator> SetPropagationDirection (Propagator const & iprop,
                                                     PropagationDirection dir) {
  std::unique_ptr<Propagator> p(iprop.clone());
  p->setPropagationDirection(dir);

  return p;
}

