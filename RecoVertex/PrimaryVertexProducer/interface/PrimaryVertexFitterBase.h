#ifndef PrimaryVertexFitterBase_h
#define PrimaryVertexFitterBase_h

#include <vector>
#include "DataFormats/BeamSpot/interface/BeamSpotFwd.h"

/**\class PrimaryVertexFitterBase
 
  Description: base class for primary vertex fitters

*/
namespace edm {
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

namespace reco {
  class TransientTrack;
}  // namespace reco

class TransientVertex;

class PrimaryVertexFitterBase {
public:
  PrimaryVertexFitterBase(const edm::ParameterSet &conf) {}
  PrimaryVertexFitterBase() {}
  virtual ~PrimaryVertexFitterBase() = default;
  virtual std::vector<TransientVertex> fit(const std::vector<reco::TransientTrack> &,
                                           const std::vector<TransientVertex> &,
                                           const reco::BeamSpot &,
                                           const bool) = 0;
};
#endif
