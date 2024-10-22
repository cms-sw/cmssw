#ifndef usercode_PrimaryVertexAnalyzer_VertexTimeAlgorithmBase_h
#define usercode_PrimaryVertexAnalyzer_VertexTimeAlgorithmBase_h
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
  class ConsumesCollector;
}  // namespace edm

class VertexTimeAlgorithmBase {
public:
  VertexTimeAlgorithmBase(const edm::ParameterSet& conf, edm::ConsumesCollector& iC) {}
  virtual ~VertexTimeAlgorithmBase() = default;
  VertexTimeAlgorithmBase(const VertexTimeAlgorithmBase&) = delete;
  VertexTimeAlgorithmBase& operator=(const VertexTimeAlgorithmBase&) = delete;

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc) {}

  virtual void setEvent(edm::Event& iEvent, edm::EventSetup const& iSetup) = 0;

  /**
   * estimate the vertex time and time uncertainty for transient vertex
   * 
   * returns true when a valid time has been determined, otherwise return false
   */
  virtual bool vertexTime(float& vtxTime, float& vtxTimeError, TransientVertex const& vtx) const = 0;

  /**
   * replace the vertices in the input vector by new vertices with time coordinates
   * determined by the vertexTime method
   * this implementation does not alter the weights from the previous fit
   * must be overridden to change weights, coordinates, tracklists or to add or remove vertices
   */
  virtual void fill_vertex_times(std::vector<TransientVertex>& pvs) {
    for (unsigned int i = 0; i < pvs.size(); i++) {
      auto vtx = pvs[i];
      if (vtx.isValid()) {
        auto vtxTime(0.f), vtxTimeError(-1.f);
        if (vertexTime(vtxTime, vtxTimeError, vtx)) {
          auto err = vtx.positionError().matrix4D();
          err(3, 3) = vtxTimeError * vtxTimeError;
          auto trkWeightMap3d = vtx.weightMap();
          auto vtx_with_time = TransientVertex(
              vtx.position(), vtxTime, err, vtx.originalTracks(), vtx.totalChiSquared(), vtx.degreesOfFreedom());
          vtx_with_time.weightMap(trkWeightMap3d);
          pvs[i] = vtx_with_time;
        }
      }
    }
  }
};

#endif
