#ifndef usercode_PrimaryVertexAnalyzer_VertexTimeLegacy4D_h
#define usercode_PrimaryVertexAnalyzer_VertexTimeLegacy4D_h

#include "VertexTimeAlgorithmBase.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

class VertexTimeAlgorithmLegacy4D : public VertexTimeAlgorithmBase {
public:
  VertexTimeAlgorithmLegacy4D(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  ~VertexTimeAlgorithmLegacy4D() override = default;

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  void setEvent(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  bool vertexTime(float& vtxTime, float& vtxTimeError, TransientVertex const& vtx) const override;
};

#endif
