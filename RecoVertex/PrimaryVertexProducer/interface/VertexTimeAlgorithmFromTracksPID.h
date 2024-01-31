
#ifndef usercode_PrimaryVertexAnalyzer_VertexTimeAlgorithmFromTracksPID_h
#define usercode_PrimaryVertexAnalyzer_VertexTimeAlgorithmFromTracksPID_h

#include "VertexTimeAlgorithmBase.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/ValueMap.h"

class VertexTimeAlgorithmFromTracksPID : public VertexTimeAlgorithmBase {
public:
  VertexTimeAlgorithmFromTracksPID(const edm::ParameterSet& conf, edm::ConsumesCollector& iC);
  ~VertexTimeAlgorithmFromTracksPID() override = default;

  static void fillPSetDescription(edm::ParameterSetDescription& iDesc);

  void setEvent(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  bool vertexTime(float& vtxTime, float& vtxTimeError, TransientVertex const& vtx) const override;

protected:
  struct TrackInfo {
    double trkWeight;
    double trkTimeError;
    double trkTimeHyp[3];
  };

  edm::EDGetTokenT<edm::ValueMap<float>> const trackMTDTimeToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> const trackMTDTimeErrorToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> const trackMTDTimeQualityToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> const trackMTDTofPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> const trackMTDTofKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> const trackMTDTofPToken_;

  double const minTrackVtxWeight_;
  double const minTrackTimeQuality_;
  double const probPion_;
  double const probKaon_;
  double const probProton_;
  double const Tstart_;
  double const coolingFactor_;

  edm::ValueMap<float> trackMTDTimes_;
  edm::ValueMap<float> trackMTDTimeErrors_;
  edm::ValueMap<float> trackMTDTimeQualities_;
  edm::ValueMap<float> trackMTDTofPi_;
  edm::ValueMap<float> trackMTDTofK_;
  edm::ValueMap<float> trackMTDTofP_;
};

#endif
