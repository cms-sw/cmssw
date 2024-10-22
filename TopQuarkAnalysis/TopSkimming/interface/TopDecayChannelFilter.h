#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

template <typename S>
class TopDecayChannelFilter : public edm::stream::EDFilter<> {
public:
  TopDecayChannelFilter(const edm::ParameterSet&);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;
  edm::InputTag src_;
  edm::EDGetTokenT<TtGenEvent> genEvt_;
  edm::EDGetTokenT<reco::GenParticleCollection> parts_;
  S sel_;
  bool checkedSrcType_;
  bool useTtGenEvent_;
};

template <typename S>
TopDecayChannelFilter<S>::TopDecayChannelFilter(const edm::ParameterSet& cfg)
    : src_(cfg.template getParameter<edm::InputTag>("src")),
      genEvt_(mayConsume<TtGenEvent>(src_)),
      parts_(mayConsume<reco::GenParticleCollection>(src_)),
      sel_(cfg),
      checkedSrcType_(false),
      useTtGenEvent_(false) {}

template <typename S>
bool TopDecayChannelFilter<S>::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<TtGenEvent> genEvt;

  if (!checkedSrcType_) {
    checkedSrcType_ = true;
    if (genEvt = iEvent.getHandle(genEvt_); genEvt.isValid()) {
      useTtGenEvent_ = true;
      return sel_(genEvt->particles(), src_.label());
    }
  } else {
    if (useTtGenEvent_) {
      genEvt = iEvent.getHandle(genEvt_);
      return sel_(genEvt->particles(), src_.label());
    }
  }
  const auto& parts = iEvent.get(parts_);
  return sel_(parts, src_.label());
}
