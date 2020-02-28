#ifndef SimGeneral_PremixingModule_PreMixingDigiSimLinkWorker_h
#define SimGeneral_PremixingModule_PreMixingDigiSimLinkWorker_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"

template <typename DigiSimLinkCollection>
class PreMixingDigiSimLinkWorker : public PreMixingWorker {
public:
  PreMixingDigiSimLinkWorker(const edm::ParameterSet& ps, edm::ProducesCollector, edm::ConsumesCollector&& iC);
  ~PreMixingDigiSimLinkWorker() override = default;

  void initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) override {}
  void addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) override;
  void put(edm::Event& iEvent,
           edm::EventSetup const& iSetup,
           std::vector<PileupSummaryInfo> const& ps,
           int bunchSpacing) override;

private:
  edm::EDGetTokenT<DigiSimLinkCollection> signalToken_;
  edm::InputTag pileupTag_;
  std::string collectionDM_;  // secondary name to be given to new digis

  std::unique_ptr<DigiSimLinkCollection> merged_;
};

template <typename DigiSimLinkCollection>
PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::PreMixingDigiSimLinkWorker(const edm::ParameterSet& ps,
                                                                              edm::ProducesCollector producesCollector,
                                                                              edm::ConsumesCollector&& iC)
    : signalToken_(iC.consumes<DigiSimLinkCollection>(ps.getParameter<edm::InputTag>("labelSig"))),
      pileupTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
      collectionDM_(ps.getParameter<std::string>("collectionDM")) {
  producesCollector.produces<DigiSimLinkCollection>(collectionDM_);
}

template <typename DigiSimLinkCollection>
void PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::addSignals(edm::Event const& iEvent,
                                                                   edm::EventSetup const& iSetup) {
  edm::Handle<DigiSimLinkCollection> digis;
  iEvent.getByToken(signalToken_, digis);

  if (digis.isValid()) {
    merged_ = std::make_unique<DigiSimLinkCollection>(*digis);  // for signal we can just copy
  } else {
    merged_ = std::make_unique<DigiSimLinkCollection>();
  }
}

template <typename DigiSimLinkCollection>
void PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::addPileups(PileUpEventPrincipal const& pep,
                                                                   edm::EventSetup const& iSetup) {
  edm::Handle<DigiSimLinkCollection> digis;
  pep.getByLabel(pileupTag_, digis);
  if (digis.isValid()) {
    for (const auto& detsetSource : *digis) {
      auto& detsetTarget = merged_->find_or_insert(detsetSource.detId());
      std::copy(detsetSource.begin(), detsetSource.end(), std::back_inserter(detsetTarget));
    }
  }
}

template <typename DigiSimLinkCollection>
void PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::put(edm::Event& iEvent,
                                                            edm::EventSetup const& iSetup,
                                                            std::vector<PileupSummaryInfo> const& ps,
                                                            int bunchSpacing) {
  iEvent.put(std::move(merged_), collectionDM_);
}

#endif
