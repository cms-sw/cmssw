#ifndef SimGeneral_PreMixingModule_PreMixingMuonWorker_h
#define SimGeneral_PreMixingModule_PreMixingMuonWorker_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"

template <typename DigiCollection>
class PreMixingMuonWorker : public PreMixingWorker {
public:
  PreMixingMuonWorker(const edm::ParameterSet& ps,
                      edm::ProducesCollector producesCollector,
                      edm::ConsumesCollector&& iC)
      : PreMixingMuonWorker(ps, producesCollector, iC) {}
  PreMixingMuonWorker(const edm::ParameterSet& ps, edm::ProducesCollector, edm::ConsumesCollector& iC);
  ~PreMixingMuonWorker() override = default;

  void initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) override {}
  void addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) override;
  void put(edm::Event& iEvent,
           edm::EventSetup const& iSetup,
           std::vector<PileupSummaryInfo> const& ps,
           int bunchSpacing) override {
    put(iEvent);
  }

  void put(edm::Event& iEvent);

private:
  edm::EDGetTokenT<DigiCollection> signalToken_;
  edm::InputTag pileupTag_;
  std::string collectionDM_;  // secondary name to be given to new digis

  std::unique_ptr<DigiCollection> accumulated_;
};

template <typename DigiCollection>
PreMixingMuonWorker<DigiCollection>::PreMixingMuonWorker(const edm::ParameterSet& ps,
                                                         edm::ProducesCollector producesCollector,
                                                         edm::ConsumesCollector& iC)
    : signalToken_(iC.consumes<DigiCollection>(ps.getParameter<edm::InputTag>("digiTagSig"))),
      pileupTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
      collectionDM_(ps.getParameter<std::string>("collectionDM")) {
  producesCollector.produces<DigiCollection>(collectionDM_);
}

template <typename DigiCollection>
void PreMixingMuonWorker<DigiCollection>::addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  edm::Handle<DigiCollection> digis;
  iEvent.getByToken(signalToken_, digis);

  accumulated_ = std::make_unique<DigiCollection>(*digis);  // for signal we can just copy
}

template <typename DigiCollection>
void PreMixingMuonWorker<DigiCollection>::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) {
  edm::Handle<DigiCollection> digis;
  pep.getByLabel(pileupTag_, digis);
  for (const auto& elem : *digis) {
    accumulated_->put(elem.second, elem.first);
  }
}

template <typename DigiCollection>
void PreMixingMuonWorker<DigiCollection>::put(edm::Event& iEvent) {
  iEvent.put(std::move(accumulated_), collectionDM_);
}

#endif
