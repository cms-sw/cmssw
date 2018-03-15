#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"

#include "PreMixingWorker.h"

namespace edm {
  template <typename DigiSimLinkCollection>
  class PreMixingDigiSimLinkWorker: public PreMixingWorker {
  public:
    PreMixingDigiSimLinkWorker(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector&& iC);
    ~PreMixingDigiSimLinkWorker() override = default;

    void initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) override {}
    void addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
    void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) override;
    void put(edm::Event& iEvent, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bunchSpacing) override;
    
  private:
    edm::EDGetTokenT<DigiSimLinkCollection> signalToken_;
    edm::InputTag pileupTag_;
    std::string collectionDM_; // secondary name to be given to new digis

    std::unique_ptr<DigiSimLinkCollection> merged_;
  };

  template <typename DigiSimLinkCollection>
  PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::PreMixingDigiSimLinkWorker(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector&& iC):
    signalToken_(iC.consumes<DigiSimLinkCollection>(ps.getParameter<edm::InputTag>("labelSig"))),
    pileupTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
    collectionDM_(ps.getParameter<std::string>("collectionDM"))
  {
    producer.produces<DigiSimLinkCollection>(collectionDM_);
  }

  template <typename DigiSimLinkCollection>
  void PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    Handle<DigiSimLinkCollection> digis;
    iEvent.getByToken(signalToken_, digis);

    if(digis.isValid()) {
      merged_ = std::make_unique<DigiSimLinkCollection>(*digis); // for signal we can just copy
    }
    else {
      merged_ = std::make_unique<DigiSimLinkCollection>();
    }
  }

  template <typename DigiSimLinkCollection>
  void PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) {
    edm::Handle<DigiSimLinkCollection> digis;
    pep.getByLabel(pileupTag_, digis);
    if(digis.isValid()) {
      for(const auto& detsetSource: *digis) {
        auto& detsetTarget = merged_->find_or_insert(detsetSource.detId());
        std::copy(detsetSource.begin(), detsetSource.end(), std::back_inserter(detsetTarget));
      }
    }
  }

  template <typename DigiSimLinkCollection>
  void PreMixingDigiSimLinkWorker<DigiSimLinkCollection>::put(edm::Event& iEvent, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bunchSpacing) {
    iEvent.put(std::move(merged_), collectionDM_);
  }

  // Specialize for DT
  template <>
  void PreMixingDigiSimLinkWorker<DTDigiSimLinkCollection>::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) {
    edm::Handle<DTDigiSimLinkCollection> digis;
    pep.getByLabel(pileupTag_, digis);
    if(digis.isValid()) {
      for(const auto& elem: *digis) {
        merged_->put(elem.second, elem.first);
      }
    }
  }
  
  using PreMixingPixelDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<PixelDigiSimLink> >;
  using PreMixingStripDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<StripDigiSimLink> >;
  using PreMixingRPCDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<RPCDigiSimLink> >;
  using PreMixingDTDigiSimLinkWorker = PreMixingDigiSimLinkWorker<DTDigiSimLinkCollection>;
}

// register plugins
#include "PreMixingWorkerFactory.h"
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, edm::PreMixingPixelDigiSimLinkWorker , "PreMixingPixelDigiSimLinkWorker");
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, edm::PreMixingStripDigiSimLinkWorker , "PreMixingStripDigiSimLinkWorker");
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, edm::PreMixingRPCDigiSimLinkWorker , "PreMixingRPCDigiSimLinkWorker");
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, edm::PreMixingDTDigiSimLinkWorker , "PreMixingDTDigiSimLinkWorker");
