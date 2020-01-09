#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"

namespace edm {
  template <typename T>
  class PreMixingCrossingFrameWorker : public PreMixingWorker {
  public:
    PreMixingCrossingFrameWorker(const edm::ParameterSet& ps, edm::ProducesCollector, edm::ConsumesCollector&& iC);
    ~PreMixingCrossingFrameWorker() override = default;

    void initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) override {}
    void addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
    void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) override;
    void put(edm::Event& iEvent,
             edm::EventSetup const& iSetup,
             std::vector<PileupSummaryInfo> const& ps,
             int bunchSpacing) override;

  private:
    edm::EDGetTokenT<CrossingFrame<T> > signalToken_;
    edm::InputTag pileupTag_;
    std::string collectionDM_;  // secondary name to be given to new digis

    std::unique_ptr<PCrossingFrame<T> > merged_;
  };

  template <typename T>
  PreMixingCrossingFrameWorker<T>::PreMixingCrossingFrameWorker(const edm::ParameterSet& ps,
                                                                edm::ProducesCollector producesCollector,
                                                                edm::ConsumesCollector&& iC)
      : signalToken_(iC.consumes<CrossingFrame<T> >(ps.getParameter<edm::InputTag>("labelSig"))),
        pileupTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
        collectionDM_(ps.getParameter<std::string>("collectionDM")) {
    producesCollector.produces<PCrossingFrame<T> >(
        collectionDM_);  // TODO: this is needed only to store the pointed-to objects, do we really need it?
    producesCollector.produces<CrossingFrame<T> >(collectionDM_);
  }

  template <typename T>
  void PreMixingCrossingFrameWorker<T>::addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
    edm::Handle<CrossingFrame<T> > hcf;
    iEvent.getByToken(signalToken_, hcf);

    const auto& cf = *hcf;
    if (!cf.getPileups().empty()) {
      throw cms::Exception("LogicError") << "Got CrossingFrame from signal with pileup content?";
    }

    merged_ = std::make_unique<PCrossingFrame<T> >(cf);  // for signal we just copy
  }

  template <typename T>
  void PreMixingCrossingFrameWorker<T>::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) {
    edm::Handle<PCrossingFrame<T> > hcf;
    pep.getByLabel(pileupTag_, hcf);

    const auto& cf = *hcf;
    if (!cf.getSignals().empty()) {
      throw cms::Exception("LogicError") << "Got PCrossingFrame from pileup with signal content?";
    }

    merged_->setAllExceptSignalFrom(cf);
  }

  template <typename T>
  void PreMixingCrossingFrameWorker<T>::put(edm::Event& iEvent,
                                            edm::EventSetup const& iSetup,
                                            std::vector<PileupSummaryInfo> const& ps,
                                            int bunchSpacing) {
    auto orphanHandle = iEvent.put(std::move(merged_), collectionDM_);
    const auto& pcf = *orphanHandle;

    const auto bx = pcf.getBunchRange();
    auto cf = std::make_unique<CrossingFrame<T> >(
        bx.first, bx.second, pcf.getBunchSpace(), pcf.getSubDet(), pcf.getMaxNbSources());
    cf->addSignals(&(pcf.getSignals()), pcf.getEventID());
    cf->setPileups(pcf.getPileups());
    iEvent.put(std::move(cf), collectionDM_);
  }
}  // namespace edm

#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
using PreMixingCrossingFramePSimHitWorker = edm::PreMixingCrossingFrameWorker<PSimHit>;

DEFINE_PREMIXING_WORKER(PreMixingCrossingFramePSimHitWorker);
