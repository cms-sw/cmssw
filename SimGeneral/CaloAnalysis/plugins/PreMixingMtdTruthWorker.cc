#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerClusterFwd.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

class PreMixingMtdTruthWorker : public PreMixingWorker {
public:
  PreMixingMtdTruthWorker(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &&iC);
  ~PreMixingMtdTruthWorker() override = default;

  void initializeEvent(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;
  void addSignals(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;
  void addPileups(PileUpEventPrincipal const &pep, edm::EventSetup const &iSetup) override;
  void put(edm::Event &iEvent,
           edm::EventSetup const &iSetup,
           std::vector<PileupSummaryInfo> const &ps,
           int bunchSpacing) override;

private:
  void add(const MtdSimLayerClusterCollection &clusters);

  edm::EDGetTokenT<MtdSimLayerClusterCollection> sigClusterToken_;

  edm::InputTag clusterPileInputTag_;
  std::string mtdSimLCCollectionDM_;

  std::unique_ptr<MtdSimLayerClusterCollection> newClusters_;
};

PreMixingMtdTruthWorker::PreMixingMtdTruthWorker(const edm::ParameterSet &ps,
                                                 edm::ProducesCollector producesCollector,
                                                 edm::ConsumesCollector &&iC)
    : sigClusterToken_(iC.consumes<MtdSimLayerClusterCollection>(ps.getParameter<edm::InputTag>("labelSig"))),
      clusterPileInputTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
      mtdSimLCCollectionDM_(ps.getParameter<std::string>("collectionDM")) {
  producesCollector.produces<MtdSimLayerClusterCollection>(mtdSimLCCollectionDM_);
}

void PreMixingMtdTruthWorker::initializeEvent(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  newClusters_ = std::make_unique<MtdSimLayerClusterCollection>();
}

void PreMixingMtdTruthWorker::addSignals(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  edm::Handle<MtdSimLayerClusterCollection> clusters;
  iEvent.getByToken(sigClusterToken_, clusters);

  if (clusters.isValid()) {
    add(*clusters);
  }
}

void PreMixingMtdTruthWorker::addPileups(PileUpEventPrincipal const &pep, edm::EventSetup const &iSetup) {
  edm::Handle<MtdSimLayerClusterCollection> clusters;
  pep.getByLabel(clusterPileInputTag_, clusters);

  if (clusters.isValid()) {
    add(*clusters);
  }
}

void PreMixingMtdTruthWorker::add(const MtdSimLayerClusterCollection &clusters) {
  // Copy MtdSimLayerClusters
  newClusters_->reserve(newClusters_->size() + clusters.size());
  std::copy(clusters.begin(), clusters.end(), std::back_inserter(*newClusters_));
}

void PreMixingMtdTruthWorker::put(edm::Event &iEvent,
                                  edm::EventSetup const &iSetup,
                                  std::vector<PileupSummaryInfo> const &ps,
                                  int bunchSpacing) {
  iEvent.put(std::move(newClusters_), mtdSimLCCollectionDM_);
}

DEFINE_PREMIXING_WORKER(PreMixingMtdTruthWorker);
