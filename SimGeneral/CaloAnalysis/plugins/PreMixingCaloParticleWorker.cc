#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

class PreMixingCaloParticleWorker: public PreMixingWorker {
public:
  PreMixingCaloParticleWorker(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector && iC);
  ~PreMixingCaloParticleWorker() override = default;

  void initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) override;
  void addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) override;
  void put(edm::Event& iEvent, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bunchSpacing) override;

private:
  using EnergyMap = std::vector<std::pair<unsigned, float> >;

  void add(const SimClusterCollection& clusters, const CaloParticleCollection& particles, const EnergyMap& energyMap);

  edm::EDGetTokenT<SimClusterCollection> sigClusterToken_;
  edm::EDGetTokenT<CaloParticleCollection> sigParticleToken_;
  edm::EDGetTokenT<EnergyMap> sigEnergyToken_;

  edm::InputTag particlePileInputTag_;
  std::string particleCollectionDM_;

  std::unordered_map<unsigned, float> totalEnergy_;

  std::unique_ptr<SimClusterCollection> newClusters_;
  std::unique_ptr<CaloParticleCollection> newParticles_;
  SimClusterRefProd clusterRef_;
};

PreMixingCaloParticleWorker::PreMixingCaloParticleWorker(const edm::ParameterSet& ps, edm::ProducerBase& producer, edm::ConsumesCollector && iC):
  sigClusterToken_(iC.consumes<SimClusterCollection>(ps.getParameter<edm::InputTag>("labelSig"))),
  sigParticleToken_(iC.consumes<CaloParticleCollection>(ps.getParameter<edm::InputTag>("labelSig"))),
  sigEnergyToken_(iC.consumes<EnergyMap>(ps.getParameter<edm::InputTag>("labelSig"))),
  particlePileInputTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
  particleCollectionDM_(ps.getParameter<std::string>("collectionDM"))
{
  producer.produces<SimClusterCollection>(particleCollectionDM_);
  producer.produces<CaloParticleCollection>(particleCollectionDM_);
}

void PreMixingCaloParticleWorker::initializeEvent(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  newClusters_ = std::make_unique<SimClusterCollection>();
  newParticles_ = std::make_unique<CaloParticleCollection>();

  // need RefProds in order to re-key the CaloParticle->SimCluster refs
  // TODO: try to remove const_cast, requires making Event non-const in BMixingModule::initializeEvent
  clusterRef_ = const_cast<edm::Event&>(iEvent).getRefBeforePut<SimClusterCollection>(particleCollectionDM_);
}

void PreMixingCaloParticleWorker::addSignals(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  edm::Handle<SimClusterCollection> clusters;
  iEvent.getByToken(sigClusterToken_, clusters);

  edm::Handle<CaloParticleCollection> particles;
  iEvent.getByToken(sigParticleToken_, particles);

  edm::Handle<EnergyMap> energy;
  iEvent.getByToken(sigEnergyToken_, energy);

  if(clusters.isValid() && particles.isValid() && energy.isValid()) {
    add(*clusters, *particles, *energy);
  }
}

void PreMixingCaloParticleWorker::addPileups(PileUpEventPrincipal const& pep, edm::EventSetup const& iSetup) {
  edm::Handle<SimClusterCollection> clusters;
  pep.getByLabel(particlePileInputTag_, clusters);

  edm::Handle<CaloParticleCollection> particles;
  pep.getByLabel(particlePileInputTag_, particles);

  edm::Handle<EnergyMap> energy;
  pep.getByLabel(particlePileInputTag_, energy);

  if(clusters.isValid() && particles.isValid() && energy.isValid()) {
    add(*clusters, *particles, *energy);
  }
}

void PreMixingCaloParticleWorker::add(const SimClusterCollection& clusters, const CaloParticleCollection& particles, const EnergyMap& energy) {
  const size_t startingIndex = newClusters_->size();

  // Copy SimClusters
  newClusters_->reserve(newClusters_->size() + clusters.size());
  std::copy(clusters.begin(), clusters.end(), std::back_inserter(*newClusters_));

  // Copy CaloParticles
  newParticles_->reserve(newParticles_->size() + particles.size());
  for(const auto& p: particles) {
    newParticles_->push_back(p);
    auto& particle = newParticles_->back();

    // re-key the refs to SimClusters
    particle.clearSimClusters();
    for(const auto& ref: p.simClusters()) {
      particle.addSimCluster(SimClusterRef(clusterRef_, startingIndex + ref.index()));
    }
  }

  // Add energies
  for(const auto elem: energy) {
    totalEnergy_[elem.first] += elem.second;
  }
}

void PreMixingCaloParticleWorker::put(edm::Event& iEvent, edm::EventSetup const& iSetup, std::vector<PileupSummaryInfo> const& ps, int bunchSpacing) {
  for (auto& sc : *newClusters_) {
    auto hitsAndEnergies = sc.hits_and_fractions();
    sc.clearHitsAndFractions();
    for (auto& hAndE : hitsAndEnergies) {
      const float totalenergy = totalEnergy_[hAndE.first];
      float fraction = 0.;
      if (totalenergy > 0)
        fraction = hAndE.second / totalenergy;
      else
        edm::LogWarning("PreMixingParticleWorker") << "TotalSimEnergy for hit " << hAndE.first
                                                   << " is 0! The fraction for this hit cannot be computed.";
      sc.addRecHitAndFraction(hAndE.first, fraction);
    }
  }

  // clear memory
  std::unordered_map<unsigned, float>{}.swap(totalEnergy_);

  iEvent.put(std::move(newClusters_), particleCollectionDM_);
  iEvent.put(std::move(newParticles_), particleCollectionDM_);
}

DEFINE_PREMIXING_WORKER(PreMixingCaloParticleWorker);
