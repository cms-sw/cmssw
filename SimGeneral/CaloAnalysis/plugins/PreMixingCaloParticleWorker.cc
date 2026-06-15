#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/SimClusterFwd.h"

#include "SimGeneral/PreMixingModule/interface/PreMixingWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
  /// (DetId, total simhit energy) pairs produced by CaloTruthAccumulator in premix stage1
  using EnergyMap = std::vector<std::pair<unsigned, float>>;

  constexpr std::size_t kNSimClusterProducts = 4;
  enum SimClusterIndex { kLegacy = 0, kBoundary = 1, kCaloParticle = 2, kMerged = 3 };

  // One state object per CaloTruth SimCluster collection. Premixing appends
  // signal and pileup collections into these outputs, so the RefProd is kept
  // here to rebuild refs after indices shift.
  struct SimClusterProduct {
    std::string
        instance;  ///< productInstanceName for this SimCluster collection (eg MergedCaloTruthBoundaryTrackSimCluster)
    edm::EDGetTokenT<SimClusterCollection> signalToken;
    std::unique_ptr<SimClusterCollection> output;
    SimClusterRefProd refProd;
  };

  // Maps produced by CaloTruthAccumulator are RefVectors whose refs point to
  // another CaloTruth collection. They cannot be copied verbatim after merging
  // events because each appended collection gets an index offset.
  struct SimClusterMapProduct {
    /// productInstanceName for this SimClusterRefVector (eg MergedCaloTruthBoundaryTrackSimCluster)
    std::string instance;
    edm::EDGetTokenT<SimClusterRefVector> signalToken;
    std::unique_ptr<SimClusterRefVector> output;
  };

  struct EventProducts {
    edm::Handle<CaloParticleCollection> caloparticle_h;
    edm::Handle<EnergyMap> energymap_h;
    std::array<edm::Handle<SimClusterCollection>, kNSimClusterProducts> simclusters_handles;
    std::array<edm::Handle<SimClusterRefVector>, kNSimClusterProducts> simclustersToCaloparticle_map_handles;
    edm::Handle<SimClusterRefVector> boundaryToMergedSimCluster_map_h;
  };

  /// Helper function to create InputTag updating the instance name
  edm::InputTag inputTagWithInstance(const edm::InputTag &tag, const std::string &instance) {
    return edm::InputTag(tag.label(), instance, tag.process());
  }

  void copySimClusters(const SimClusterCollection &clusters, SimClusterProduct &product);

  void copySimClusterRefVector(const SimClusterRefVector &inputRefs,
                               SimClusterRefVector &outputRefs,
                               const SimClusterRefProd &outputRefProd,
                               unsigned offset);

  template <typename SimCaloCollection>
  void normalize(SimCaloCollection &collection, const std::unordered_map<unsigned, float> &totalEnergy);
}  // namespace

class PreMixingCaloParticleWorker : public PreMixingWorker {
public:
  PreMixingCaloParticleWorker(const edm::ParameterSet &ps, edm::ProducesCollector, edm::ConsumesCollector &&iC);
  ~PreMixingCaloParticleWorker() override = default;

  void initializeEvent(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;
  void addSignals(edm::Event const &iEvent, edm::EventSetup const &iSetup) override;
  void addPileups(PileUpEventPrincipal const &pep, edm::EventSetup const &iSetup) override;
  void put(edm::Event &iEvent,
           edm::EventSetup const &iSetup,
           std::vector<PileupSummaryInfo> const &ps,
           int bunchSpacing) override;

private:
  void add(const EventProducts &products);
  void copyCaloParticles(const CaloParticleCollection &particles, unsigned legacyClusterOffset);

  edm::EDGetTokenT<CaloParticleCollection> sig_caloParticle_token_;
  edm::EDGetTokenT<EnergyMap> sig_energymap_token_;
  std::array<SimClusterProduct, kNSimClusterProducts> simClusterProducts_;
  std::array<SimClusterMapProduct, kNSimClusterProducts> clusterToCaloParticleMapProducts_;
  SimClusterMapProduct boundaryToMergedMapProduct_;

  edm::InputTag particlePileInputTag_;
  std::string collectionDM_;  ///< name prefix of instance name for all products (usually MergedCaloTruth)

  /// Keep track of the total energy per DetId on the fully mixed event.
  std::unordered_map<unsigned, float> totalEnergy_;

  std::unique_ptr<CaloParticleCollection> newCaloParticles_;
};

PreMixingCaloParticleWorker::PreMixingCaloParticleWorker(const edm::ParameterSet &ps,
                                                         edm::ProducesCollector producesCollector,
                                                         edm::ConsumesCollector &&iC)
    : sig_caloParticle_token_(iC.consumes<CaloParticleCollection>(ps.getParameter<edm::InputTag>("labelSig"))),
      sig_energymap_token_(iC.consumes<EnergyMap>(ps.getParameter<edm::InputTag>("labelSig"))),
      particlePileInputTag_(ps.getParameter<edm::InputTag>("pileInputTag")),
      collectionDM_(ps.getParameter<std::string>("collectionDM")) {
  const auto labelSig = ps.getParameter<edm::InputTag>("labelSig");
  // Keep this list in sync with CaloTruthAccumulator output instances.
  const std::array<std::string, kNSimClusterProducts> simClusterProductNames = {
      collectionDM_,
      collectionDM_ + "BoundaryTrackSimCluster",
      collectionDM_ + "CaloParticle",
      collectionDM_ + "MergedSimCluster"};

  for (unsigned i = 0; i < simClusterProducts_.size(); ++i) {
    simClusterProducts_[i].instance = simClusterProductNames[i];
    simClusterProducts_[i].signalToken =
        iC.consumes<SimClusterCollection>(inputTagWithInstance(labelSig, simClusterProductNames[i]));
    clusterToCaloParticleMapProducts_[i].instance = simClusterProductNames[i];
    clusterToCaloParticleMapProducts_[i].signalToken =
        iC.consumes<SimClusterRefVector>(inputTagWithInstance(labelSig, simClusterProductNames[i]));

    producesCollector.produces<SimClusterCollection>(simClusterProductNames[i]);
    producesCollector.produces<SimClusterRefVector>(simClusterProductNames[i]);
  }

  boundaryToMergedMapProduct_.instance = collectionDM_ + "MergedSimClusterMapFromSubCluster";
  boundaryToMergedMapProduct_.signalToken =
      iC.consumes<SimClusterRefVector>(inputTagWithInstance(labelSig, boundaryToMergedMapProduct_.instance));

  producesCollector.produces<CaloParticleCollection>(collectionDM_);
  producesCollector.produces<SimClusterRefVector>(boundaryToMergedMapProduct_.instance);
}

void PreMixingCaloParticleWorker::initializeEvent(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  for (auto &product : simClusterProducts_) {
    product.output = std::make_unique<SimClusterCollection>();
  }
  for (auto &product : clusterToCaloParticleMapProducts_) {
    product.output = std::make_unique<SimClusterRefVector>();
  }
  boundaryToMergedMapProduct_.output = std::make_unique<SimClusterRefVector>();
  newCaloParticles_ = std::make_unique<CaloParticleCollection>();

  // need RefProds in order to re-key the refs between merged products
  // TODO: try to remove const_cast, requires making Event non-const in
  // BMixingModule::initializeEvent
  for (auto &product : simClusterProducts_) {
    product.refProd = const_cast<edm::Event &>(iEvent).getRefBeforePut<SimClusterCollection>(product.instance);
  }
}

void PreMixingCaloParticleWorker::addSignals(edm::Event const &iEvent, edm::EventSetup const &iSetup) {
  EventProducts products;
  for (unsigned i = 0; i < simClusterProducts_.size(); ++i) {
    iEvent.getByToken(simClusterProducts_[i].signalToken, products.simclusters_handles[i]);
    iEvent.getByToken(clusterToCaloParticleMapProducts_[i].signalToken,
                      products.simclustersToCaloparticle_map_handles[i]);
  }
  iEvent.getByToken(boundaryToMergedMapProduct_.signalToken, products.boundaryToMergedSimCluster_map_h);

  iEvent.getByToken(sig_caloParticle_token_, products.caloparticle_h);
  iEvent.getByToken(sig_energymap_token_, products.energymap_h);

  if (products.caloparticle_h.isValid() && products.energymap_h.isValid()) {
    add(products);
  }
}

void PreMixingCaloParticleWorker::addPileups(PileUpEventPrincipal const &pep, edm::EventSetup const &iSetup) {
  EventProducts products;
  for (unsigned i = 0; i < simClusterProducts_.size(); ++i) {
    pep.getByLabel(inputTagWithInstance(particlePileInputTag_, simClusterProducts_[i].instance),
                   products.simclusters_handles[i]);
    pep.getByLabel(inputTagWithInstance(particlePileInputTag_, clusterToCaloParticleMapProducts_[i].instance),
                   products.simclustersToCaloparticle_map_handles[i]);
  }
  pep.getByLabel(inputTagWithInstance(particlePileInputTag_, boundaryToMergedMapProduct_.instance),
                 products.boundaryToMergedSimCluster_map_h);

  pep.getByLabel(particlePileInputTag_, products.caloparticle_h);
  pep.getByLabel(particlePileInputTag_, products.energymap_h);

  if (products.caloparticle_h.isValid() && products.energymap_h.isValid()) {
    add(products);
  }
}

void PreMixingCaloParticleWorker::add(const EventProducts &products) {
  // Capture sizes before appending this event. These offsets are applied to
  // refs read from the input event so they point into the final mixed products.
  std::array<unsigned, kNSimClusterProducts> clusterOffsets;
  for (unsigned i = 0; i < simClusterProducts_.size(); ++i) {
    clusterOffsets[i] = static_cast<unsigned>(simClusterProducts_[i].output->size());
  }

  for (unsigned i = 0; i < simClusterProducts_.size(); ++i) {
    if (products.simclusters_handles[i].isValid()) {
      copySimClusters(*products.simclusters_handles[i], simClusterProducts_[i]);
    }
    if (products.simclustersToCaloparticle_map_handles[i].isValid()) {
      copySimClusterRefVector(*products.simclustersToCaloparticle_map_handles[i],
                              *clusterToCaloParticleMapProducts_[i].output,
                              simClusterProducts_[kCaloParticle].refProd,
                              clusterOffsets[kCaloParticle]);
    }
  }

  if (products.boundaryToMergedSimCluster_map_h.isValid()) {
    copySimClusterRefVector(*products.boundaryToMergedSimCluster_map_h,
                            *boundaryToMergedMapProduct_.output,
                            simClusterProducts_[kMerged].refProd,
                            clusterOffsets[kMerged]);
  }

  copyCaloParticles(*products.caloparticle_h, clusterOffsets[kLegacy]);
  for (const auto &elem : *products.energymap_h) {
    totalEnergy_[elem.first] += elem.second;
  }
}

void PreMixingCaloParticleWorker::copyCaloParticles(const CaloParticleCollection &particles,
                                                    unsigned legacyClusterOffset) {
  newCaloParticles_->reserve(newCaloParticles_->size() + particles.size());
  for (const auto &inputParticle : particles) {
    newCaloParticles_->push_back(inputParticle);
    auto &particle = newCaloParticles_->back();

    // re-key the refs to legacy SimClusters
    particle.clearSimClusters();
    for (const auto &ref : inputParticle.simClusters()) {
      particle.addSimCluster(SimClusterRef(simClusterProducts_[kLegacy].refProd, legacyClusterOffset + ref.index()));
    }
  }
}

namespace {
  void copySimClusters(const SimClusterCollection &clusters, SimClusterProduct &product) {
    product.output->reserve(product.output->size() + clusters.size());
    std::copy(clusters.begin(), clusters.end(), std::back_inserter(*product.output));
  }

  void copySimClusterRefVector(const SimClusterRefVector &inputRefs,
                               SimClusterRefVector &outputRefs,
                               const SimClusterRefProd &outputRefProd,
                               unsigned offset) {
    outputRefs.reserve(outputRefs.size() + inputRefs.size());
    for (const auto &ref : inputRefs) {
      outputRefs.push_back(SimClusterRef(outputRefProd, offset + ref.index()));
    }
  }
}  // namespace

void PreMixingCaloParticleWorker::put(edm::Event &iEvent,
                                      edm::EventSetup const &iSetup,
                                      std::vector<PileupSummaryInfo> const &ps,
                                      int bunchSpacing) {
  for (auto &product : simClusterProducts_) {
    normalize(*product.output, totalEnergy_);
  }
  normalize(*newCaloParticles_, totalEnergy_);

  // clear memory
  std::unordered_map<unsigned, float>{}.swap(totalEnergy_);

  for (auto &product : simClusterProducts_) {
    iEvent.put(std::move(product.output), product.instance);
  }
  iEvent.put(std::move(newCaloParticles_), collectionDM_);
  for (auto &product : clusterToCaloParticleMapProducts_) {
    iEvent.put(std::move(product.output), product.instance);
  }
  iEvent.put(std::move(boundaryToMergedMapProduct_.output), boundaryToMergedMapProduct_.instance);
}

namespace {
  template <typename SimCaloCollection>
  void normalize(SimCaloCollection &collection, const std::unordered_map<unsigned, float> &totalEnergy) {
    // Stage-1 premixing stores absolute hit energies. After all signal and
    // pileup inputs have been accumulated here, convert them to fractions using
    // the total energy per DetId over the fully mixed event.
    for (auto &sc : collection) {
      auto hitsAndEnergies = sc.hits_and_fractions();
      sc.clearHitsAndFractions();
      for (auto &hAndE : hitsAndEnergies) {
        auto totalEnergyItr = totalEnergy.find(hAndE.first);
        const float totalenergy = totalEnergyItr != totalEnergy.end() ? totalEnergyItr->second : 0.f;
        float fraction = 0.;
        if (totalenergy > 0)
          fraction = hAndE.second / totalenergy;
        else
          edm::LogWarning("PreMixingCaloParticleWorker")
              << "TotalSimEnergy for hit " << hAndE.first << " is 0! The fraction for this hit cannot be computed.";
        sc.addRecHitAndFraction(hAndE.first, fraction);
      }
    }
  }
}  // namespace

DEFINE_PREMIXING_WORKER(PreMixingCaloParticleWorker);
