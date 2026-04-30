// PUGenParticleProducer.cc
//
// Reads CrossingFrame<HepMCProduct> produced by MixingModule and creates a
// reco::GenParticleCollection containing ALL pileup particles with:
//   - full mother/daughter Ref<> links (within each PU sub-event)
//   - production vertex (mm -> cm)
//   - GenStatusFlags via MCTruthHelper
//   - collisionId() == bunch-crossing number (0 = in-time PU, non-zero = OOT PU)
// Signal particles are excluded (they are already in the standard
// "genParticles" collection).
//
// This collection is designed to be fed into GenParticlePruner to apply
// the standard prunedGenParticles selection.

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthHelper.h"

#include <vector>
#include <unordered_map>

static constexpr int PDGCacheMax = 32768;
static constexpr double mmToCm = 0.1;

// ---- Charge lookup cache (copied from GenParticleProducer) ----
namespace {
  struct IDto3Charge {
    IDto3Charge(HepPDT::ParticleDataTable const&, bool abortOnUnknownPDGCode);
    int chargeTimesThree(int id) const;

  private:
    std::vector<int> chargeP_, chargeM_;
    std::unordered_map<int, int> chargeMap_;
    bool abortOnUnknownPDGCode_;
  };

  IDto3Charge::IDto3Charge(HepPDT::ParticleDataTable const& iTable, bool iAbortOnUnknownPDGCode)
      : chargeP_(PDGCacheMax, 0), chargeM_(PDGCacheMax, 0), abortOnUnknownPDGCode_(iAbortOnUnknownPDGCode) {
    for (auto const& p : iTable) {
      const HepPDT::ParticleID& id = p.first;
      int pdgId = id.pid(), apdgId = std::abs(pdgId);
      int q3 = id.threeCharge();
      if (apdgId < PDGCacheMax && pdgId > 0) {
        chargeP_[apdgId] = q3;
        chargeM_[apdgId] = -q3;
      } else if (apdgId < PDGCacheMax) {
        chargeP_[apdgId] = -q3;
        chargeM_[apdgId] = q3;
      } else {
        chargeMap_.emplace(pdgId, q3);
        chargeMap_.emplace(-pdgId, -q3);
      }
    }
  }

  int IDto3Charge::chargeTimesThree(int id) const {
    if (std::abs(id) < PDGCacheMax)
      return id > 0 ? chargeP_[id] : chargeM_[-id];
    auto f = chargeMap_.find(id);
    if (f == chargeMap_.end()) {
      if (abortOnUnknownPDGCode_)
        throw edm::Exception(edm::errors::LogicError) << "PUGenParticleProducer: invalid PDG id: " << id;
      else
        return HepPDT::ParticleID(id).threeCharge();
    }
    return f->second;
  }
}  // namespace

// ---- The producer ----
class PUGenParticleProducer : public edm::global::EDProducer<edm::RunCache<IDto3Charge>> {
public:
  explicit PUGenParticleProducer(const edm::ParameterSet&);
  ~PUGenParticleProducer() override = default;

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  std::shared_ptr<IDto3Charge> globalBeginRun(const edm::Run&, const edm::EventSetup&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}

private:
  void convertParticle(reco::GenParticle& cand, const HepMC::GenParticle* part, const IDto3Charge& id2Charge) const;

  void fillDaughters(reco::GenParticleCollection& cands,
                     const HepMC::GenParticle* part,
                     reco::GenParticleRefProd const& ref,
                     size_t index,
                     std::unordered_map<int, size_t>& barcodes) const;

  void fillIndices(const HepMC::GenEvent* mc,
                   std::vector<const HepMC::GenParticle*>& particles,
                   int offset,
                   std::unordered_map<int, size_t>& barcodes) const;

  edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct>> cfToken_;
  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> pdtToken_;

  MCTruthHelper<HepMC::GenParticle> mcTruthHelper_;
};

// ---- Constructor ----
PUGenParticleProducer::PUGenParticleProducer(const edm::ParameterSet& cfg)
    : cfToken_(consumes<CrossingFrame<edm::HepMCProduct>>(cfg.getParameter<edm::InputTag>("src"))) {
  pdtToken_ = esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord, edm::Transition::BeginRun>();
  produces<reco::GenParticleCollection>();
}

// ---- Run-level charge table setup ----
std::shared_ptr<IDto3Charge> PUGenParticleProducer::globalBeginRun(const edm::Run&, const edm::EventSetup& es) const {
  edm::ESHandle<HepPDT::ParticleDataTable> pdt = es.getHandle(pdtToken_);
  return std::make_shared<IDto3Charge>(*pdt, false);  // don't abort on unknown PDG codes
}

// ---- Per-event produce ----
void PUGenParticleProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  edm::Handle<CrossingFrame<edm::HepMCProduct>> cfHandle;
  evt.getByToken(cfToken_, cfHandle);

  MixCollection<edm::HepMCProduct> mix(cfHandle.product());

  // --- First pass: collect PU events with their BX and count total particles ---
  struct PUEvent {
    const HepMC::GenEvent* genEvt;
    int bx;
  };
  std::vector<PUEvent> puEvents;
  size_t totalSize = 0;

  for (auto it = mix.begin(); it != mix.end(); ++it) {
    // Skip signal events
    if (it.getTrigger())
      continue;

    const HepMC::GenEvent* genEvt = (*it).GetEvent();
    if (!genEvt)
      continue;

    int bx = it.bunch();
    size_t npart = genEvt->particles_size();
    if (npart == 0)
      continue;

    puEvents.push_back({genEvt, bx});
    totalSize += npart;
  }

  // --- Pre-allocate output collection and get RefBeforePut ---
  auto candsPtr = std::make_unique<reco::GenParticleCollection>(totalSize);
  reco::GenParticleRefProd ref = evt.getRefBeforePut<reco::GenParticleCollection>();
  reco::GenParticleCollection& cands = *candsPtr;

  // --- Particle pointer array (parallel to cands) ---
  std::vector<const HepMC::GenParticle*> particles(totalSize);

  const IDto3Charge& id2Charge = *runCache(evt.getRun().index());

  // --- Second pass: fill per sub-event ---
  size_t offset = 0;
  for (const auto& puEvt : puEvents) {
    const HepMC::GenEvent* mc = puEvt.genEvt;
    int bx = puEvt.bx;
    size_t numParticles = mc->particles_size();

    // Build barcode -> index map for this sub-event
    std::unordered_map<int, size_t> barcodes;
    barcodes.reserve(numParticles);
    fillIndices(mc, particles, offset, barcodes);

    // Convert each HepMC particle to reco::GenParticle
    for (size_t ipar = offset; ipar < offset + numParticles; ++ipar) {
      const HepMC::GenParticle* part = particles[ipar];
      reco::GenParticle& cand = cands[ipar];
      convertParticle(cand, part, id2Charge);
      cand.resetDaughters(ref.id());
      cand.setCollisionId(bx);
    }

    // Fill mother/daughter references
    for (size_t d = offset; d < offset + numParticles; ++d) {
      const HepMC::GenParticle* part = particles[d];
      if (part->production_vertex() != nullptr) {
        fillDaughters(cands, part, ref, d, barcodes);
      }
    }

    offset += numParticles;
  }

  evt.put(std::move(candsPtr));
}

// ---- Convert a single HepMC particle to reco::GenParticle ----
void PUGenParticleProducer::convertParticle(reco::GenParticle& cand,
                                            const HepMC::GenParticle* part,
                                            const IDto3Charge& id2Charge) const {
  reco::Candidate::LorentzVector p4(part->momentum());
  int pdgId = part->pdg_id();

  cand.setThreeCharge(id2Charge.chargeTimesThree(pdgId));
  cand.setPdgId(pdgId);
  cand.setStatus(part->status());
  cand.setP4(p4);
  cand.setCollisionId(0);  // will be overwritten with BX

  const HepMC::GenVertex* v = part->production_vertex();
  if (v != nullptr) {
    HepMC::ThreeVector vtx = v->point3d();
    reco::Candidate::Point vertex(vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm);
    cand.setVertex(vertex);
  } else {
    cand.setVertex(reco::Candidate::Point(0, 0, 0));
  }

  mcTruthHelper_.fillGenStatusFlags(*part, cand.statusFlags());
}

// ---- Fill mother/daughter Refs for a particle ----
void PUGenParticleProducer::fillDaughters(reco::GenParticleCollection& cands,
                                          const HepMC::GenParticle* part,
                                          reco::GenParticleRefProd const& ref,
                                          size_t index,
                                          std::unordered_map<int, size_t>& barcodes) const {
  const HepMC::GenVertex* productionVertex = part->production_vertex();
  size_t numberOfMothers = productionVertex->particles_in_size();
  if (numberOfMothers > 0) {
    for (auto motherIt = productionVertex->particles_in_const_begin();
         motherIt != productionVertex->particles_in_const_end();
         ++motherIt) {
      const HepMC::GenParticle* mother = *motherIt;
      auto bmIt = barcodes.find(mother->barcode());
      if (bmIt != barcodes.end()) {
        size_t m = bmIt->second;
        cands[m].addDaughter(reco::GenParticleRef(ref, index));
        cands[index].addMother(reco::GenParticleRef(ref, m));
      }
    }
  }
}

// ---- Build barcode -> index map and fill particle pointer array ----
void PUGenParticleProducer::fillIndices(const HepMC::GenEvent* mc,
                                        std::vector<const HepMC::GenParticle*>& particles,
                                        int offset,
                                        std::unordered_map<int, size_t>& barcodes) const {
  size_t idx = offset;
  for (auto p = mc->particles_begin(); p != mc->particles_end(); ++p) {
    const HepMC::GenParticle* particle = *p;
    int barCode = particle->barcode();
    particles[idx] = particle;
    barcodes.insert(std::make_pair(barCode, idx));
    ++idx;
  }
}

DEFINE_FWK_MODULE(PUGenParticleProducer);
