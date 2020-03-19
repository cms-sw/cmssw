/* \class GenPUProtonProducer
 *
 * Modification of GenParticleProducer.
 * Saves final state protons from HepMC events in Crossing Frame, in the generator-particle format.
 *
 * Note: Use the option USER_CXXFLAGS=-DEDM_ML_DEBUG with SCRAM in order to enable debug messages.
 *
 *    March  9, 2017   : Initial version.
 *    March 14, 2017   : Updated debug messages.
 *     July 27, 2017   : Removed extra loop initially inherited from GenParticleProducer. 
 *   August 17, 2017   : Replaced std::auto_ptr with std::unique_ptr. 
 * September 6, 2017   : Updated module to edm::global::EDProducer with ConvertParticle as RunCache following GenParticleProducer. 
 *
 */

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <vector>
#include <string>
#include <unordered_map>

namespace {

  class ConvertParticle {
  public:
    static constexpr int PDGCacheMax = 32768;
    static constexpr double mmToCm = 0.1;

    ConvertParticle()
        : abortOnUnknownPDGCode_(true), initialized_(false), chargeP_(PDGCacheMax, 0), chargeM_(PDGCacheMax, 0){};

    ConvertParticle(bool abortOnUnknownPDGCode)
        : abortOnUnknownPDGCode_(abortOnUnknownPDGCode),
          initialized_(false),
          chargeP_(PDGCacheMax, 0),
          chargeM_(PDGCacheMax, 0){};

    ~ConvertParticle(){};

    bool initialized() const { return initialized_; }

    void init(HepPDT::ParticleDataTable const& pdt) {
      if (!initialized_) {
        for (HepPDT::ParticleDataTable::const_iterator p = pdt.begin(); p != pdt.end(); ++p) {
          HepPDT::ParticleID const& id = p->first;
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
        initialized_ = true;
      }
    }

    bool operator()(reco::GenParticle& cand, HepMC::GenParticle const* part) const {
      reco::Candidate::LorentzVector p4(part->momentum());
      int pdgId = part->pdg_id();
      cand.setThreeCharge(chargeTimesThree(pdgId));
      cand.setPdgId(pdgId);
      cand.setStatus(part->status());
      cand.setP4(p4);
      cand.setCollisionId(0);
      HepMC::GenVertex const* v = part->production_vertex();
      if (v != nullptr) {
        HepMC::ThreeVector vtx = v->point3d();
        reco::Candidate::Point vertex(vtx.x() * mmToCm, vtx.y() * mmToCm, vtx.z() * mmToCm);
        cand.setVertex(vertex);
      } else {
        cand.setVertex(reco::Candidate::Point(0, 0, 0));
      }
      return true;
    }

  private:
    bool abortOnUnknownPDGCode_;
    bool initialized_;
    std::vector<int> chargeP_, chargeM_;
    std::unordered_map<int, int> chargeMap_;

    int chargeTimesThree(int id) const {
      if (std::abs(id) < PDGCacheMax)
        return id > 0 ? chargeP_[id] : chargeM_[-id];

      auto f = chargeMap_.find(id);
      if (f == chargeMap_.end()) {
        if (abortOnUnknownPDGCode_)
          throw edm::Exception(edm::errors::LogicError) << "invalid PDG id: " << id << std::endl;
        else
          return HepPDT::ParticleID(id).threeCharge();
      }
      return f->second;
    }
  };

  class SelectProton {
  public:
    bool operator()(HepMC::GenParticle const* part, double minPz) const {
      bool selection = ((!part->end_vertex() && part->status() == 1) && (part->pdg_id() == 2212) &&
                        (TMath::Abs(part->momentum().pz()) >= minPz));
      return selection;
    }
  };

}  // Anonymous namespace

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "PhysicsTools/HepMCCandAlgos/interface/MCTruthHelper.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

namespace edm {
  class ParameterSet;
}

class GenPUProtonProducer : public edm::global::EDProducer<edm::RunCache<ConvertParticle> > {
public:
  GenPUProtonProducer(const edm::ParameterSet&);
  ~GenPUProtonProducer() override;

  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup&) const override;
  std::shared_ptr<ConvertParticle> globalBeginRun(const edm::Run&, const edm::EventSetup&) const override;
  void globalEndRun(edm::Run const&, edm::EventSetup const&) const override{};

private:
  edm::EDGetTokenT<CrossingFrame<edm::HepMCProduct> > mixToken_;

  bool abortOnUnknownPDGCode_;
  std::vector<int> bunchList_;
  double minPz_;
  SelectProton select_;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <algorithm>

using namespace edm;
using namespace reco;
using namespace std;
using namespace HepMC;

GenPUProtonProducer::GenPUProtonProducer(const ParameterSet& cfg)
    : abortOnUnknownPDGCode_(cfg.getUntrackedParameter<bool>("abortOnUnknownPDGCode", true)),
      bunchList_(cfg.getParameter<vector<int> >("bunchCrossingList")),
      minPz_(cfg.getParameter<double>("minPz")) {
  produces<GenParticleCollection>();
  mixToken_ =
      consumes<CrossingFrame<HepMCProduct> >(InputTag(cfg.getParameter<std::string>("mix"), "generatorSmeared"));
}

GenPUProtonProducer::~GenPUProtonProducer() {}

std::shared_ptr<ConvertParticle> GenPUProtonProducer::globalBeginRun(const Run&, const EventSetup& es) const {
  ESHandle<HepPDT::ParticleDataTable> pdt;
  es.getData(pdt);
  auto convert_ptr = std::make_shared<ConvertParticle>(abortOnUnknownPDGCode_);
  if (!convert_ptr->initialized())
    convert_ptr->init(*pdt);

  return convert_ptr;
}

void GenPUProtonProducer::produce(StreamID, Event& evt, const EventSetup& es) const {
  size_t totalSize = 0;
  size_t npiles = 1;

  Handle<CrossingFrame<HepMCProduct> > cf;
  evt.getByToken(mixToken_, cf);
  std::unique_ptr<MixCollection<HepMCProduct> > cfhepmcprod(new MixCollection<HepMCProduct>(cf.product()));
  npiles = cfhepmcprod->size();

  LogDebug("GenPUProtonProducer") << " Number of pile-up events : " << npiles << endl;

  for (size_t icf = 0; icf < npiles; ++icf) {
    LogDebug("GenPUProtonProducer") << "CF " << icf
                                    << " size : " << cfhepmcprod->getObject(icf).GetEvent()->particles_size() << endl;
    totalSize += cfhepmcprod->getObject(icf).GetEvent()->particles_size();
  }
  LogDebug("GenPUProtonProducer") << "Total size : " << totalSize << endl;

  // Initialise containers
  auto candsPtr = std::make_unique<GenParticleCollection>();
  GenParticleCollection& cands = *candsPtr;

  // Loop over pile-up events
  ConvertParticle const& convertParticle_ = *runCache(evt.getRun().index());

  MixCollection<HepMCProduct>::MixItr mixHepMC_itr;
  unsigned int total_number_of_protons = 0;
  size_t idx_mix = 0;
  // Fill collection
  for (mixHepMC_itr = cfhepmcprod->begin(); mixHepMC_itr != cfhepmcprod->end(); ++mixHepMC_itr, ++idx_mix) {
    int bunch = mixHepMC_itr.bunch();

    if (find(bunchList_.begin(), bunchList_.end(), bunch) != bunchList_.end()) {
      auto event = (*mixHepMC_itr).GetEvent();

      size_t num_particles = event->particles_size();

      // Fill output collection
      unsigned int number_of_protons = 0;
      for (auto p = event->particles_begin(); p != event->particles_end(); ++p) {
        HepMC::GenParticle const* part = *p;
        if (select_(part, minPz_)) {
          reco::GenParticle cand;
          convertParticle_(cand, part);
          ++number_of_protons;
          cands.push_back(cand);
        }
      }
      LogDebug("GenPUProtonProducer") << "Idx : " << idx_mix << " Bunch : " << bunch
                                      << " Number of particles : " << num_particles
                                      << " Number of protons : " << number_of_protons << endl;

      total_number_of_protons += number_of_protons;
    }
  }
  LogDebug("GenPUProtonProducer") << "Total number of protons : " << total_number_of_protons << endl;
  LogDebug("GenPUProtonProducer") << "Output collection size : " << cands.size() << endl;

  evt.put(std::move(candsPtr));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(GenPUProtonProducer);
