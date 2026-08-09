/**
 * Builds a vector<int> as a mask associated to a trackster collection, to identify tracksters that are best-matched to simulation
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include "DataFormats/HGCalReco/interface/Common.h"

#include <CLHEP/Units/SystemOfUnits.h>

#include <numeric>
#include <vector>
#include <algorithm>

// using namespace ticl;
using std::vector;
using ticl::AssociationMap;
using ticl::Trackster;
using ticl::TracksterCollection;

using TracksterToTracksterMap =
    ticl::AssociationMap<ticl::mapWithSharedEnergyAndScore, vector<ticl::Trackster>, vector<ticl::Trackster>>;

class TracksterSuperclusteringValidCandidateMaskProducer : public edm::stream::EDProducer<> {
public:
  explicit TracksterSuperclusteringValidCandidateMaskProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<TracksterCollection> tracksters_token_;

  const edm::EDGetTokenT<TracksterToTracksterMap> simToRecoAssociator_;
  const edm::EDGetTokenT<TracksterToTracksterMap> recoToSimAssociator_;

  const edm::EDPutTokenT<vector<int>> mask_output_token_;

  float recoToSimScoreCut_;
  bool ignoreSuperclusterSeed_;
  vector<Trackster::ParticleType> particleTypes_signal_;
};

DEFINE_FWK_MODULE(TracksterSuperclusteringValidCandidateMaskProducer);

TracksterSuperclusteringValidCandidateMaskProducer::TracksterSuperclusteringValidCandidateMaskProducer(
    const edm::ParameterSet& ps)
    : tracksters_token_(consumes<TracksterCollection>(ps.getParameter<edm::InputTag>("tracksters"))),
      simToRecoAssociator_(consumes<TracksterToTracksterMap>(ps.getParameter<edm::InputTag>("associatorSimToReco"))),
      recoToSimAssociator_(consumes<TracksterToTracksterMap>(ps.getParameter<edm::InputTag>("associatorRecoToSim"))),
      mask_output_token_(produces<vector<int>>()),
      recoToSimScoreCut_(ps.getParameter<double>("recoToSimScoreCut")),
      ignoreSuperclusterSeed_(ps.getParameter<bool>("ignoreSuperclusterSeed")),
      particleTypes_signal_() {
  vector<int> v = ps.getParameter<vector<int>>("particleTypesSignal");
  particleTypes_signal_.reserve(v.size());
  for (auto x : v)
    particleTypes_signal_.push_back(static_cast<Trackster::ParticleType>(x));
}

void TracksterSuperclusteringValidCandidateMaskProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksters")->setComment("Input tag for tracksters");

  desc.add<edm::InputTag>("associatorRecoToSim")->setComment("Input tag for the SimToReco associator");
  desc.add<edm::InputTag>("associatorSimToReco")->setComment("Input tag for the RecoToSim associator");

  desc.add<double>("recoToSimScoreCut")
      ->setComment("Cut on reco2Sim score to consider a trackster 'signal' (if reco2Sim < cut)");
  desc.add<vector<int>>("particleTypesSignal")
      ->setComment("ticl::Trackster::ParticleType list to filter on SimTrackster (ie electron is 1, photon 0)");
  desc.add<bool>("ignoreSuperclusterSeed", true)
      ->setComment(
          "Do not include the 'seed' trackster of the supercluster (the one with largest shared energy with sim "
          "object) in the output mask");

  descriptions.addWithDefaultLabel(desc);
}

void TracksterSuperclusteringValidCandidateMaskProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  TracksterCollection const& tracksters = evt.get(tracksters_token_);
  TracksterToTracksterMap const& simToRecoMap = evt.get(simToRecoAssociator_);
  TracksterToTracksterMap const& recoToSimMap = evt.get(recoToSimAssociator_);

  vector<int> mask_res(tracksters.size(), 1);

  for (size_t i_sim = 0; i_sim < simToRecoMap.size(); ++i_sim) {  // Looping over sim
    Trackster const& simTs = *simToRecoMap.getRefFirst(i_sim);

    // Ignore sim objects that are not of gen pdgId of interest
    const double id_probability_value =
        std::transform_reduce(particleTypes_signal_.begin(),
                              particleTypes_signal_.end(),
                              0.,
                              std::plus<>{},
                              [&simTs](Trackster::ParticleType partType) { return simTs.id_probability(partType); });
    if (id_probability_value < 1.)
      continue;

    // Finding the supercluster seed, as the trackster with the largest shared energy with the SimTrackster
    auto best_reco_trackster_it = std::ranges::max_element(
        simToRecoMap[i_sim],
        [](TracksterToTracksterMap::AssociationElementType const& a,
           TracksterToTracksterMap::AssociationElementType const& b) { return a.sharedEnergy() < b.sharedEnergy(); });

    // Loop over all reco tracksters having some energy shared with the sim object
    for (auto simToRecoAssoc : simToRecoMap[i_sim]) {
      if (ignoreSuperclusterSeed_ && simToRecoAssoc.index() == best_reco_trackster_it->index()) {
        mask_res[simToRecoAssoc.index()] = 3;  // mark as failing but with special value
        continue;
      }

      if (simToRecoAssoc.score() == 1.)
        continue;  // cut on simToReco as early exit (if sim->reco score is 1, no need to find reco->sim score which will also be 1)

      // retrieve reco->sim score
      auto recoToSimAssoc = std::ranges::find_if(
          recoToSimMap[simToRecoAssoc.index()],
          [i_sim](const TracksterToTracksterMap::AssociationElementType& e) { return e.index() == i_sim; });

      assert(recoToSimAssoc != recoToSimMap[simToRecoAssoc.index()].end());

      if (recoToSimAssoc->score() > recoToSimScoreCut_)
        mask_res[simToRecoAssoc.index()] = 2;  // 2 means failing (evaluates to false)
      else
        mask_res[simToRecoAssoc.index()] = 0;  // passes all selections
    }
  }

  evt.emplace(mask_output_token_, std::move(mask_res));
}
