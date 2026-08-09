/**
 * Selects a subset of tracksters based on their association score to simulation, to compute efficiencies and fake rates.
 * Builds vector<bool> as a mask associated to a trackster collection, for "signal" (reco2Sim < threshold) and "fake" (reco2Sim > threshold)
 * Author : Theo Cuisset (LLR)
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

class TracksterAssociationMaskProducer : public edm::stream::EDProducer<> {
public:
  explicit TracksterAssociationMaskProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  const edm::EDGetTokenT<TracksterCollection> tracksters_token_;

  const edm::EDGetTokenT<TracksterToTracksterMap> simToRecoAssociator_;
  const edm::EDGetTokenT<TracksterToTracksterMap> recoToSimAssociator_;

  const edm::EDPutTokenT<vector<int>> mask_output_token_, mask_fake_output_token_;

  float recoToSimScoreCut_;
  float recoToSimScoreCut_forFakes_;
  vector<Trackster::ParticleType> particleTypes_signal_;
};

DEFINE_FWK_MODULE(TracksterAssociationMaskProducer);

TracksterAssociationMaskProducer::TracksterAssociationMaskProducer(const edm::ParameterSet& ps)
    : tracksters_token_(consumes<TracksterCollection>(ps.getParameter<edm::InputTag>("tracksters"))),
      simToRecoAssociator_(consumes<TracksterToTracksterMap>(ps.getParameter<edm::InputTag>("associatorSimToReco"))),
      recoToSimAssociator_(consumes<TracksterToTracksterMap>(ps.getParameter<edm::InputTag>("associatorRecoToSim"))),
      mask_output_token_(produces<vector<int>>()),
      mask_fake_output_token_(produces<vector<int>>("fakes")),
      recoToSimScoreCut_(ps.getParameter<double>("recoToSimScoreCut")),
      recoToSimScoreCut_forFakes_(ps.getParameter<double>("recoToSimScoreCut_forFakes")),
      particleTypes_signal_() {
  vector<int> v = ps.getParameter<vector<int>>("particleTypesSignal");
  particleTypes_signal_.reserve(v.size());
  for (auto x : v)
    particleTypes_signal_.push_back(static_cast<Trackster::ParticleType>(x));
}

void TracksterAssociationMaskProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("tracksters")->setComment("Input tag for tracksters");

  desc.add<edm::InputTag>("associatorRecoToSim")->setComment("Input tag for the RecoToSim associator");
  desc.add<edm::InputTag>("associatorSimToReco")->setComment("Input tag for the SimToReco associator");

  desc.add<double>("recoToSimScoreCut")
      ->setComment("Cut on reco2Sim score to consider a trackster 'signal' (if reco2Sim < cut)");
  desc.add<double>("recoToSimScoreCut_forFakes")
      ->setComment(
          "Cut on reco2Sim score to consider a trackster as 'non-signal' (if reco2Sim > cut). Should be greater than "
          "recoToSimScoreCut to make sense");
  desc.add<vector<int>>("particleTypesSignal")
      ->setComment("ticl::Trackster::ParticleType list to filter on SimTrackster (ie electron is 1, photon 0)");

  descriptions.addWithDefaultLabel(desc);
}

void TracksterAssociationMaskProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  TracksterCollection const& tracksters = evt.get(tracksters_token_);
  TracksterToTracksterMap const& simToRecoMap = evt.get(simToRecoAssociator_);
  TracksterToTracksterMap const& recoToSimMap = evt.get(recoToSimAssociator_);

  vector<int> mask_res(tracksters.size(), 1);
  for (size_t i_sim = 0; i_sim < simToRecoMap.size(); ++i_sim) {  // Looping over sim
    Trackster const& simTs = *simToRecoMap.getRefFirst(i_sim);
    const double id_probability_value =
        std::transform_reduce(particleTypes_signal_.begin(),
                              particleTypes_signal_.end(),
                              0.,
                              std::plus<>{},
                              [&simTs](Trackster::ParticleType partType) { return simTs.id_probability(partType); });
    if (id_probability_value < 1.)
      continue;

    auto best_reco_trackster_it = std::ranges::max_element(
        simToRecoMap[i_sim],
        [](TracksterToTracksterMap::AssociationElementType const& a,
           TracksterToTracksterMap::AssociationElementType const& b) { return a.sharedEnergy() < b.sharedEnergy(); });

    if (best_reco_trackster_it == simToRecoMap[i_sim].end()) {
      continue;  // SimTrackster having no association to reco
    }

    // We check the reco-sim score for the same pair
    auto recoToSimAssoc = std::ranges::find_if(
        recoToSimMap[best_reco_trackster_it->index()],
        [i_sim](const TracksterToTracksterMap::AssociationElementType& e) { return e.index() == i_sim; });
    assert(recoToSimAssoc != recoToSimMap[best_reco_trackster_it->index()].end());
    if (recoToSimAssoc->score() > recoToSimScoreCut_)
      mask_res[best_reco_trackster_it->index()] = 2;  // 2 means failing (evaluates to false)
    else
      mask_res[best_reco_trackster_it->index()] = 0;  // passes all selections
  }

  evt.emplace(mask_output_token_, std::move(mask_res));

  // Building the "fake" mask, ie tracksters that are not matched to any signal
  // essentially an "inverted" mask_res, but with potentially different cuts (to avoid in-between area of "partially-gen-matched" tracksters)
  vector<int> mask_fake_res(tracksters.size(), 1);

  for (size_t i_reco = 0; i_reco < recoToSimMap.size(); ++i_reco) {  // Looping over reco
    bool isPotentiallySignalMatched = false;
    // Listing all simTracksters matched to reco, looking if there is one well-matched to a speciifc PID
    for (auto assoc : recoToSimMap[i_reco]) {
      if (assoc.score() > recoToSimScoreCut_forFakes_)
        continue;  // ignore bad association scores

      Trackster const& simTs = *recoToSimMap.getRefSecond(assoc.index());
      const double id_probability_value =
          std::transform_reduce(particleTypes_signal_.begin(),
                                particleTypes_signal_.end(),
                                0.,
                                std::plus<>{},
                                [&simTs](Trackster::ParticleType partType) { return simTs.id_probability(partType); });
      if (id_probability_value > 0.) {
        // We have a trackster with good reco2sim score with a simTrackster that has a PID from signal
        // do not include that trackster as a fake
        isPotentiallySignalMatched = true;
        break;
      }
    }
    if (isPotentiallySignalMatched)
      continue;

    mask_fake_res[i_reco] = 0;  // This is a potential fake
  }

  evt.emplace(mask_fake_output_token_, std::move(mask_fake_res));
}
