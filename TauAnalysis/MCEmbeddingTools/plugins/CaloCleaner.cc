#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleaner.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"

typedef CaloCleaner<EcalRecHit> EcalRecHitColCleaner;
typedef CaloCleaner<HBHERecHit> HBHERecHitColCleaner;
typedef CaloCleaner<HFRecHit> HFRecHitColCleaner;
typedef CaloCleaner<HORecHit> HORecHitColCleaner;
typedef CaloCleaner<CastorRecHit> CastorRecHitColCleaner;
typedef CaloCleaner<ZDCRecHit> ZDCRecHitColCleaner;

template <typename T>
CaloCleaner<T>::CaloCleaner(const edm::ParameterSet &iConfig)
    : mu_input_(consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("MuonCollection"))),
      propagatorToken_(esConsumes(edm::ESInputTag("", "SteppingHelixPropagatorAny"))) {
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag>>("oldCollection");
  for (const auto &inCollection : inCollections) {
    inputs_[inCollection.instance()] = consumes<RecHitCollection>(inCollection);
    produces<RecHitCollection>(inCollection.instance());
  }

  is_preshower_ = iConfig.getUntrackedParameter<bool>("is_preshower", false);
  edm::ParameterSet parameters = iConfig.getParameter<edm::ParameterSet>("TrackAssociatorParameters");
  edm::ConsumesCollector iC = consumesCollector();
  parameters_.loadParameters(parameters, iC);
  // trackAssociator_.useDefaultPropagator();
}

template <typename T>
CaloCleaner<T>::~CaloCleaner() {
  // nothing to be done yet...
}

template <typename T>
void CaloCleaner<T>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  auto const &propagator = iSetup.getData(propagatorToken_);
  trackAssociator_.setPropagator(&propagator);

  edm::Handle<edm::View<pat::Muon>> muonHandle;
  iEvent.getByToken(mu_input_, muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;

  std::map<uint32_t, float> correction_map;

  // Fill the correction map
  for (edm::View<pat::Muon>::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
    // get the basic informaiton like fill reco mouon does
    //     RecoMuon/MuonIdentification/plugins/MuonIdProducer.cc
    const reco::Track *track = nullptr;
    if (iMuon->track().isNonnull())
      track = iMuon->track().get();
    else if (iMuon->standAloneMuon().isNonnull())
      track = iMuon->standAloneMuon().get();
    else
      throw cms::Exception("FatalError")
          << "Failed to fill muon id information for a muon with undefined references to tracks";
    TrackDetMatchInfo info =
        trackAssociator_.associate(iEvent, iSetup, *track, parameters_, TrackDetectorAssociator::Any);
    fill_correction_map(&info, &correction_map);
  }

  // Copy the old collection and correct if necessary
  for (auto input_ : inputs_) {
    std::unique_ptr<RecHitCollection> recHitCollection_output(new RecHitCollection());
    edm::Handle<RecHitCollection> recHitCollection;
    // iEvent.getByToken(input_.second[0], recHitCollection);
    iEvent.getByToken(input_.second, recHitCollection);
    for (typename RecHitCollection::const_iterator recHit = recHitCollection->begin();
         recHit != recHitCollection->end();
         ++recHit) {
      if (correction_map[recHit->detid().rawId()] > 0) {
        float new_energy = recHit->energy() - correction_map[recHit->detid().rawId()];
        if (new_energy <= 0)
          continue;  // Do not save empty Hits
        T newRecHit(*recHit);
        newRecHit.setEnergy(new_energy);
        recHitCollection_output->push_back(newRecHit);
      } else {
        recHitCollection_output->push_back(*recHit);
      }
      /* For the inveted collection
           if (correction_map[recHit->detid().rawId()] > 0){
          float new_energy =   correction_map[recHit->detid().rawId()];
          if (new_energy < 0) new_energy =0;
          T newRecHit(*recHit);
          newRecHit.setEnergy(new_energy);
          recHitCollection_output->push_back(newRecHit);
            }*/
    }
    // Save the new collection
    recHitCollection_output->sort();
    iEvent.put(std::move(recHitCollection_output), input_.first);
  }
}

//-------------------------------------------------------------------------------
// define 'buildRecHit' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T>
void CaloCleaner<T>::fill_correction_map(TrackDetMatchInfo *, std::map<uint32_t, float> *) {
  assert(0);  // CV: make sure general function never gets called;
              //     always use template specializations
}

template <>
void CaloCleaner<EcalRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  if (is_preshower_) {
    for (std::vector<DetId>::const_iterator detId = info->crossedPreshowerIds.begin();
         detId != info->crossedPreshowerIds.end();
         ++detId) {
      (*cor_map)[detId->rawId()] = 9999999;  // just remove all energy (Below 0 is not possible)
    }
  } else {
    for (std::vector<const EcalRecHit *>::const_iterator hit = info->crossedEcalRecHits.begin();
         hit != info->crossedEcalRecHits.end();
         hit++) {
      //    (*cor_map) [(*hit)->detid().rawId()] +=(*hit)->energy();
      (*cor_map)[(*hit)->detid().rawId()] = (*hit)->energy();
    }
  }
}

template <>
void CaloCleaner<HBHERecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  for (std::vector<const HBHERecHit *>::const_iterator hit = info->crossedHcalRecHits.begin();
       hit != info->crossedHcalRecHits.end();
       hit++) {
    (*cor_map)[(*hit)->detid().rawId()] = (*hit)->energy();
  }
}

template <>
void CaloCleaner<HORecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  for (std::vector<const HORecHit *>::const_iterator hit = info->crossedHORecHits.begin();
       hit != info->crossedHORecHits.end();
       hit++) {
    (*cor_map)[(*hit)->detid().rawId()] = (*hit)->energy();
  }
}

template <>
void CaloCleaner<HFRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  return;  // No corrections for HF
}

template <>
void CaloCleaner<CastorRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  return;  // No corrections for Castor
}

template <>
void CaloCleaner<ZDCRecHit>::fill_correction_map(TrackDetMatchInfo *info, std::map<uint32_t, float> *cor_map) {
  return;  // No corrections for Castor
}

DEFINE_FWK_MODULE(EcalRecHitColCleaner);
DEFINE_FWK_MODULE(HBHERecHitColCleaner);
DEFINE_FWK_MODULE(HORecHitColCleaner);
// no  need for cleaning outside of tracker, so just a copy of the old collection
DEFINE_FWK_MODULE(HFRecHitColCleaner);
DEFINE_FWK_MODULE(CastorRecHitColCleaner);
DEFINE_FWK_MODULE(ZDCRecHitColCleaner);
