#include "TauAnalysis/MCEmbeddingTools/plugins/TrackerCleaner.h"

#include "DataFormats/TrackerRecHit2D/interface/ProjectedSiStripRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

typedef TrackerCleaner<SiPixelCluster> PixelColCleaner;
typedef TrackerCleaner<SiStripCluster> StripColCleaner;

template <typename T>
TrackerCleaner<T>::TrackerCleaner(const edm::ParameterSet &iConfig)
    : mu_input_(consumes<edm::View<pat::Muon>>(iConfig.getParameter<edm::InputTag>("MuonCollection")))

{
  std::vector<edm::InputTag> inCollections = iConfig.getParameter<std::vector<edm::InputTag>>("oldCollection");
  for (const auto &inCollection : inCollections) {
    inputs_[inCollection.instance()] = consumes<TrackClusterCollection>(inCollection);
    produces<TrackClusterCollection>(inCollection.instance());
  }
}

template <typename T>
TrackerCleaner<T>::~TrackerCleaner() {
  // nothing to be done yet...
}

template <typename T>
void TrackerCleaner<T>::produce(edm::Event &iEvent, const edm::EventSetup &iSetup) {
  using namespace edm;

  edm::Handle<edm::View<pat::Muon>> muonHandle;
  iEvent.getByToken(mu_input_, muonHandle);
  edm::View<pat::Muon> muons = *muonHandle;

  for (auto input_ : inputs_) {
    edm::Handle<TrackClusterCollection> inputClusters;
    iEvent.getByToken(input_.second, inputClusters);

    std::vector<bool> vetodClusters;

    vetodClusters.resize(inputClusters->dataSize(), false);

    for (edm::View<pat::Muon>::const_iterator iMuon = muons.begin(); iMuon != muons.end(); ++iMuon) {
      if (!iMuon->isGlobalMuon())
        continue;
      const reco::Track *mutrack = iMuon->globalTrack().get();
      //  reco::Track *mutrack = new reco::Track(*(iMuon->innerTrack() ));
      for (trackingRecHit_iterator hitIt = mutrack->recHitsBegin(); hitIt != mutrack->recHitsEnd(); ++hitIt) {
        const TrackingRecHit &murechit = **hitIt;
        if (!(murechit).isValid())
          continue;

        if (match_rechit_type(murechit)) {
          auto &thit = reinterpret_cast<BaseTrackerRecHit const &>(murechit);
          auto const &cluster = thit.firstClusterRef();
          vetodClusters[cluster.key()] = true;
        }
        auto &thit = reinterpret_cast<BaseTrackerRecHit const &>(murechit);
        if (trackerHitRTTI::isMatched(thit)) {
          vetodClusters[reinterpret_cast<SiStripMatchedRecHit2D const &>(murechit).stereoClusterRef().key()] = true;
        }
      }
    }
    std::unique_ptr<TrackClusterCollection> output(new TrackClusterCollection());

    int idx = 0;
    for (typename TrackClusterCollection::const_iterator clustSet = inputClusters->begin();
         clustSet != inputClusters->end();
         ++clustSet) {
      DetId detIdObject(clustSet->detId());
      typename TrackClusterCollection::FastFiller spc(*output, detIdObject);
      for (typename edmNew::DetSet<T>::const_iterator clustIt = clustSet->begin(); clustIt != clustSet->end();
           ++clustIt) {
        idx++;
        if (vetodClusters[idx - 1])
          continue;
        // if (!vetodClusters[idx-1]) continue; for inverted selction
        spc.push_back(*clustIt);
      }
    }
    iEvent.put(std::move(output), input_.first);
  }
}

//-------------------------------------------------------------------------------
// define 'buildRecHit' functions used for different types of recHits
//-------------------------------------------------------------------------------

template <typename T>
bool TrackerCleaner<T>::match_rechit_type(const TrackingRecHit &murechit) {
  assert(0);  // CV: make sure general function never gets called;
              //     always use template specializations
  return false;
}

template <>
bool TrackerCleaner<SiStripCluster>::match_rechit_type(const TrackingRecHit &murechit) {
  const std::type_info &hit_type = typeid(murechit);
  if (hit_type == typeid(SiStripRecHit2D))
    return true;
  else if (hit_type == typeid(SiStripRecHit1D))
    return true;
  else if (hit_type == typeid(SiStripMatchedRecHit2D))
    return true;
  else if (hit_type == typeid(ProjectedSiStripRecHit2D))
    return true;

  return false;
}

template <>
bool TrackerCleaner<SiPixelCluster>::match_rechit_type(const TrackingRecHit &murechit) {
  const std::type_info &hit_type = typeid(murechit);
  if (hit_type == typeid(SiPixelRecHit))
    return true;

  return false;
}

DEFINE_FWK_MODULE(PixelColCleaner);
DEFINE_FWK_MODULE(StripColCleaner);
