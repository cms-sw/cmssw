#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"

//
// class declaration
//
class UpdatedMuonInnerTrackRef : public edm::global::EDProducer<> {
public:
  explicit UpdatedMuonInnerTrackRef(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  reco::TrackRef findNewRef(reco::TrackRef const& oldTrackRef,
                            edm::Handle<reco::TrackCollection> const& newTrackCollection) const;

  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;
  edm::EDGetTokenT<reco::TrackCollection> oldTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection> newTrackToken_;

  double maxInvPtDiff;
  double minDR;
};

/////////////////////////////////////////////////////////////////////////////////////
UpdatedMuonInnerTrackRef::UpdatedMuonInnerTrackRef(const edm::ParameterSet& pset) {
  // What is being produced
  produces<std::vector<reco::Muon> >();

  // Input products
  muonToken_ =
      consumes<edm::View<reco::Muon> >(pset.getUntrackedParameter<edm::InputTag>("MuonTag", edm::InputTag("muons")));
  oldTrackToken_ = consumes<reco::TrackCollection>(
      pset.getUntrackedParameter<edm::InputTag>("OldTrackTag", edm::InputTag("generalTracks")));
  newTrackToken_ = consumes<reco::TrackCollection>(
      pset.getUntrackedParameter<edm::InputTag>("NewTrackTag", edm::InputTag("generalTracksSkim")));

  // matching criteria products
  maxInvPtDiff = pset.getUntrackedParameter<double>("maxInvPtDiff", 0.005);
  minDR = pset.getUntrackedParameter<double>("minDR", 0.1);
}

/////////////////////////////////////////////////////////////////////////////////////
void UpdatedMuonInnerTrackRef::produce(edm::StreamID, edm::Event& ev, const edm::EventSetup& iSetup) const {
  // Muon collection
  edm::Handle<edm::View<reco::Muon> > muonCollectionHandle;
  if (!ev.getByToken(muonToken_, muonCollectionHandle)) {
    edm::LogError("") << ">>> Muon collection does not exist !!!";
    return;
  }

  edm::Handle<reco::TrackCollection> oldTrackCollection;
  if (!ev.getByToken(oldTrackToken_, oldTrackCollection)) {
    edm::LogError("") << ">>> Old Track collection does not exist !!!";
    return;
  }

  edm::Handle<reco::TrackCollection> newTrackCollection;
  if (!ev.getByToken(newTrackToken_, newTrackCollection)) {
    edm::LogError("") << ">>> New Track collection does not exist !!!";
    return;
  }

  unsigned int muonCollectionSize = muonCollectionHandle->size();
  std::unique_ptr<reco::MuonCollection> newmuons(new reco::MuonCollection);

  for (unsigned int i = 0; i < muonCollectionSize; i++) {
    edm::RefToBase<reco::Muon> mu = muonCollectionHandle->refAt(i);
    std::unique_ptr<reco::Muon> newmu{mu->clone()};

    if (mu->innerTrack().isNonnull()) {
      reco::TrackRef newTrackRef = findNewRef(mu->innerTrack(), newTrackCollection);
      /*               printf(" %6.2f %+6.2f %+6.2f --> ",mu->innerTrack()->pt (), mu->innerTrack()->eta(), mu->innerTrack()->phi());
               if(newTrackRef.isNonnull()){
                  printf(" %6.2f %+6.2f %+6.2f\n",newTrackRef->pt (), newTrackRef->eta(), newTrackRef->phi());
               }else{
                  printf("\n");
               }
*/
      newmu->setInnerTrack(newTrackRef);
    }

    newmuons->push_back(*newmu);
  }

  ev.put(std::move(newmuons));
}

reco::TrackRef UpdatedMuonInnerTrackRef::findNewRef(
    reco::TrackRef const& oldTrackRef, edm::Handle<reco::TrackCollection> const& newTrackCollection) const {
  float dRMin = 1000;
  int found = -1;
  for (unsigned int i = 0; i < newTrackCollection->size(); i++) {
    reco::TrackRef newTrackRef = reco::TrackRef(newTrackCollection, i);
    if (newTrackRef.isNull())
      continue;

    if (fabs((1.0 / newTrackRef->pt()) - (1.0 / oldTrackRef->pt())) > maxInvPtDiff)
      continue;
    float dR = deltaR(newTrackRef->momentum(), oldTrackRef->momentum());
    if (dR <= minDR && dR < dRMin) {
      dRMin = dR;
      found = i;
    }
  }

  if (found >= 0) {
    return reco::TrackRef(newTrackCollection, found);
  } else {
    return reco::TrackRef();
  }
}

DEFINE_FWK_MODULE(UpdatedMuonInnerTrackRef);
