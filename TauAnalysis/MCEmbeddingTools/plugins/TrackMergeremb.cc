#include "TauAnalysis/MCEmbeddingTools/plugins/TrackMergeremb.h"

#include <memory>

#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"

#include "DataFormats/EgammaTrackReco/interface/ConversionTrack.h"
#include "DataFormats/EgammaTrackReco/interface/ConversionTrackFwd.h"

#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"

#include "DataFormats/MuonReco/interface/MuonQuality.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "RecoMuon/MuonIdentification/interface/MuonTimingFiller.h"
#include "DataFormats/MuonReco/interface/MuonCocktails.h"
#include "DataFormats/MuonReco/interface/MuonTime.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtra.h"
#include "DataFormats/MuonReco/interface/MuonTimeExtraMap.h"
#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/MuonReco/interface/CaloMuon.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/ParticleFlowReco/interface/PFRecTrackFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecTrack.h"

typedef TrackMergeremb<reco::TrackCollection> TrackColMerger;
typedef TrackMergeremb<reco::GsfTrackCollection> GsfTrackColMerger;
typedef TrackMergeremb<reco::MuonCollection> MuonColMerger;
typedef TrackMergeremb<reco::GsfElectronCollection> GsfElectronColMerger;
typedef TrackMergeremb<reco::PhotonCollection> PhotonColMerger;
typedef TrackMergeremb<reco::ConversionCollection> ConversionColMerger;
typedef TrackMergeremb<reco::PFCandidateCollection> PFColMerger;

// Here some overloaded functions, which are needed such that the right merger function is called for the indivudal Collections
template <typename T1>
void TrackMergeremb<T1>::willproduce(std::string instance, std::string alias) {
  produces<TrackCollectionemb>(instance);
}

template <typename T1>
void TrackMergeremb<T1>::willconsume(const edm::ParameterSet& iConfig) {}

template <typename T1>
void TrackMergeremb<T1>::merg_and_put(edm::Event& iEvent,
                                      std::string instance,
                                      std::vector<edm::EDGetTokenT<TrackCollectionemb> >& to_merge) {
  std::unique_ptr<TrackCollectionemb> outTracks = std::unique_ptr<TrackCollectionemb>(new TrackCollectionemb);

  for (auto akt_collection : to_merge) {
    edm::Handle<TrackCollectionemb> track_col_in;
    iEvent.getByToken(akt_collection, track_col_in);

    size_t sedref_it = 0;
    for (typename TrackCollectionemb::const_iterator it = track_col_in->begin(); it != track_col_in->end();
         ++it, ++sedref_it) {
      outTracks->push_back(typename T1::value_type(*it));
    }

  }  // end merge

  iEvent.put(std::move(outTracks), instance);
}

template <>
void TrackMergeremb<reco::TrackCollection>::willproduce(std::string instance, std::string alias) {
  produces<reco::TrackCollection>(instance).setBranchAlias(alias + "Tracks");
  produces<reco::TrackExtraCollection>(instance).setBranchAlias(alias + "TrackExtras");
  produces<TrackingRecHitCollection>(instance).setBranchAlias(alias + "RecHits");
  produces<TrackToTrackMapnew>();
}

template <>
void TrackMergeremb<reco::TrackCollection>::merg_and_put(
    edm::Event& iEvent, std::string instance, std::vector<edm::EDGetTokenT<reco::TrackCollection> >& to_merge) {
  std::unique_ptr<reco::TrackCollection> outTracks = std::make_unique<reco::TrackCollection>();
  std::unique_ptr<reco::TrackExtraCollection> outTracks_ex = std::make_unique<reco::TrackExtraCollection>();
  std::unique_ptr<TrackingRecHitCollection> outTracks_rh = std::make_unique<TrackingRecHitCollection>();
  std::unique_ptr<TrackToTrackMapnew> outTracks_refs = std::make_unique<TrackToTrackMapnew>();

  auto rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  // auto rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();

  std::vector<reco::TrackRefVector> trackRefColl;
  //std::vector<reco::TrackRef> trackRefColl;

  for (auto akt_collection : to_merge) {
    edm::Handle<reco::TrackCollection> track_col_in;
    iEvent.getByToken(akt_collection, track_col_in);

    unsigned sedref_it = 0;
    for (reco::TrackCollection::const_iterator it = track_col_in->begin(); it != track_col_in->end();
         ++it, ++sedref_it) {
      outTracks->push_back(reco::Track(*it));
      auto rechits = it->recHits();
      // Fixing geometry records of detector components for tracking rec. hits
      for (auto ith = rechits.begin(); ith!= rechits.end(); ith++){
        auto hit = *(&(*ith));
        hit->setDet(*geometry_->idToDet(hit->rawId()));
      }
      outTracks_ex->push_back(reco::TrackExtra(*it->extra()));
      outTracks->back().setExtra(reco::TrackExtraRef(rTrackExtras, outTracks_ex->size() - 1));
      reco::TrackRef trackRefold(track_col_in, sedref_it);

      reco::TrackRefVector trackRefColl_helpvec;
      trackRefColl_helpvec.push_back(trackRefold);
      trackRefColl.push_back(trackRefColl_helpvec);
    }

  }  // end merge

  edm::OrphanHandle<reco::TrackCollection> trackHandle = iEvent.put(std::move(outTracks), instance);
  iEvent.put(std::move(outTracks_ex), instance);

  TrackToTrackMapnew::Filler filler(*outTracks_refs);
  filler.insert(trackHandle, trackRefColl.begin(), trackRefColl.end());
  filler.fill();

  iEvent.put(std::move(outTracks_refs));
  iEvent.put(std::move(outTracks_rh), instance);  // not implemented so far
}

template <>
void  TrackMergeremb<reco::GsfTrackCollection>::willconsume(const edm::ParameterSet& iConfig) {
  //track refs for trackerdriven seeds
  inputs_fixtrackrefs_ = consumes<TrackToTrackMapnew >( edm::InputTag("generalTracks") );
  inputs_fixtrackcol_ = consumes<reco::TrackCollection >( edm::InputTag("generalTracks") );
  inputs_rElectronMergedSeeds_ = consumes<reco::ElectronSeedCollection >(edm::InputTag("electronMergedSeeds") );
  inputs_rElectronMergedSeedViews_ = consumes<edm::View<TrajectorySeed>>(edm::InputTag("electronMergedSeeds") );
}

template <>
void TrackMergeremb<reco::GsfTrackCollection>::willproduce(std::string instance, std::string alias) {
  produces<reco::GsfTrackCollection>(instance).setBranchAlias(alias + "GsfTracks");
  produces<reco::TrackExtraCollection>(instance).setBranchAlias(alias + "TrackExtras");
  produces<reco::GsfTrackExtraCollection>(instance).setBranchAlias(alias + "GsfTrackExtras");
  produces<TrackingRecHitCollection>(instance).setBranchAlias(alias + "RecHits");
  produces<GsfTrackToTrackMapnew>();
}

template <>
void TrackMergeremb<reco::GsfTrackCollection>::merg_and_put(
    edm::Event& iEvent, std::string instance, std::vector<edm::EDGetTokenT<reco::GsfTrackCollection> >& to_merge) {
  std::unique_ptr<reco::GsfTrackCollection> outTracks = std::make_unique<reco::GsfTrackCollection>();
  std::unique_ptr<reco::TrackExtraCollection> outTracks_ex = std::make_unique<reco::TrackExtraCollection>();
  std::unique_ptr<reco::GsfTrackExtraCollection> outTracks_exgsf = std::make_unique<reco::GsfTrackExtraCollection>();
  std::unique_ptr<TrackingRecHitCollection> outTracks_rh = std::make_unique<TrackingRecHitCollection>();
  std::unique_ptr<GsfTrackToTrackMapnew> outTracks_refs = std::unique_ptr<GsfTrackToTrackMapnew>(new GsfTrackToTrackMapnew());

  auto rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
  auto rTrackExtras_gsf = iEvent.getRefBeforePut<reco::GsfTrackExtraCollection>();

  auto rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
  std::vector<reco::GsfTrackRefVector> trackRefColl;

  for (auto akt_collection : to_merge) {
    edm::Handle<reco::GsfTrackCollection> track_col_in;
    iEvent.getByToken(akt_collection, track_col_in);

    size_t sedref_it = 0;
    for (reco::GsfTrackCollection::const_iterator it = track_col_in->begin(); it != track_col_in->end();
         ++it, ++sedref_it) {
      outTracks->push_back(reco::GsfTrack(*it));
      outTracks_ex->push_back(reco::TrackExtra(*it->extra()));
      outTracks_exgsf->push_back(reco::GsfTrackExtra(*it->gsfExtra()));

      outTracks->back().setExtra(reco::TrackExtraRef(rTrackExtras, outTracks_ex->size() - 1));
      outTracks->back().setGsfExtra(reco::GsfTrackExtraRef(rTrackExtras_gsf, outTracks_exgsf->size() - 1));

      reco::GsfTrackRef trackRefold(track_col_in, sedref_it);
      reco::GsfTrackRefVector trackRefColl_helpvec;
      trackRefColl_helpvec.push_back(trackRefold);
      trackRefColl.push_back(trackRefColl_helpvec);
    }

  }  // end merge

  edm::OrphanHandle<reco::GsfTrackCollection> trackHandle = iEvent.put(std::move(outTracks),instance);
  GsfTrackToTrackMapnew::Filler filler(*outTracks_refs);
  filler.insert(trackHandle, trackRefColl.begin(), trackRefColl.end() );
  filler.fill();


  //track to track map for trackerdriven seed fix
  edm::Handle<TrackToTrackMapnew> track_ref_map;
  iEvent.getByToken(inputs_fixtrackrefs_, track_ref_map);

  edm::Handle<reco::TrackCollection> track_new_col;
  iEvent.getByToken(inputs_fixtrackcol_, track_new_col);
  std::map<reco::TrackRef , reco::TrackRef > simple_track_to_track_map; //I didn't find a more elegant way, so just build a good old fassion std::map
  for (unsigned abc =0;  abc < track_new_col->size();  ++abc) {
    reco::TrackRef trackRef(track_new_col, abc);
    simple_track_to_track_map[((*track_ref_map)[trackRef])[0]] = trackRef;
  }

  edm::Handle<reco::ElectronSeedCollection> elSeeds;
  iEvent.getByToken(inputs_rElectronMergedSeeds_,elSeeds);
  auto bElSeeds = elSeeds->cbegin();
  auto eElSeeds = elSeeds->cend();

  edm::Handle<edm::View<TrajectorySeed> > seedViewsHandle;
  iEvent.getByToken(inputs_rElectronMergedSeedViews_,seedViewsHandle);

  std::unique_ptr<reco::TrackExtraCollection> outTracks_ex_new = std::unique_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());

  //fix track extras to new merged seeds
  for ( typename reco::TrackExtraCollection::const_iterator tex = outTracks_ex->begin(); tex!= outTracks_ex->end(); ++tex ) {
    reco::TrackExtra newTrackExtra(*tex);

    if (tex->seedRef().isAvailable()) {
      const reco::ElectronSeedRef & trSeedRef (tex->seedRef().castTo<reco::ElectronSeedRef>());
      if( trSeedRef.isAvailable() ) {

        //find new seed with corresponding supercluster or ctfTrack
        //note that seed can be both Ecal- and Tracker-driven
        size_t sedref_it = 0;
        for( auto tseed = bElSeeds; tseed != eElSeeds; ++tseed, ++sedref_it) {
          const reco::ElectronSeedRef elSeedRef (elSeeds, sedref_it);

          if (trSeedRef->isEcalDriven() && elSeedRef->isEcalDriven()) {
            //match seeds by pair of detIds
            if ( trSeedRef->detId(0) == elSeedRef->detId(0) && trSeedRef->detId(1) == elSeedRef->detId(1)) {
              edm::RefToBase<TrajectorySeed> traSeedRef(seedViewsHandle, sedref_it);
              newTrackExtra.setSeedRef(traSeedRef);
            }
          }
          if (trSeedRef->isTrackerDriven() && elSeedRef->isTrackerDriven()) {
            //if tracker driven, check for same ctf track
            if (simple_track_to_track_map[trSeedRef->ctfTrack()] == elSeedRef->ctfTrack()) {
              edm::RefToBase<TrajectorySeed> traSeedRef(seedViewsHandle, sedref_it);
              newTrackExtra.setSeedRef(traSeedRef);
            }
          }
        }
      }
    }

    outTracks_ex_new->push_back(newTrackExtra);
  }

  iEvent.put(std::move(outTracks_refs));
  iEvent.put(std::move(outTracks_ex_new),instance);
  iEvent.put(std::move(outTracks_exgsf), instance);
  iEvent.put(std::move(outTracks_rh), instance);
}

template <>
void TrackMergeremb<reco::MuonCollection>::willproduce(std::string instance, std::string alias) {
  produces<reco::MuonCollection>();
  produces<reco::CaloMuonCollection>();
  produces<reco::MuonTimeExtraMap>("combined");
  produces<reco::MuonTimeExtraMap>("dt");
  produces<reco::MuonTimeExtraMap>("csc");

  // todo make this configurable (or not )
  produces<reco::IsoDepositMap>("tracker");
  produces<reco::IsoDepositMap>("ecal");
  produces<reco::IsoDepositMap>("hcal");
  produces<reco::IsoDepositMap>("ho");
  produces<reco::IsoDepositMap>("jets");

  produces<reco::MuonToMuonMap>();
}

template <>
void TrackMergeremb<reco::MuonCollection>::willconsume(const edm::ParameterSet& iConfig) {
  inputs_fixtrackrefs_ = consumes<TrackToTrackMapnew>(edm::InputTag("generalTracks"));
  inputs_fixtrackcol_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
}

template <>
void TrackMergeremb<reco::MuonCollection>::merg_and_put(
    edm::Event& iEvent, std::string instance, std::vector<edm::EDGetTokenT<reco::MuonCollection> >& to_merge) {
  std::unique_ptr<reco::MuonCollection> outTracks = std::make_unique<reco::MuonCollection>();
  std::unique_ptr<reco::CaloMuonCollection> calomu =
      std::make_unique<reco::CaloMuonCollection>();  //not implemented so far

  edm::Handle<TrackToTrackMapnew> track_ref_map;
  iEvent.getByToken(inputs_fixtrackrefs_, track_ref_map);

  edm::Handle<reco::TrackCollection> track_new_col;
  iEvent.getByToken(inputs_fixtrackcol_, track_new_col);
  std::map<reco::TrackRef, reco::TrackRef>
      simple_track_to_track_map;  //I didn't find a more elegant way, so just build a good old fassion std::map
  for (unsigned abc = 0; abc < track_new_col->size(); ++abc) {
    reco::TrackRef trackRef(track_new_col, abc);
    simple_track_to_track_map[((*track_ref_map)[trackRef])[0]] = trackRef;
  }

  std::vector<reco::MuonRef> muonRefColl;
  reco::MuonRefProd outputMuonsRefProd = iEvent.getRefBeforePut<reco::MuonCollection>();
  unsigned new_idx = 0;
  for (auto akt_collection : to_merge) {
    edm::Handle<reco::MuonCollection> track_col_in;
    iEvent.getByToken(akt_collection, track_col_in);
    unsigned old_idx = 0;
    for (reco::MuonCollection::const_iterator it = track_col_in->begin(); it != track_col_in->end();
         ++it, ++old_idx, ++new_idx) {
      outTracks->push_back(reco::Muon(*it));
      reco::MuonRef muRefold(track_col_in, old_idx);
      muonRefColl.push_back(muRefold);
      reco::MuonRef muRefnew(outputMuonsRefProd, new_idx);

      if (it->track().isNonnull()) {
        //std::cout<<"pfmerge tr: "<<it->trackRef().id()<< " "<< it->trackRef().key()<< " " << simple_track_to_track_map[it->trackRef()].id() <<  " " << simple_track_to_track_map[it->trackRef()].key() <<std::endl;
        outTracks->back().setTrack(simple_track_to_track_map[it->track()]);
      }
    }

  }  // end merge

  const int nMuons = outTracks->size();

  std::vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
  std::vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
  std::vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);
  std::vector<reco::IsoDeposit> trackDepColl(nMuons);
  std::vector<reco::IsoDeposit> ecalDepColl(nMuons);
  std::vector<reco::IsoDeposit> hcalDepColl(nMuons);
  std::vector<reco::IsoDeposit> hoDepColl(nMuons);
  std::vector<reco::IsoDeposit> jetDepColl(nMuons);

  edm::OrphanHandle<reco::MuonCollection> muonHandle = iEvent.put(std::move(outTracks));

  auto fillMap = [](auto refH, auto& vec, edm::Event& ev, const std::string& cAl = "") {
    typedef edm::ValueMap<typename std::decay<decltype(vec)>::type::value_type> MapType;
    std::unique_ptr<MapType> oMap(new MapType());
    {
      typename MapType::Filler filler(*oMap);
      filler.insert(refH, vec.begin(), vec.end());
      vec.clear();
      filler.fill();
    }
    ev.put(std::move(oMap), cAl);
  };

  fillMap(muonHandle, combinedTimeColl, iEvent, "combined");
  fillMap(muonHandle, dtTimeColl, iEvent, "dt");
  fillMap(muonHandle, cscTimeColl, iEvent, "csc");
  fillMap(muonHandle, trackDepColl, iEvent, "tracker");
  fillMap(muonHandle, ecalDepColl, iEvent, "ecal");
  fillMap(muonHandle, hcalDepColl, iEvent, "hcal");
  fillMap(muonHandle, hoDepColl, iEvent, "ho");
  fillMap(muonHandle, jetDepColl, iEvent, "jets");
  fillMap(muonHandle, muonRefColl, iEvent);
  iEvent.put(std::move(calomu));
}

template <>
void TrackMergeremb<reco::PFCandidateCollection>::willproduce(std::string instance, std::string alias) {
  produces<reco::PFCandidateCollection>(instance);
  // std::cout<<"Produce PF Collection: "<<instance<<std::endl;
}

template <>
void TrackMergeremb<reco::PFCandidateCollection>::willconsume(const edm::ParameterSet& iConfig) {
  inputs_fixtrackrefs_ = consumes<TrackToTrackMapnew>(edm::InputTag("generalTracks"));
  inputs_fixtrackcol_ = consumes<reco::TrackCollection>(edm::InputTag("generalTracks"));
  inputs_fixgsftrackrefs_ = consumes<GsfTrackToTrackMapnew>( edm::InputTag("electronGsfTracks") );
  inputs_fixgsftrackcol_ = consumes<reco::GsfTrackCollection>( edm::InputTag("electronGsfTracks") );
  inputs_fixmurefs_ = consumes<reco::MuonToMuonMap>(edm::InputTag("muons1stStep"));
  inputs_fixmucol_ = consumes<reco::MuonCollection>(edm::InputTag("muons1stStep"));
  inputs_SC_ = consumes<reco::SuperClusterCollection>(edm::InputTag("particleFlowEGamma") );
}

template <>
void TrackMergeremb<reco::PFCandidateCollection>::merg_and_put(
    edm::Event& iEvent, std::string instance, std::vector<edm::EDGetTokenT<reco::PFCandidateCollection> >& to_merge) {
  std::unique_ptr<reco::PFCandidateCollection> outTracks = std::make_unique<reco::PFCandidateCollection>();

  edm::Handle<TrackToTrackMapnew> track_ref_map;
  iEvent.getByToken(inputs_fixtrackrefs_, track_ref_map);

  edm::Handle<reco::TrackCollection> track_new_col;
  iEvent.getByToken(inputs_fixtrackcol_, track_new_col);
  std::map<reco::TrackRef, reco::TrackRef>
      simple_track_to_track_map;  //I didn't find a more elegant way, so just build a good old fassion std::map
  for (unsigned abc = 0; abc < track_new_col->size(); ++abc) {
    reco::TrackRef trackRef(track_new_col, abc);
    simple_track_to_track_map[((*track_ref_map)[trackRef])[0]] = trackRef;
  }

  edm::Handle<GsfTrackToTrackMapnew> gsftrack_ref_map;
  iEvent.getByToken(inputs_fixgsftrackrefs_, gsftrack_ref_map);

  edm::Handle<reco::GsfTrackCollection> gsftrack_new_col;
  iEvent.getByToken(inputs_fixgsftrackcol_, gsftrack_new_col);
  std::map<reco::GsfTrackRef , reco::GsfTrackRef > simple_gsftrack_to_gsftrack_map; //I didn't find a more elegant way, so just build a good old fassion std::map
  for (unsigned abc =0;  abc < gsftrack_new_col->size();  ++abc) {
    reco::GsfTrackRef gsfTrackRef(gsftrack_new_col, abc);
    simple_gsftrack_to_gsftrack_map[((*gsftrack_ref_map)[gsfTrackRef])[0]] = gsfTrackRef;
  }

  edm::Handle<reco::MuonToMuonMap> muon_ref_map;
  iEvent.getByToken(inputs_fixmurefs_, muon_ref_map);

  edm::Handle<reco::MuonCollection> muon_new_col;
  iEvent.getByToken(inputs_fixmucol_, muon_new_col);
  std::map<reco::MuonRef, reco::MuonRef>
      simple_mu_to_mu_map;  //I didn't find a more elegant way, so just build a good old fassion std::map
  for (unsigned abc = 0; abc < muon_new_col->size(); ++abc) {
    reco::MuonRef muRef(muon_new_col, abc);
    simple_mu_to_mu_map[(*muon_ref_map)[muRef]] = muRef;
  }

  //used for photon matching
  edm::Handle<reco::SuperClusterCollection> sCs;
  iEvent.getByToken(inputs_SC_,sCs);
  auto bSc = sCs->cbegin();
  auto eSc = sCs->cend();

  for (auto akt_collection : to_merge) {
    edm::Handle<reco::PFCandidateCollection> track_col_in;
    iEvent.getByToken(akt_collection, track_col_in);
    for (reco::PFCandidateCollection::const_iterator it = track_col_in->begin(); it != track_col_in->end(); ++it) {
      outTracks->push_back(reco::PFCandidate(*it));
      //if (fabs(it->pdgId()) == 13){
      if (it->trackRef().isNonnull() && outTracks->back().charge()) {
        //std::cout<<"pfmerge tr: "<<it->trackRef().id()<< " "<< it->trackRef().key()<< " " << simple_track_to_track_map[it->trackRef()].id() <<  " " << simple_track_to_track_map[it->trackRef()].key() <<std::endl;
        outTracks->back().setTrackRef(simple_track_to_track_map[it->trackRef()]);
      }
      if(it->gsfTrackRef().isNonnull()) {
        outTracks->back().setGsfTrackRef( simple_gsftrack_to_gsftrack_map[it->gsfTrackRef()] );
      }
      if (it->superClusterRef().isNonnull()) {
        const reco::SuperClusterRef & pfScRef(it->superClusterRef());

        float dx, dy, dz, dr;
        //float drMin = std::numeric_limits<float>::infinity();
        float drMin = 10.0;//also used as treshold for matching
        reco::SuperClusterRef ccrefMin;
        for( auto sc = bSc; sc != eSc; ++sc ) {
          const reco::SuperClusterRef & scRef(reco::SuperClusterRef(sCs,std::distance(bSc,sc)));
          dx = fabs(scRef->x()-pfScRef->x());
          dy = fabs(scRef->y()-pfScRef->y());
          dz = fabs(scRef->z()-pfScRef->z());
          dr = sqrt(dx*dx+dy*dy+dz*dz);
          if ( dr < drMin ) {
            drMin = dr;
            outTracks->back().setSuperClusterRef(scRef);
          }
        }
      }
      if (it->muonRef().isNonnull()) {
        //std::cout<<"pfmerge mu: "<<it->muonRef().id()<< " "<< it->muonRef().key()<< " " << simple_mu_to_mu_map[it->muonRef()].id() <<  " " << simple_mu_to_mu_map[it->muonRef()].key() <<std::endl;
        outTracks->back().setMuonRef(simple_mu_to_mu_map[it->muonRef()]);
      }
    }
  }  // end merge

  iEvent.put(std::move(outTracks), instance);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackColMerger);
DEFINE_FWK_MODULE(GsfTrackColMerger);
DEFINE_FWK_MODULE(MuonColMerger);
DEFINE_FWK_MODULE(GsfElectronColMerger);
DEFINE_FWK_MODULE(PhotonColMerger);
DEFINE_FWK_MODULE(ConversionColMerger);
DEFINE_FWK_MODULE(PFColMerger);
