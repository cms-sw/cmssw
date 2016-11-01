#include "TauAnalysis/MCEmbeddingTools/plugins/TrackMergeremb.h"

#include "DataFormats/TrackReco/interface/TrackExtra.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"



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




typedef TrackMergeremb<reco::TrackCollection> TrackMerger;
typedef TrackMergeremb<reco::GsfTrackCollection> GsfTrackMerger;
typedef TrackMergeremb<reco::ConversionTrackCollection> embConversionTrackMerger;
typedef TrackMergeremb<reco::MuonTrackLinksCollection> globalMuonMerger;
typedef TrackMergeremb<edm::ValueMap<reco::MuonQuality> > MuonQualityMerger;
typedef TrackMergeremb<reco::MuonCollection > MuonMerger;
typedef TrackMergeremb<reco::GsfElectronCollection > GsfElectronMerger;
typedef TrackMergeremb<reco::PhotonCollection > PhotonMerger;

typedef TrackMergeremb<reco::PFRecTrackCollection > PFTrackMerger;
typedef TrackMergeremb<reco::PFCandidateCollection > PFMerger;



// Here some overloaded functions, which are needed such that the right merger function is called for the indivudal Collections
template <typename T1>
void  TrackMergeremb<T1>::willproduce(std::string instance, std::string alias)
{
  produces<TrackCollectionemb>(instance);
}

template <typename T1>
void  TrackMergeremb<T1>::willconsume(const edm::ParameterSet& iConfig)
{

}

template <typename T1>
void  TrackMergeremb<T1>::merg_and_put(edm::Event& iEvent, std::string instance,  std::vector<edm::EDGetTokenT<TrackCollectionemb> > &to_merge)
{

      std::unique_ptr<TrackCollectionemb> outTracks = std::unique_ptr<TrackCollectionemb>(new TrackCollectionemb);
      
      
    for (auto akt_collection : to_merge){
        edm::Handle<TrackCollectionemb> track_col_in;  
        iEvent.getByToken(akt_collection, track_col_in);
          
        size_t sedref_it = 0;
        for (typename TrackCollectionemb::const_iterator it = track_col_in->begin(); it != track_col_in->end(); ++it, ++sedref_it) {
        outTracks->push_back( typename T1::value_type( *it ) );
	    	    
        }
	          
    }// end merge 
    
    iEvent.put(std::move(outTracks),instance);
  
  
  
}

template <>
void  TrackMergeremb<reco::TrackCollection>::willproduce(std::string instance, std::string alias)
{

    produces<reco::TrackCollection>(instance).setBranchAlias( alias + "Tracks" );
    produces<reco::TrackExtraCollection>(instance).setBranchAlias( alias + "TrackExtras" );
    produces<TrackingRecHitCollection>(instance).setBranchAlias( alias + "RecHits" );
    ///aaaaaa
    produces<TrackToTrackMapnew>();
    
}

template <>
void  TrackMergeremb<reco::TrackCollection>::merg_and_put(edm::Event& iEvent, std::string instance,  std::vector<edm::EDGetTokenT<TrackCollectionemb> > &to_merge )
{
    
    std::unique_ptr<TrackCollectionemb> outTracks = std::unique_ptr<TrackCollectionemb>(new TrackCollectionemb);
    std::unique_ptr<reco::TrackExtraCollection> outTracks_ex = std::unique_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());
    std::unique_ptr<TrackingRecHitCollection> outTracks_rh = std::unique_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
      
  //  auto rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
    auto rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
    auto rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
      
   std::vector<reco::TrackRef> trackRefColl;
    
    for (auto akt_collection : to_merge){
        edm::Handle<TrackCollectionemb> track_col_in;  
        iEvent.getByToken(akt_collection, track_col_in);
          
        unsigned sedref_it = 0;
        for (typename TrackCollectionemb::const_iterator it = track_col_in->begin(); it != track_col_in->end(); ++it, ++sedref_it) {
        outTracks->push_back( reco::Track( *it ) );
        outTracks_ex->push_back( reco::TrackExtra( *it->extra() ) );
        outTracks->back().setExtra( reco::TrackExtraRef( rTrackExtras, outTracks_ex->size() - 1) );	
	reco::TrackRef trackRefold(track_col_in, sedref_it);
	trackRefColl.push_back(trackRefold);	
        }
	          
    }// end merge 
    
  //  iEvent.put(std::move(outTracks),instance);
    
    edm::OrphanHandle<reco::TrackCollection> trackHandle = iEvent.put(std::move(outTracks),instance);
    iEvent.put(std::move(outTracks_ex),instance);
    iEvent.put(std::move(outTracks_rh),instance);
    
    
   auto fillMap = [](auto refH, auto& vec, edm::Event& ev, const std::string& cAl = ""){
     typedef  edm::ValueMap<typename std::decay<decltype(vec)>::type::value_type> MapType;
     std::unique_ptr<MapType > oMap(new MapType());
     
     {
       typename MapType::Filler filler(*oMap);
       filler.insert(refH, vec.begin(), vec.end());
       vec.clear();
       filler.fill();
     }
     ev.put(std::move(oMap), cAl);
    };
       
  fillMap(trackHandle, trackRefColl, iEvent);
 
}



template <>
void  TrackMergeremb<reco::GsfTrackCollection>::willproduce(std::string instance, std::string alias)
{
    produces<reco::GsfTrackCollection>(instance).setBranchAlias( alias + "GsfTracks" );
    produces<reco::TrackExtraCollection>(instance).setBranchAlias( alias + "TrackExtras" );
    produces<reco::GsfTrackExtraCollection>(instance).setBranchAlias( alias + "GsfTrackExtras" );
    produces<TrackingRecHitCollection>(instance).setBranchAlias( alias + "RecHits" );
}

template <>
void  TrackMergeremb<reco::GsfTrackCollection>::merg_and_put(edm::Event& iEvent, std::string instance,  std::vector<edm::EDGetTokenT<TrackCollectionemb> > &to_merge )
{
    std::unique_ptr<TrackCollectionemb> outTracks = std::unique_ptr<TrackCollectionemb>(new TrackCollectionemb);
    std::unique_ptr<reco::TrackExtraCollection> outTracks_ex = std::unique_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());
    std::unique_ptr<reco::GsfTrackExtraCollection> outTracks_exgsf = std::unique_ptr<reco::GsfTrackExtraCollection>(new reco::GsfTrackExtraCollection());
    std::unique_ptr<TrackingRecHitCollection> outTracks_rh = std::unique_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
    
    
    auto rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
    auto rTrackExtras_gsf = iEvent.getRefBeforePut<reco::GsfTrackExtraCollection>();
    
    auto rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
      
    for (auto akt_collection : to_merge){
        edm::Handle<TrackCollectionemb> track_col_in;  
        iEvent.getByToken(akt_collection, track_col_in);
          
        size_t sedref_it = 0;
        for (typename TrackCollectionemb::const_iterator it = track_col_in->begin(); it != track_col_in->end(); ++it, ++sedref_it) {
        outTracks->push_back( reco::GsfTrack( *it ) );
        outTracks_ex->push_back( reco::TrackExtra( *it->extra() ) );
        outTracks_exgsf->push_back( reco::GsfTrackExtra( *it->gsfExtra() ) );
        
        outTracks->back().setExtra( reco::TrackExtraRef( rTrackExtras, outTracks_ex->size() - 1) );
        outTracks->back().setGsfExtra( reco::GsfTrackExtraRef( rTrackExtras_gsf, outTracks_exgsf->size() - 1) );
        }
	          
    }// end merge 
    
    
    iEvent.put(std::move(outTracks),instance);
    iEvent.put(std::move(outTracks_ex),instance);
    iEvent.put(std::move(outTracks_exgsf),instance);
    iEvent.put(std::move(outTracks_rh),instance);

}



template <>
void  TrackMergeremb<reco::MuonTrackLinksCollection>::willproduce(std::string instance, std::string alias)
{

  produces<reco::TrackCollection>().setBranchAlias(alias + "Tracks");
  produces<TrackingRecHitCollection>().setBranchAlias(alias + "RecHits");
  produces<reco::TrackExtraCollection>().setBranchAlias( alias+ "TrackExtras");
  produces<reco::MuonTrackLinksCollection>().setBranchAlias( alias+ "s");
  
  produces<edm::ValueMap<reco::MuonQuality> >(instance);
  inputs_qual_[instance].push_back(consumes<edm::ValueMap<reco::MuonQuality> >(edm::InputTag("glbTrackQual",instance,"SIMembedding")));
  inputs_qual_[instance].push_back(consumes<edm::ValueMap<reco::MuonQuality> >(edm::InputTag("glbTrackQual",instance,"LHEembeddingCLEAN")));

    
}

template <>
void  TrackMergeremb<reco::MuonTrackLinksCollection>::merg_and_put(edm::Event& iEvent, std::string instance,  std::vector<edm::EDGetTokenT<TrackCollectionemb> > &to_merge )
{
  
    std::unique_ptr<reco::MuonTrackLinksCollection> outTracks_col = std::unique_ptr<reco::MuonTrackLinksCollection>(new reco::MuonTrackLinksCollection());
    std::unique_ptr<reco::TrackCollection> outTracks = std::unique_ptr<reco::TrackCollection>(new reco::TrackCollection);
    std::unique_ptr<reco::TrackExtraCollection> outTracks_ex = std::unique_ptr<reco::TrackExtraCollection>(new reco::TrackExtraCollection());
    std::unique_ptr<TrackingRecHitCollection> outTracks_rh = std::unique_ptr<TrackingRecHitCollection>(new TrackingRecHitCollection());
    
    std::unique_ptr<edm::ValueMap<reco::MuonQuality> > outTracks_qual = std::unique_ptr<edm::ValueMap<reco::MuonQuality> >(new edm::ValueMap<reco::MuonQuality>);
     edm::ValueMap<reco::MuonQuality>::Filler fillerQual(*outTracks_qual); 
    
  //  auto rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
    auto rTrackExtras = iEvent.getRefBeforePut<reco::TrackExtraCollection>();
    auto rTrack = iEvent.getRefBeforePut<reco::TrackCollection>();
    auto rHits = iEvent.getRefBeforePut<TrackingRecHitCollection>();
    unsigned col_count = 0;
    std::vector<reco::MuonQuality> valuesQual;
    
    for (auto akt_collection : to_merge){
        edm::Handle<TrackCollectionemb> track_col_in;  
        iEvent.getByToken(akt_collection, track_col_in);
	
	edm::Handle<edm::ValueMap<reco::MuonQuality> > track_col_qual;
        iEvent.getByToken(inputs_qual_[instance][col_count], track_col_qual);
        
        for (typename TrackCollectionemb::const_iterator it = track_col_in->begin(); it != track_col_in->end(); ++it) {
	  
	  reco::Track newTrackerTrack(*it->trackerTrack());
          outTracks->push_back( newTrackerTrack );
          outTracks_ex->push_back( reco::TrackExtra( *newTrackerTrack.extra() ) );
          outTracks->back().setExtra( reco::TrackExtraRef( rTrackExtras, outTracks_ex->size() - 1) );
	
	  reco::Track newStandAloneTrack(*it->standAloneTrack());
	  outTracks->push_back( newStandAloneTrack );
          outTracks_ex->push_back( reco::TrackExtra( *newStandAloneTrack.extra() ) );
          outTracks->back().setExtra( reco::TrackExtraRef( rTrackExtras, outTracks_ex->size() - 1) );

	  reco::Track newGlobalTrack(*it->globalTrack());
          outTracks->push_back( newGlobalTrack );
          outTracks_ex->push_back( reco::TrackExtra( *newGlobalTrack.extra() ) );
          outTracks->back().setExtra( reco::TrackExtraRef( rTrackExtras, outTracks_ex->size() - 1) );
	  
	  outTracks_col->push_back( reco::MuonTrackLinks( reco::TrackRef(rTrack ,outTracks->size() -3 ), 
							reco::TrackRef(rTrack ,outTracks->size() -2 ),
							reco::TrackRef(rTrack ,outTracks->size() -1 ) ) );
	
	
	 
       //   valuesQual.reserve(glbMuons->size());
	  valuesQual.push_back((*track_col_qual)[(it->globalTrack())]);
	  valuesQual.push_back((*track_col_qual)[(it->globalTrack())]);
	  valuesQual.push_back((*track_col_qual)[(it->globalTrack())]);
	    	    
        }

	col_count++;         
    }// end merge 
    iEvent.put(std::move(outTracks_col),instance);
    auto rTrack_after = iEvent.put(std::move(outTracks),instance);
    iEvent.put(std::move(outTracks_ex),instance);
    iEvent.put(std::move(outTracks_rh),instance);
        fillerQual.insert(rTrack_after, valuesQual.begin(), valuesQual.end());
 
    fillerQual.fill();
    iEvent.put(std::move(outTracks_qual),instance); 
 
}




template <>
void  TrackMergeremb<edm::ValueMap<reco::MuonQuality> >::willproduce(std::string instance, std::string alias)
{

   produces<edm::ValueMap<reco::MuonQuality> >(instance);
    
}

template <>
void  TrackMergeremb<edm::ValueMap<reco::MuonQuality> >::merg_and_put(edm::Event& iEvent, std::string instance,  std::vector<edm::EDGetTokenT<TrackCollectionemb> > &to_merge )
{
    std::unique_ptr<TrackCollectionemb> outTracks = std::unique_ptr<TrackCollectionemb>(new TrackCollectionemb);
    edm::ValueMap<reco::MuonQuality>::Filler fillerQual(*outTracks);  
    for (auto akt_collection : to_merge){
        edm::Handle<TrackCollectionemb> track_col_in;  
        iEvent.getByToken(akt_collection, track_col_in);
        
        fillerQual.add(*track_col_in);
        
    }// end merge 
    iEvent.put(std::move(outTracks),instance);    
}


template <>
void  TrackMergeremb<reco::MuonCollection >::willproduce(std::string instance, std::string alias)
{
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
void  TrackMergeremb<reco::MuonCollection >::willconsume(const edm::ParameterSet& iConfig)
{

   inputs_fixtrackrefs_ = consumes<TrackToTrackMapnew >( edm::InputTag("generalTracks") );
   inputs_fixtrackcol_ = consumes<reco::TrackCollection >( edm::InputTag("generalTracks") );
   
   
   
}


template <>
void  TrackMergeremb<reco::MuonCollection  >::merg_and_put(edm::Event& iEvent, std::string instance,  std::vector<edm::EDGetTokenT<TrackCollectionemb> > &to_merge )
{
    std::unique_ptr<TrackCollectionemb> outTracks = std::unique_ptr<TrackCollectionemb>(new TrackCollectionemb);
    std::unique_ptr<reco::CaloMuonCollection> calomu = std::unique_ptr<reco::CaloMuonCollection>(new reco::CaloMuonCollection);
    
    
    
    
    edm::Handle<TrackToTrackMapnew> track_ref_map;
    iEvent.getByToken(inputs_fixtrackrefs_, track_ref_map);
    
    edm::Handle<reco::TrackCollection> track_new_col;
    iEvent.getByToken(inputs_fixtrackcol_, track_new_col);   
    std::map<reco::TrackRef , reco::TrackRef > simple_track_to_track_map; //I didn't find a more elegant way, so just build a good old fassion std::map
     for (unsigned abc =0;  abc < track_new_col->size();  ++abc) {
      reco::TrackRef trackRef(track_new_col, abc);      
      simple_track_to_track_map[(*track_ref_map)[trackRef]] = trackRef;
    }
    
    
    
    
    std::vector<reco::MuonRef> muonRefColl;
    
    
    reco::MuonRefProd outputMuonsRefProd = iEvent.getRefBeforePut<reco::MuonCollection>();
    unsigned new_idx=0;
    for (auto akt_collection : to_merge){
        edm::Handle<TrackCollectionemb> track_col_in;  
        iEvent.getByToken(akt_collection, track_col_in);
	unsigned old_idx=0;
        for (typename TrackCollectionemb::const_iterator it = track_col_in->begin(); it != track_col_in->end(); ++it, ++old_idx, ++new_idx) {
        outTracks->push_back( reco::Muon( *it ) );
	 reco::MuonRef muRefold(track_col_in, old_idx);
	 muonRefColl.push_back(muRefold);
	 reco::MuonRef muRefnew(outputMuonsRefProd, new_idx);
         
         
        if (it->track().isNonnull()){
            //std::cout<<"pfmerge tr: "<<it->trackRef().id()<< " "<< it->trackRef().key()<< " " << simple_track_to_track_map[it->trackRef()].id() <<  " " << simple_track_to_track_map[it->trackRef()].key() <<std::endl;
	    outTracks->back().setTrack( simple_track_to_track_map[it->track()]  );
	 }
        }
	          
    }// end merge 
    
     const int nMuons=outTracks->size();

     std::vector<reco::MuonTimeExtra> dtTimeColl(nMuons);
     std::vector<reco::MuonTimeExtra> cscTimeColl(nMuons);
     std::vector<reco::MuonTimeExtra> combinedTimeColl(nMuons);
     std::vector<reco::IsoDeposit> trackDepColl(nMuons);
     std::vector<reco::IsoDeposit> ecalDepColl(nMuons);
     std::vector<reco::IsoDeposit> hcalDepColl(nMuons);
     std::vector<reco::IsoDeposit> hoDepColl(nMuons);
     std::vector<reco::IsoDeposit> jetDepColl(nMuons);
     
     //std::vector<reco::MuonRef> muonRefColl(nMuons);
     
     
    edm::OrphanHandle<reco::MuonCollection> muonHandle = iEvent.put(std::move(outTracks));
    
    auto fillMap = [](auto refH, auto& vec, edm::Event& ev, const std::string& cAl = ""){
     typedef  edm::ValueMap<typename std::decay<decltype(vec)>::type::value_type> MapType;
     std::unique_ptr<MapType > oMap(new MapType());
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
void  TrackMergeremb<reco::PFCandidateCollection >::willproduce(std::string instance, std::string alias)
{

  produces<reco::PFCandidateCollection>(instance);
  std::cout<<"Produce PF Collection: "<<instance<<std::endl;  
    
}


template <>
void  TrackMergeremb<reco::PFCandidateCollection>::willconsume(const edm::ParameterSet& iConfig)
{

   inputs_fixtrackrefs_ = consumes<TrackToTrackMapnew >( edm::InputTag("generalTracks") );
   inputs_fixtrackcol_ = consumes<reco::TrackCollection >( edm::InputTag("generalTracks") );
   
   inputs_fixmurefs_ = consumes<reco::MuonToMuonMap >( edm::InputTag("muons1stStep") );
   inputs_fixmucol_ = consumes<reco::MuonCollection >( edm::InputTag("muons1stStep") );
   
   
}



template <>
void  TrackMergeremb<reco::PFCandidateCollection >::merg_and_put(edm::Event& iEvent, std::string instance,  std::vector<edm::EDGetTokenT<TrackCollectionemb> > &to_merge )
{
    std::unique_ptr<TrackCollectionemb> outTracks = std::unique_ptr<TrackCollectionemb>(new TrackCollectionemb);
    
    edm::Handle<TrackToTrackMapnew> track_ref_map;
    iEvent.getByToken(inputs_fixtrackrefs_, track_ref_map);
    
    edm::Handle<reco::TrackCollection> track_new_col;
    iEvent.getByToken(inputs_fixtrackcol_, track_new_col);   
    std::map<reco::TrackRef , reco::TrackRef > simple_track_to_track_map; //I didn't find a more elegant way, so just build a good old fassion std::map
     for (unsigned abc =0;  abc < track_new_col->size();  ++abc) {
      reco::TrackRef trackRef(track_new_col, abc);      
      simple_track_to_track_map[(*track_ref_map)[trackRef]] = trackRef;
    }
    
    
    edm::Handle<reco::MuonToMuonMap> muon_ref_map;
    iEvent.getByToken(inputs_fixmurefs_, muon_ref_map);
    
    edm::Handle<reco::MuonCollection> muon_new_col;
    iEvent.getByToken(inputs_fixmucol_, muon_new_col);   
    std::map<reco::MuonRef , reco::MuonRef > simple_mu_to_mu_map; //I didn't find a more elegant way, so just build a good old fassion std::map
    for (unsigned abc =0;  abc < muon_new_col->size();  ++abc) {
      reco::MuonRef muRef(muon_new_col, abc);      
      simple_mu_to_mu_map[(*muon_ref_map)[muRef]] = muRef;
    }
    
    
      
    for (auto akt_collection : to_merge){
        edm::Handle<TrackCollectionemb> track_col_in;  
        iEvent.getByToken(akt_collection, track_col_in);     
        for (typename TrackCollectionemb::const_iterator it = track_col_in->begin(); it != track_col_in->end(); ++it) {
        outTracks->push_back( reco::PFCandidate( *it ) );
	//if (fabs(it->pdgId()) == 13){
	if (it->trackRef().isNonnull() && outTracks->back().charge()){
            //std::cout<<"pfmerge tr: "<<it->trackRef().id()<< " "<< it->trackRef().key()<< " " << simple_track_to_track_map[it->trackRef()].id() <<  " " << simple_track_to_track_map[it->trackRef()].key() <<std::endl;
            
	    outTracks->back().setTrackRef( simple_track_to_track_map[it->trackRef()]  );
	 }
	 
        if (it->muonRef().isNonnull()){
        	//std::cout<<"pfmerge mu: "<<it->muonRef().id()<< " "<< it->muonRef().key()<< " " << simple_mu_to_mu_map[it->muonRef()].id() <<  " " << simple_mu_to_mu_map[it->muonRef()].key() <<std::endl;
		outTracks->back().setMuonRef( simple_mu_to_mu_map[it->muonRef()]  );		
	        }
	 
        }	          
    }// end merge 

    iEvent.put(std::move(outTracks),instance);
     
}




#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackMerger);
DEFINE_FWK_MODULE(GsfTrackMerger);
DEFINE_FWK_MODULE(embConversionTrackMerger);
DEFINE_FWK_MODULE(globalMuonMerger);
DEFINE_FWK_MODULE(MuonQualityMerger);
DEFINE_FWK_MODULE(MuonMerger);
DEFINE_FWK_MODULE(PFTrackMerger);
DEFINE_FWK_MODULE(GsfElectronMerger);
DEFINE_FWK_MODULE(PhotonMerger);
DEFINE_FWK_MODULE(PFMerger);
