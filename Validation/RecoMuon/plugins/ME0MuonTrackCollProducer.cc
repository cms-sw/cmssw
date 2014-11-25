
#include "Validation/RecoMuon/plugins/ME0MuonTrackCollProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h" 
#include <sstream>

// std::vector<double> ME0MuonTrackCollProducer::findSimVtx(edm::Event& iEvent){

//   edm::Handle<reco::GenParticleCollection> genParticles;
//   iEvent.getByLabel("genParticles", genParticles);
//   std::vector<double> vtxCoord;
//   vtxCoord.push_back(0);

//   if(genParticles.isValid()){

//   	for(reco::GenParticleCollection::const_iterator itg = genParticles->begin(); itg != genParticles->end(); ++itg ){

// 		int id = itg->pdgId();
// 		int status = itg->status();
// 		//std::cout<<"Id = "<<id<<std::endl;
// 		//int nDaughters = itg->numberOfDaughters();
// 		//double phiGen = itg->phi();
// 		//double etaGen = itg->eta();
// 		//std::cout<<"id "<<id<<" "<<phiGen<<" "<<etaGen<<std::endl;

// 		if(fabs(id) == 23 && status == 3) vtxCoord[0] = 1;

// 		if(fabs(id) == 13 && status == 3){

// 			vtxCoord.push_back(itg->vx()); 
// 			vtxCoord.push_back(itg->vy());
// 			vtxCoord.push_back(itg->vz());

// 		}

// 	}

//   }


//   //std::cout<<vtxCoord.size()<<" "<<vtxCoord[0]<<std::endl;
//   return vtxCoord;

// }


ME0MuonTrackCollProducer::ME0MuonTrackCollProducer(const edm::ParameterSet& parset) :
  muonsTag(parset.getParameter< edm::InputTag >("muonsTag")),
  vxtTag(parset.getParameter< edm::InputTag >("vxtTag")),
  useIPxy(parset.getUntrackedParameter< bool >("useIPxy", true)),
  useIPz(parset.getUntrackedParameter< bool >("useIPz", true)),
  selectionTags(parset.getParameter< std::vector<std::string> >("selectionTags")),
  trackType(parset.getParameter< std::string >("trackType")),
  parset_(parset)
{
  produces<reco::TrackCollection>();
}

ME0MuonTrackCollProducer::~ME0MuonTrackCollProducer() {
}

void ME0MuonTrackCollProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  //iEvent.getByLabel(muonsTag,muonCollectionH);


  iEvent.getByLabel <std::vector<reco::ME0Muon> > ("me0SegmentMatching", OurMuons);

  
  std::auto_ptr<reco::TrackCollection> selectedTracks(new reco::TrackCollection);
 
  reco::TrackRefProd rTracks = iEvent.getRefBeforePut<reco::TrackCollection>();
  


  for(std::vector<reco::ME0Muon>::const_iterator muon = OurMuons->begin();
       muon != OurMuons->end(); ++muon) {

    reco::TrackRef trackref;    
    if (muon->innerTrack().isNonnull()) trackref = muon->innerTrack();

      const reco::Track* trk = &(*trackref);
      // pointer to old track:
      reco::Track* newTrk = new reco::Track(*trk);

      selectedTracks->push_back( *newTrk );
      //selectedTrackExtras->push_back( *newExtra );
  }
  iEvent.put(selectedTracks);
  //iEvent.put(selectedTrackExtras);
  //iEvent.put(selectedTrackHits);
}
