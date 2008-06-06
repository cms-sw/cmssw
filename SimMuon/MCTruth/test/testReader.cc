#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimMuon/MCTruth/test/testReader.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/RecoCandidate/interface/TrackAssociation.h"

testReader::testReader(const edm::ParameterSet& parset) :
  tracksTag(parset.getParameter< edm::InputTag >("tracksTag")),
  tpTag(parset.getParameter< edm::InputTag >("tpTag")),
  assoMapsTag(parset.getParameter< edm::InputTag >("assoMapsTag"))
{
}

testReader::~testReader() {
}

void testReader::beginJob(const edm::EventSetup & setup) {
}

void testReader::analyze(const edm::Event& event, const edm::EventSetup& setup)
{  
  edm::Handle<edm::View<reco::Track> > trackCollectionH;
  LogTrace("testReader") << "testReader::analyze : getting reco::Track collection, "<<tracksTag;
  event.getByLabel(tracksTag,trackCollectionH);
  const edm::View<reco::Track>  trackCollection = *(trackCollectionH.product()); 
  LogTrace("testReader") << "... size = "<<trackCollection.size();

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  LogTrace("testReader") << "testReader::analyze : getting TrackingParticle collection, "<<tpTag;
  event.getByLabel(tpTag,TPCollectionH);
  const TrackingParticleCollection tPC   = *(TPCollectionH.product());
  LogTrace("testReader") << "... size = "<<tPC.size();

  edm::Handle<reco::RecoToSimCollection> recSimH;
  reco::RecoToSimCollection recSimColl;
  LogTrace("testReader") << "testReader::analyze : getting  RecoToSimCollection - "<<assoMapsTag;
  event.getByLabel(assoMapsTag,recSimH);
  if (recSimH.isValid()) {
    recSimColl = *(recSimH.product());
    LogTrace("testReader") << "... size = "<<recSimColl.size();
  } else {
    LogTrace("testReader") << "... NO  RecoToSimCollection found !";
  }

  edm::Handle<reco::SimToRecoCollection> simRecH;
  reco::SimToRecoCollection simRecColl;
  LogTrace("testReader") << "testReader::analyze : getting  SimToRecoCollection - "<<assoMapsTag;
  event.getByLabel(assoMapsTag,simRecH);
  if (simRecH.isValid()) {
    simRecColl = *(simRecH.product());
    LogTrace("testReader") << "... size = "<<simRecColl.size();
  } else {
    LogTrace("testReader") << "... NO  SimToRecoCollection found !";
  }

  edm::LogVerbatim("testReader") <<"\n === Event ID = "<< event.id()<<" ===";
  
  //RECOTOSIM 
  edm::LogVerbatim("testReader") 
    << "\n                      ****************** Reco To Sim ****************** ";
  edm::LogVerbatim("testReader")
    << "\n There are " << trackCollection.size() << " reco::Track's"<<"\n";

  for(edm::View<reco::Track>::size_type i=0; i<trackCollection.size(); ++i) {
    edm::RefToBase<reco::Track> track(trackCollectionH, i);
    
    if(recSimColl.find(track) != recSimColl.end()) {
      std::vector<std::pair<TrackingParticleRef, double> > recSimAsso = recSimColl[track];
      
      if (recSimAsso.size()!=0) {
	for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator IT = recSimAsso.begin(); 
	     IT != recSimAsso.end(); ++IT) {
	  TrackingParticleRef trpart = IT->first;
	  double quality = IT->second;
	  edm::LogVerbatim("testReader") <<"reco::Track #" << int(i) << " with pt = " << track->pt()
					 << " associated to TrackingParticle #" <<trpart.key()
					 << " (pt = " << trpart->pt() << ") with Quality = " << quality;
	}
      }
    } else {
      edm::LogVerbatim("testReader") << "reco::Track #" << int(i) << " with pt = " << track->pt()
				     << " NOT associated to any TrackingParticle" << "\n";		   
    } 
  }

  //SIMTORECO
  edm::LogVerbatim("testReader") 
    << "\n                      ****************** Sim To Reco ****************** ";
  edm::LogVerbatim("testReader")
    << "\n There are " << tPC.size() << " TrackingParticle's"<<"\n";
  bool any_trackingParticle_matched = false;
 
  for (TrackingParticleCollection::size_type i=0; i<tPC.size(); i++) {
    TrackingParticleRef trpart(TPCollectionH, i);
    
    std::vector<std::pair<edm::RefToBase<reco::Track>, double> > simRecAsso;
    if(simRecColl.find(trpart) != simRecColl.end()) { 
      simRecAsso = (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >) simRecColl[trpart];
      
      if (simRecAsso.size()!=0) {
	for (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >::const_iterator IT = simRecAsso.begin(); 
	     IT != simRecAsso.end(); ++IT) {
	  edm::RefToBase<reco::Track> track = IT->first;
	  double quality = IT->second;
	  edm::LogVerbatim("testReader") <<"TrackingParticle #" << int(i)<< " with pt = " << trpart->pt()
					 << " associated to reco::Track #" <<track.key()
					 << " (pt = " << track->pt() << ") with Quality = " << quality;
	  any_trackingParticle_matched = true;
	}
      }
    }
    
    //    else 
    //      {
    //	LogTrace("testReader") << "TrackingParticle #" << int(i)<< " with pt = " << trpart->pt() 
    //			       << " NOT associated to any reco::Track" ;
    //      }    
  }
  
  if (!any_trackingParticle_matched) {
    edm::LogVerbatim("testReader") << "NO TrackingParticle associated to ANY input reco::Track !" << "\n";
  }
  
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testReader);


