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

void testReader::analyze(const edm::Event& event, const edm::EventSetup& setup)
{  
  edm::Handle<edm::View<reco::Track> > trackCollectionH;
  edm::View<reco::Track> trackCollection;
  LogTrace("testReader") << "testReader::analyze : getting reco::Track collection, "<<tracksTag;
  event.getByLabel(tracksTag,trackCollectionH);
  if (trackCollectionH.isValid()) { 
    trackCollection = *(trackCollectionH.product()); 
    LogTrace("testReader") << "... size = "<<trackCollection.size();
  }
  else LogTrace("testReader") << "... NOT FOUND.";

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  TrackingParticleCollection tPC;
  LogTrace("testReader") << "testReader::analyze : getting TrackingParticle collection, "<<tpTag;
  event.getByLabel(tpTag,TPCollectionH);
  if (TPCollectionH.isValid()) {
    tPC   = *(TPCollectionH.product());
    LogTrace("testReader") << "... size = "<<tPC.size();
  }
  else LogTrace("testReader") << "... NOT FOUND.";
  
  edm::Handle<reco::RecoToSimCollection> recSimH;
  reco::RecoToSimCollection recSimColl;
  LogTrace("testReader") << "testReader::analyze : getting  RecoToSimCollection - "<<assoMapsTag;
  event.getByLabel(assoMapsTag,recSimH);
  if (recSimH.isValid()) {
    recSimColl = *(recSimH.product());
    LogTrace("testReader") << "... size = "<<recSimColl.size();
  } else {
    LogTrace("testReader") << "... NOT FOUND.";
  }

  edm::Handle<reco::SimToRecoCollection> simRecH;
  reco::SimToRecoCollection simRecColl;
  LogTrace("testReader") << "testReader::analyze : getting  SimToRecoCollection - "<<assoMapsTag;
  event.getByLabel(assoMapsTag,simRecH);
  if (simRecH.isValid()) {
    simRecColl = *(simRecH.product());
    LogTrace("testReader") << "... size = "<<simRecColl.size();
  } else {
    LogTrace("testReader") << "... NOT FOUND.";
  }

  edm::LogVerbatim("testReader") <<"\n === Event ID = "<< event.id()<<" ===";
  
  //RECOTOSIM 
  edm::LogVerbatim("testReader") 
    << "\n                      ****************** Reco To Sim ****************** ";
  if (recSimH.isValid()) {
    
  edm::LogVerbatim("testReader")
    << "\n There are " << trackCollection.size() << " reco::Tracks "<< "("<<recSimColl.size()<<" matched) \n";

  for(edm::View<reco::Track>::size_type i=0; i<trackCollection.size(); ++i) {
    edm::RefToBase<reco::Track> track(trackCollectionH, i);
    
    if(recSimColl.find(track) != recSimColl.end()) {
      std::vector<std::pair<TrackingParticleRef, double> > recSimAsso = recSimColl[track];
      
      for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator IT = recSimAsso.begin(); 
	   IT != recSimAsso.end(); ++IT) {
	TrackingParticleRef trpart = IT->first;
	double purity = IT->second;
	edm::LogVerbatim("testReader") <<"reco::Track #" << int(i) << " with pt = " << track->pt()
				       << " associated to TrackingParticle #" <<trpart.key()
				       << " (pt = " << trpart->pt() << ") with Quality = " << purity;
      }
    } else {
      edm::LogVerbatim("testReader") << "reco::Track #" << int(i) << " with pt = " << track->pt()
				     << " NOT associated to any TrackingParticle" << "\n";		   
    } 
  }

  } else  edm::LogVerbatim("testReader") << "\n RtS map not found in the Event.";

  //SIMTORECO
  edm::LogVerbatim("testReader") 
    << "\n                      ****************** Sim To Reco ****************** ";
  if (simRecH.isValid()) {

  edm::LogVerbatim("testReader")
    << "\n There are " << tPC.size() << " TrackingParticles "<<"("<<simRecColl.size()<<" matched) \n";
  bool any_trackingParticle_matched = false;
 
  for (TrackingParticleCollection::size_type i=0; i<tPC.size(); i++) {
    TrackingParticleRef trpart(TPCollectionH, i);
    
    std::vector<std::pair<edm::RefToBase<reco::Track>, double> > simRecAsso;
    
    if(simRecColl.find(trpart) != simRecColl.end()) { 
      simRecAsso = (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >) simRecColl[trpart];
      
      for (std::vector<std::pair<edm::RefToBase<reco::Track>, double> >::const_iterator IT = simRecAsso.begin(); 
	   IT != simRecAsso.end(); ++IT) {
	edm::RefToBase<reco::Track> track = IT->first;
	double quality = IT->second;
	any_trackingParticle_matched = true;
	
	// find the purity from RecoToSim association (set purity = -1 for unmatched recoToSim)
	double purity = -1.;
	if(recSimColl.find(track) != recSimColl.end()) {
	  std::vector<std::pair<TrackingParticleRef, double> > recSimAsso = recSimColl[track];
	  for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator ITS = recSimAsso.begin(); 
	       ITS != recSimAsso.end(); ++ITS) {
	    TrackingParticleRef tp = ITS->first;
	    if (tp == trpart) purity = ITS->second;
	  }
	}
	
	edm::LogVerbatim("testReader") <<"TrackingParticle #" << int(i)<< " with pt = " << trpart->pt()
				       << " associated to reco::Track #" <<track.key()
				       << " (pt = " << track->pt() << ") with Quality = " << quality
	                               << " and Purity = "<< purity;
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

  } else edm::LogVerbatim("testReader") <<"\n StR map not found in the Event.";
  
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testReader);


