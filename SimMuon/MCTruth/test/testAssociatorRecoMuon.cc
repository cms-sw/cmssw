#include <memory>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimMuon/MCTruth/interface/MuonToSimAssociatorBase.h"
#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include <iostream>
#include <string>
#include <map>

class testAssociatorRecoMuon : public edm::EDAnalyzer {
  
 public:
  testAssociatorRecoMuon(const edm::ParameterSet&);
  virtual ~testAssociatorRecoMuon();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  edm::InputTag muonsTag;
  edm::InputTag tpTag;

  std::string associatorLabel_;
 
  MuonToSimAssociatorBase::MuonTrackType trackType_;
};

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"

testAssociatorRecoMuon::testAssociatorRecoMuon(const edm::ParameterSet& parset) :
  muonsTag(parset.getParameter< edm::InputTag >("muonsTag")),
  tpTag(parset.getParameter< edm::InputTag >("tpTag")),
  associatorLabel_(parset.getParameter< std::string >("associatorLabel"))
{
    std::string trackType = parset.getParameter< std::string >("trackType");
    if (trackType == "inner") trackType_ = MuonToSimAssociatorBase::InnerTk;
    else if (trackType == "outer") trackType_ = MuonToSimAssociatorBase::OuterTk;
    else if (trackType == "global") trackType_ = MuonToSimAssociatorBase::GlobalTk;
    else if (trackType == "segments") trackType_ = MuonToSimAssociatorBase::Segments;
    else throw cms::Exception("Configuration") << "Track type '" << trackType << "' not supported.\n";
}

testAssociatorRecoMuon::~testAssociatorRecoMuon() {
}

void testAssociatorRecoMuon::analyze(const edm::Event& event, const edm::EventSetup& setup)
{  
  edm::ESHandle<MuonToSimAssociatorBase> associatorBase;
  setup.get<TrackAssociatorRecord>().get(associatorLabel_, associatorBase);
  const MuonToSimAssociatorBase* assoByHits = associatorBase.product();
  if (assoByHits == 0) throw cms::Exception("Configuration") << "The Track Associator with label '" << associatorLabel_ << "' is not a MuonAssociatorByHits.\n";

  edm::Handle<edm::View<reco::Muon> > muonCollectionH;
  LogTrace("testAssociatorRecoMuon") << "getting reco::Track collection "<<muonsTag;
  event.getByLabel(muonsTag,muonCollectionH);

  const edm::View<reco::Muon>  muonCollection = *(muonCollectionH.product()); 
  LogTrace("testAssociatorRecoMuon") << "...size = "<<muonCollection.size();

  edm::Handle<TrackingParticleCollection>  TPCollectionH ;
  LogTrace("testAssociatorRecoMuon") << "getting TrackingParticle collection "<<tpTag;
  event.getByLabel(tpTag,TPCollectionH);

  LogTrace("testAssociatorRecoMuon") << "...size = "<< TPCollectionH->size();

  edm::LogVerbatim("testAssociatorRecoMuon") <<"\n === Event ID = "<< event.id()<<" ===";
    
  //RECOTOSIM 
  edm::LogVerbatim("testAssociatorRecoMuon") 
    << "\n                      ****************** Reco To Sim ****************** ";

  MuonToSimAssociatorBase::MuonToSimCollection recSimColl;
  MuonToSimAssociatorBase::SimToMuonCollection simRecColl;

  assoByHits->associateMuons(recSimColl, simRecColl, muonCollectionH, trackType_, TPCollectionH, &event, &setup);


  edm::LogVerbatim("testAssociatorRecoMuon")
    << "\n There are " << muonCollection.size() << " reco::Muon "<< "("<<recSimColl.size()<<" matched) \n";

  for(edm::View<reco::Muon>::size_type i=0; i<muonCollection.size(); ++i) {
    edm::RefToBase<reco::Muon> track(muonCollectionH, i);
    
    if(recSimColl.find(track) != recSimColl.end()) {
      std::vector<std::pair<TrackingParticleRef, double> > recSimAsso = recSimColl[track];
      
      for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator IT = recSimAsso.begin(); 
	   IT != recSimAsso.end(); ++IT) {
	TrackingParticleRef trpart = IT->first;
	double purity = IT->second;
	edm::LogVerbatim("testAssociatorRecoMuon") <<"reco::Muon #" << int(i) << " with pt = " << track->pt()
					     << " associated to TrackingParticle #" << trpart.key()
					     << " (pdgId = " << trpart->pdgId() << ", pt = " << trpart->pt() << ") with Quality = " << purity;
      }
    } else {
      edm::LogVerbatim("testAssociatorRecoMuon") << "reco::Muon #" << int(i) << " with pt = " << track->pt()
					   << " NOT associated to any TrackingParticle" << "\n";		   
    } 
  }

  //SIMTORECO
  edm::LogVerbatim("testAssociatorRecoMuon") 
    << "\n                      ****************** Sim To Reco ****************** ";

  edm::LogVerbatim("testAssociatorRecoMuon")
    << "\n There are " << TPCollectionH->size() << " TrackingParticles "<<"("<<simRecColl.size()<<" matched) \n";

  bool any_trackingParticle_matched = false;

  for (TrackingParticleCollection::size_type i = 0; i < TPCollectionH->size(); i++) {
    TrackingParticleRef trpart(TPCollectionH, i);

    std::vector<std::pair<edm::RefToBase<reco::Muon>, double> > simRecAsso;
    if(simRecColl.find(trpart) != simRecColl.end()) { 
      simRecAsso = simRecColl[trpart];
      
      for (std::vector<std::pair<edm::RefToBase<reco::Muon>, double> >::const_iterator IT = simRecAsso.begin(); 
	   IT != simRecAsso.end(); ++IT) {
	edm::RefToBase<reco::Muon> track = IT->first;
	double quality = IT->second;
	any_trackingParticle_matched = true;

	// find the purity from RecoToSim association (set purity = -1 for unmatched recoToSim)
	double purity = -1.;
	if(recSimColl.find(track) != recSimColl.end()) {
          std::vector<std::pair<TrackingParticleRef, double> > recSimAsso = recSimColl[track];
	  for (std::vector<std::pair<TrackingParticleRef, double> >::const_iterator ITS = recSimAsso.begin(); 
	       ITS != recSimAsso.end(); ++ITS) {
	    TrackingParticleRef tp(ITS->first);
	    if (tp == trpart) purity = ITS->second;
	  }
	}

	edm::LogVerbatim("testAssociatorRecoMuon") <<"TrackingParticle #" << int(i)<< " with pdgId = " << trpart->pdgId() << ", pt = " << trpart->pt()
					     << " associated to reco::Muon #" <<track.key()
					     << " (pt = " << track->pt() << ") with Quality = " << quality
	                                     << " and Purity = "<< purity;
      }
    } 
    
  }
  if (!any_trackingParticle_matched) {
    edm::LogVerbatim("testAssociatorRecoMuon") << "NO TrackingParticle associated to ANY input reco::Muon !" << "\n";
  }
  
}
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(testAssociatorRecoMuon);
