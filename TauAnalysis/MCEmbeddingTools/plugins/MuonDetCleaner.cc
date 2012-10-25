#include "TauAnalysis/MCEmbeddingTools/plugins/MuonDetCleaner.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <string>
#include <vector>

MuonDetCleaner::MuonDetCleaner(const edm::ParameterSet& cfg)
  : srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons"))
{
  // maps of detId to number of hits attributed to muon
  produces<detIdToIntMap>("hitsMuPlus");
  produces<detIdToIntMap>("hitsMuMinus");

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

MuonDetCleaner::~MuonDetCleaner()
{
// nothing to be done yet...
}

void MuonDetCleaner::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<detIdToIntMap> hitsMuPlus(new detIdToIntMap());
  std::auto_ptr<detIdToIntMap> hitsMuMinus(new detIdToIntMap());
  
  std::vector<reco::CandidateBaseRef > selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  fillHitMap(dynamic_cast<const reco::Muon*>(&*muPlus), *hitsMuPlus);
  fillHitMap(dynamic_cast<const reco::Muon*>(&*muMinus), *hitsMuMinus);

  evt.put(hitsMuPlus, "hitsMuPlus");
  evt.put(hitsMuMinus, "hitsMuMinus");
}

namespace
{
  void fillHitMapRH(const TrackingRecHit& rh, std::map<uint32_t, int>& hitMap, int& numHits)
  {
    std::vector<const TrackingRecHit*> rh_components = rh.recHits();
    if ( rh_components.size() == 0 ) {
      ++hitMap[rh.rawId()];
      ++numHits;
    } else {
      for ( std::vector<const TrackingRecHit*>::const_iterator rh_component = rh_components.begin();
	    rh_component != rh_components.end(); ++rh_component ) {
	fillHitMapRH(**rh_component, hitMap, numHits);
      }
    }
  }
}

void MuonDetCleaner::fillHitMap(const reco::Muon* muon, detIdToIntMap& hitMap)
{
  if ( muon->outerTrack().isNull() ) 
    throw cms::Exception("InvalidData") 
      << "Muon has no stand-alone muon track: Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() << " !!\n";
  
  const reco::TrackRef muonOuterTrack = muon->outerTrack();
  int numHits = 0;
  for ( trackingRecHit_iterator rh = muonOuterTrack->recHitsBegin();
	rh != muonOuterTrack->recHitsEnd(); ++rh ) {
    fillHitMapRH(**rh, hitMap, numHits);
  }

  if ( verbosity_ ) {
    std::string muonCharge_string = "";
    if      ( muon->charge() > +0.5 ) muonCharge_string = "+";
    else if ( muon->charge() < -0.5 ) muonCharge_string = "-";
    std::cout << "Mu" << muonCharge_string << ": Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() 
	      << " --> #Hits = " << numHits << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonDetCleaner);
