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
  edm::ParameterSet cfgTrackAssociator = cfg.getParameter<edm::ParameterSet>("trackAssociator");
  trackAssociatorParameters_.loadParameters(cfgTrackAssociator);
  trackAssociator_.useDefaultPropagator();

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
  
  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  if ( muPlus.isNonnull()  ) fillHitMap(evt, es, &(*muPlus), *hitsMuPlus);
  if ( muMinus.isNonnull() ) fillHitMap(evt, es, &(*muMinus), *hitsMuMinus);

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

  void printHitMapRH(const edm::EventSetup& es, const std::map<uint32_t, int>& hitMap)
  {
    std::cout << "detIds:";
    for ( std::map<uint32_t, int>::const_iterator rh = hitMap.begin();
	  rh != hitMap.end(); ++rh ) {
      printMuonDetId(es, rh->first);
    }
  }
}

void MuonDetCleaner::fillHitMap(edm::Event& evt, const edm::EventSetup& es, const reco::Candidate* muon, detIdToIntMap& hitMap)
{
  int numHits = 0;
  
  const reco::Muon* recoMuon = dynamic_cast<const reco::Muon*>(muon);
  if ( recoMuon && recoMuon->outerTrack().isNonnull() ) {
    const reco::TrackRef muonOuterTrack = recoMuon->outerTrack();   
    TrackDetMatchInfo trackDetMatchInfo = trackAssociator_.associate(evt, es, *muonOuterTrack, trackAssociatorParameters_, TrackDetectorAssociator::Any);
    for ( std::vector<TAMuonChamberMatch>::const_iterator rh = trackDetMatchInfo.chambers.begin();
	  rh != trackDetMatchInfo.chambers.end(); ++rh ) {
      ++hitMap[rh->id.rawId()];
      ++numHits;
    }
    for ( trackingRecHit_iterator rh = muonOuterTrack->recHitsBegin();
	  rh != muonOuterTrack->recHitsEnd(); ++rh ) {
      fillHitMapRH(**rh, hitMap, numHits);
    }
  } else {
    GlobalVector muonP3(muon->px(), muon->py(), muon->pz()); 
    GlobalPoint muonVtx(muon->vertex().x(), muon->vertex().y(), muon->vertex().z());
    TrackDetMatchInfo trackDetMatchInfo = trackAssociator_.associate(evt, es, muonP3, muonVtx, muon->charge(), trackAssociatorParameters_);
    for ( std::vector<TAMuonChamberMatch>::const_iterator rh = trackDetMatchInfo.chambers.begin();
	  rh != trackDetMatchInfo.chambers.end(); ++rh ) {
      ++hitMap[rh->id.rawId()];
      ++numHits;
    }
  }
  
  if ( verbosity_ ) {
    std::string muonCharge_string = "";
    if      ( muon->charge() > +0.5 ) muonCharge_string = "+";
    else if ( muon->charge() < -0.5 ) muonCharge_string = "-";
    std::cout << "Mu" << muonCharge_string << ": Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() 
	      << " --> #Hits = " << numHits << std::endl;
    if ( verbosity_ >= 2 ) printHitMapRH(es, hitMap);    
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonDetCleaner);
