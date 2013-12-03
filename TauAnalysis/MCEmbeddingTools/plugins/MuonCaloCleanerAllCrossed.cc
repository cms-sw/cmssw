#include "TauAnalysis/MCEmbeddingTools/plugins/MuonCaloCleanerAllCrossed.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <string>
#include <vector>

MuonCaloCleanerAllCrossed::MuonCaloCleanerAllCrossed(const edm::ParameterSet& cfg)
  : srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons")),
    srcESrecHits_(cfg.getParameter<edm::InputTag>("esRecHits"))
{
  edm::ParameterSet cfgTrackAssociator = cfg.getParameter<edm::ParameterSet>("trackAssociator");
  trackAssociatorParameters_.loadParameters(cfgTrackAssociator);
  trackAssociator_.useDefaultPropagator();

  // maps of detId to energy deposit attributed to muon
  produces<detIdToFloatMap>("energyDepositsMuPlus");
  produces<detIdToFloatMap>("energyDepositsMuMinus");
}

MuonCaloCleanerAllCrossed::~MuonCaloCleanerAllCrossed()
{
// nothing to be done yet...
}

void MuonCaloCleanerAllCrossed::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<detIdToFloatMap> energyDepositsMuPlus(new detIdToFloatMap());
  std::auto_ptr<detIdToFloatMap> energyDepositsMuMinus(new detIdToFloatMap());
  
  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  if ( muPlus.isNonnull()  ) fillEnergyDepositMap(evt, es, &(*muPlus), *energyDepositsMuPlus);
  if ( muMinus.isNonnull() ) fillEnergyDepositMap(evt, es, &(*muMinus), *energyDepositsMuMinus);

  evt.put(energyDepositsMuPlus, "energyDepositsMuPlus");
  evt.put(energyDepositsMuMinus, "energyDepositsMuMinus");
}

void MuonCaloCleanerAllCrossed::fillEnergyDepositMap(edm::Event& evt, const edm::EventSetup& es, const reco::Candidate* muon, detIdToFloatMap& energyDepositMap)
{
  TrackDetMatchInfo trackDetMatchInfo = getTrackDetMatchInfo(evt, es, trackAssociator_, trackAssociatorParameters_, muon);
  
  for ( std::vector<const EcalRecHit*>::const_iterator rh = trackDetMatchInfo.crossedEcalRecHits.begin();
	rh != trackDetMatchInfo.crossedEcalRecHits.end(); ++rh ) {
    energyDepositMap[(*rh)->detid().rawId()] += (*rh)->energy();
  }
  
  for ( std::vector<const HBHERecHit*>::const_iterator rh = trackDetMatchInfo.crossedHcalRecHits.begin();
	rh != trackDetMatchInfo.crossedHcalRecHits.end(); ++rh ) {
    energyDepositMap[(*rh)->detid().rawId()] += (*rh)->energy();
  }
  
  for ( std::vector<const HORecHit*>::const_iterator rh = trackDetMatchInfo.crossedHORecHits.begin();
	rh != trackDetMatchInfo.crossedHORecHits.end(); ++rh ) {
    energyDepositMap[(*rh)->detid().rawId()] += (*rh)->energy();
  }
  
  // TF: there exists no better way
  edm::Handle<ESRecHitCollection> esRecHits;
  evt.getByLabel(srcESrecHits_, esRecHits);
  for ( std::vector<DetId>::const_iterator detId = trackDetMatchInfo.crossedPreshowerIds.begin();
	detId != trackDetMatchInfo.crossedPreshowerIds.end(); ++detId ) {
    ESRecHitCollection::const_iterator rh = esRecHits->find(*detId);
    if ( rh != esRecHits->end() ) {
      energyDepositMap[rh->detid().rawId()] += rh->energy();
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonCaloCleanerAllCrossed);
