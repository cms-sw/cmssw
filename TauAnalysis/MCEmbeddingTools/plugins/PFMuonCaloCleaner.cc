#include "TauAnalysis/MCEmbeddingTools/plugins/PFMuonCaloCleaner.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFraction.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "DataFormats/ParticleFlowReco/interface/PFLayer.h"
#include "DataFormats/CaloRecHit/interface/CaloID.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <TMath.h>

#include <iostream>
#include <iomanip>

PFMuonCaloCleaner::PFMuonCaloCleaner(const edm::ParameterSet& cfg)
  : srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons")),
    srcPFCandidates_(cfg.getParameter<edm::InputTag>("pfCandidates")),
    dRmatch_(cfg.getParameter<double>("dRmatch"))
{
  // maps of detId to energy deposit attributed to muon
  produces<detIdToFloatMap>("muPlus");
  produces<detIdToFloatMap>("muMinus");
}

void PFMuonCaloCleaner::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<detIdToFloatMap> energyDepositMuPlus(new detIdToFloatMap());
  std::auto_ptr<detIdToFloatMap> energyDepositMuMinus(new detIdToFloatMap());

  std::vector<reco::CandidateBaseRef > selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  typedef edm::View<reco::PFCandidate> PFCandidateView;
  edm::Handle<PFCandidateView> pfCandidates;
  evt.getByLabel(srcPFCandidates_, pfCandidates);

  fillEnergyDepositMap(dynamic_cast<const reco::Muon*>(&*muPlus), *pfCandidates, *energyDepositMuPlus);
  fillEnergyDepositMap(dynamic_cast<const reco::Muon*>(&*muMinus), *pfCandidates, *energyDepositMuMinus);

  evt.put(energyDepositMuPlus, "muPlus");
  evt.put(energyDepositMuMinus, "muMinus");
}

void PFMuonCaloCleaner::fillEnergyDepositMap(const reco::Muon* muon, const edm::View<reco::PFCandidate>& pfCandidates, detIdToFloatMap& energyDepositMap)
{
  bool isMatched = false;
  for ( edm::View<reco::PFCandidate>::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    if ( pfCandidate->particleId() == reco::PFCandidate::mu ) {
      double dR = deltaR(pfCandidate->p4(), muon->p4());
      if ( dR < dRmatch_ ) {
	// NOTE: particleFlow sequence needs to be rerun for Zmumu event
	//       in order to recreate the links from PFCandidates->PFBlocks->PFClusters->PFRecHitFractions
	//      (not stored in AOD/RECO event content)
	const reco::PFCandidate::ElementsInBlocks& pfBlocks = pfCandidate->elementsInBlocks();
	for ( reco::PFCandidate::ElementsInBlocks::const_iterator pfBlock = pfBlocks.begin();
	      pfBlock != pfBlocks.end(); ++pfBlock ) {
	  const edm::OwnVector<reco::PFBlockElement>& pfBlockElements = pfBlock->first->elements();
	  for ( edm::OwnVector<reco::PFBlockElement>::const_iterator pfBlockElement = pfBlockElements.begin();
		pfBlockElement != pfBlockElements.end(); ++pfBlockElement ) {
	    if ( pfBlockElement->clusterRef().isNonnull() ) {
	      reco::PFClusterRef pfCluster = pfBlockElement->clusterRef();
	      const std::vector<reco::PFRecHitFraction>& pfRecHitFractions = pfCluster->recHitFractions();
	      for ( std::vector<reco::PFRecHitFraction>::const_iterator pfRecHitFraction = pfRecHitFractions.begin();
		    pfRecHitFraction != pfRecHitFractions.end(); ++pfRecHitFraction ) {
		const reco::PFRecHitRef& pfRecHit = pfRecHitFraction->recHitRef();
		energyDepositMap[pfRecHit->detId()] += pfRecHitFraction->fraction();
	      }
	    }
	  }
	}
      }
    }
  }
  if ( !isMatched ) {
    edm::LogWarning("PFMuonCaloCleaner") 
      << "Failed to match Muon to PFCandidate: Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() << " !!" << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFMuonCaloCleaner);





