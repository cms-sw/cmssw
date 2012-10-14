#include "TauAnalysis/MCEmbeddingTools/plugins/MuonRadiationFilter.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

#include <TMath.h>

const double nomMassZ = 91.1876;

MuonRadiationFilter::MuonRadiationFilter(const edm::ParameterSet& cfg)
{
  srcSelectedMuons_ = cfg.getParameter<edm::InputTag>("srcSelectedMuons");
  srcPFCandsNoPU_   = cfg.getParameter<edm::InputTag>("srcPFCandsNoPU");
  srcPFCandsPU_     = cfg.getParameter<edm::InputTag>("srcPFCandsPU");

  minPtLow_         = cfg.getParameter<double>("minPtLow");
  dRlowPt_          = cfg.getParameter<double>("dRlowPt");

  minPtHigh_        = cfg.getParameter<double>("minPtHigh");
  dRhighPt_         = cfg.getParameter<double>("dRhighPt");  

  dRvetoCone_       = cfg.getParameter<double>("dRvetoCone");
  dRisoCone_        = cfg.getParameter<double>("dRisoCone");
  maxRelIso_        = cfg.getParameter<double>("maxRelIso");

  maxMass_          = cfg.getParameter<double>("maxMass");
}

bool MuonRadiationFilter::filter(edm::Event& evt, const edm::EventSetup& es)
{
  typedef std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> LorentzVectorPair;
  std::vector<LorentzVectorPair> selMuonP4s;
   
  edm::Handle<reco::CompositeCandidateCollection> combCandidates;
  if ( evt.getByLabel(srcSelectedMuons_, combCandidates) ) {
    for ( reco::CompositeCandidateCollection::const_iterator combCandidate = combCandidates->begin();
	  combCandidate != combCandidates->end(); ++combCandidate ) {
      if ( combCandidate->numberOfDaughters() != 2 )
	throw cms::Exception("Configuration") 
	  << "Invalid input collection 'selectedMuons' = " << srcSelectedMuons_ << " !!\n";
      selMuonP4s.push_back(LorentzVectorPair(combCandidate->daughter(0)->p4(), combCandidate->daughter(1)->p4()));
    }
  } else {
    edm::Handle<reco::MuonCollection> selectedMuons;
    if ( evt.getByLabel(srcSelectedMuons_, selectedMuons) ) {
      for ( reco::MuonCollection::const_iterator selectedMuon1 = selectedMuons->begin();
	    selectedMuon1 != selectedMuons->end(); ++selectedMuon1 ) {
	for ( reco::MuonCollection::const_iterator selectedMuon2 = selectedMuon1 + 1;
	      selectedMuon2 != selectedMuons->end(); ++selectedMuon2 ) {
	  if ( selectedMuon1->charge()*selectedMuon2->charge() < 0. ) {
	    selMuonP4s.push_back(LorentzVectorPair(selectedMuon1->p4(), selectedMuon2->p4()));
	  }
	}
      }
    } else {
      throw cms::Exception("Configuration") 
	<< "Invalid input collection 'selectedMuons' = " << srcSelectedMuons_ << " !!\n";
    }
  }
  
  if ( selMuonP4s.size() == 0 ) return false; // not selected Z --> mu+ mu- event: reject event

  edm::Handle<PFCandidateView> pfCandidatesNoPU;
  evt.getByLabel(srcPFCandsNoPU_, pfCandidatesNoPU);
  edm::Handle<PFCandidateView> pfCandidatesPU;
  evt.getByLabel(srcPFCandsPU_, pfCandidatesPU);

  bool isMuonRadiation = false;

  for ( PFCandidateView::const_iterator pfCandidate = pfCandidatesNoPU->begin();
	pfCandidate != pfCandidatesNoPU->end(); ++pfCandidate ) {
    if ( pfCandidate->particleId() == reco::PFCandidate::gamma ) {
      double dRmin = 1.e+3;
      for ( std::vector<LorentzVectorPair>::const_iterator selMuonP4 = selMuonP4s.begin();
	    selMuonP4 != selMuonP4s.end(); ++selMuonP4 ) {
	double dR1 = deltaR(pfCandidate->p4(), selMuonP4->first);
	if ( dR1 < dRmin ) dRmin = dR1;
	double dR2 = deltaR(pfCandidate->p4(), selMuonP4->second);
	if ( dR2 < dRmin ) dRmin = dR2;
      }
      if ( (dRmin < dRlowPt_  && pfCandidate->pt() > minPtLow_ ) ||
	   (dRmin < dRhighPt_ && pfCandidate->pt() > minPtHigh_) ) {
	for ( std::vector<LorentzVectorPair>::const_iterator selMuonP4 = selMuonP4s.begin();
	      selMuonP4 != selMuonP4s.end(); ++selMuonP4 ) {
	  double massWithoutPhoton = (selMuonP4->first + selMuonP4->second).mass();
	  double massWithPhoton = (selMuonP4->first + selMuonP4->second + pfCandidate->p4()).mass();
	  if ( TMath::Abs(massWithPhoton - nomMassZ) < TMath::Abs(massWithoutPhoton - nomMassZ) && massWithPhoton < maxMass_ ) {
	    double pfChargedCandIsoSumNoPU, pfGammaIsoSumNoPU, pfNeutralHadronIsoSumNoPU;
	    compPFIso(pfCandidate->p4(), *pfCandidatesNoPU, pfChargedCandIsoSumNoPU, pfGammaIsoSumNoPU, pfNeutralHadronIsoSumNoPU);

	    double pfChargedCandIsoSumPU, pfGammaIsoSumPU, pfNeutralHadronIsoSumPU;
	    compPFIso(pfCandidate->p4(), *pfCandidatesPU, pfChargedCandIsoSumPU, pfGammaIsoSumPU, pfNeutralHadronIsoSumPU);

	    // apply delta-beta PU correction to isolation Pt sum
	    double pfIso = pfChargedCandIsoSumNoPU + TMath::Max(0., pfGammaIsoSumNoPU + pfNeutralHadronIsoSumNoPU - 0.5*pfChargedCandIsoSumPU);
	    
	    if ( pfIso < (maxRelIso_*pfCandidate->pt()) ) isMuonRadiation = true;
	  }
	}
      }
    }    
  }

  if ( isMuonRadiation ) return false; // reject events with muon -> muon + photon radiation
  else return true;
}

void MuonRadiationFilter::compPFIso(const reco::Candidate::LorentzVector& photonP4, const PFCandidateView& pfCandidates,
				    double& pfChargedCandIsoSum, double& pfGammaIsoSum, double& pfNeutralHadronIsoSum)
{
  pfChargedCandIsoSum   = 0.;
  pfGammaIsoSum         = 0.;
  pfNeutralHadronIsoSum = 0.;
  for ( PFCandidateView::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    double dR = deltaR(photonP4, pfCandidate->p4());
    if ( dR > dRvetoCone_ && dR < dRisoCone_ ) {
      if ( TMath::Abs(pfCandidate->charge()) > 0.5 ) {
	pfChargedCandIsoSum += pfCandidate->pt();
      } else if ( pfCandidate->particleId() == reco::PFCandidate::gamma ) {
	pfGammaIsoSum += pfCandidate->pt();
      } else if ( pfCandidate->particleId() == reco::PFCandidate::h0 ) {
	pfNeutralHadronIsoSum += pfCandidate->pt();
      } 
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonRadiationFilter);
