#include "TauAnalysis/MCEmbeddingTools/plugins/MuonRadiationFilter.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

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

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

bool MuonRadiationFilter::filter(edm::Event& evt, const edm::EventSetup& es)
{
  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  if ( muPlus.isNull() || muMinus.isNull() ) return false; // not selected Z --> mu+ mu- event: reject event

  typedef std::pair<reco::Particle::LorentzVector, reco::Particle::LorentzVector> LorentzVectorPair;
  std::vector<LorentzVectorPair> selMuonP4s;
  selMuonP4s.push_back(LorentzVectorPair(muPlus->p4(), muMinus->p4()));

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
      if ( verbosity_ ) {
	std::cout << "found PFGamma: Pt = " << pfCandidate->pt() << ", dR = " << dRmin << std::endl;
      }

      if ( (dRmin < dRlowPt_  && pfCandidate->pt() > minPtLow_ ) ||
	   (dRmin < dRhighPt_ && pfCandidate->pt() > minPtHigh_) ) {
	for ( std::vector<LorentzVectorPair>::const_iterator selMuonP4 = selMuonP4s.begin();
	      selMuonP4 != selMuonP4s.end(); ++selMuonP4 ) {
	  bool selMuon_isMuonRadiation = false;

	  double massWithoutPhoton = (selMuonP4->first + selMuonP4->second).mass();
	  double massWithPhoton = (selMuonP4->first + selMuonP4->second + pfCandidate->p4()).mass();
	  if ( verbosity_ ) {
	    std::cout << "muon #1:" << "Pt = " << selMuonP4->first.pt() << ", eta = " << selMuonP4->first.eta() << ", phi = " << selMuonP4->first.phi() << std::endl;
	    std::cout << "muon #2:" << "Pt = " << selMuonP4->second.pt() << ", eta = " << selMuonP4->second.eta() << ", phi = " << selMuonP4->second.phi() << std::endl;
	    std::cout << "M(mu+mu) = " << massWithoutPhoton << ", M(mu+mu+gamma) = " << massWithPhoton << std::endl;
	  }

	  if ( TMath::Abs(massWithPhoton - nomMassZ) < TMath::Abs(massWithoutPhoton - nomMassZ) && massWithPhoton < maxMass_ ) {
	    double pfChargedCandIsoSumNoPU, pfGammaIsoSumNoPU, pfNeutralHadronIsoSumNoPU;
	    compPFIso(pfCandidate->p4(), *pfCandidatesNoPU, pfChargedCandIsoSumNoPU, pfGammaIsoSumNoPU, pfNeutralHadronIsoSumNoPU);

	    double pfChargedCandIsoSumPU, pfGammaIsoSumPU, pfNeutralHadronIsoSumPU;
	    compPFIso(pfCandidate->p4(), *pfCandidatesPU, pfChargedCandIsoSumPU, pfGammaIsoSumPU, pfNeutralHadronIsoSumPU);

	    // apply delta-beta PU correction to isolation Pt sum
	    double pfIso = pfChargedCandIsoSumNoPU + TMath::Max(0., pfGammaIsoSumNoPU + pfNeutralHadronIsoSumNoPU - 0.5*pfChargedCandIsoSumPU);
	    if ( verbosity_ ) {
	      std::cout << "isolation of PFGamma = " << pfIso << std::endl;
	    }

	    if ( pfIso < (maxRelIso_*pfCandidate->pt()) ) selMuon_isMuonRadiation = true;
	  }
	  
	  if ( selMuon_isMuonRadiation ) isMuonRadiation = true;
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
