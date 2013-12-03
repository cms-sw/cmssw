#include "TauAnalysis/MCEmbeddingTools/plugins/MuonRadiationFilter.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <TMath.h>

const double nomMassZ = 91.1876;

MuonRadiationFilter::MuonRadiationFilter(const edm::ParameterSet& cfg)
{
  srcSelectedMuons_               = cfg.getParameter<edm::InputTag>("srcSelectedMuons");
  srcPFCandsNoPU_                 = cfg.getParameter<edm::InputTag>("srcPFCandsNoPU");
  srcPFCandsPU_                   = cfg.getParameter<edm::InputTag>("srcPFCandsPU");

  minPtLow_                       = cfg.getParameter<double>("minPtLow");
  dRlowPt_                        = cfg.getParameter<double>("dRlowPt");
  addCaloEnECALlowPt_             = cfg.getParameter<bool>("addCaloEnECALlowPt");
  applyMassWindowSelectionLowPt_  = cfg.getParameter<bool>("applyMassWindowSelectionLowPt");

  minPtHigh_                      = cfg.getParameter<double>("minPtHigh");
  dRhighPt_                       = cfg.getParameter<double>("dRhighPt");  
  addCaloEnECALhighPt_            = cfg.getParameter<bool>("addCaloEnECALhighPt");
  applyMassWindowSelectionHighPt_ = cfg.getParameter<bool>("applyMassWindowSelectionHighPt");

  dRvetoCone_                     = cfg.getParameter<double>("dRvetoCone");
  dRisoCone_                      = cfg.getParameter<double>("dRisoCone");
  maxRelIso_                      = cfg.getParameter<double>("maxRelIso");

  maxMass_                        = cfg.getParameter<double>("maxMass");
  
  invert_                         = cfg.getParameter<bool>("invert");
  filter_                         = cfg.getParameter<bool>("filter");
  if ( !filter_ ) {
    produces<bool>();
  }

  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
}

namespace
{
  reco::Candidate::LorentzVector makeCaloEnP4(const reco::Candidate::LorentzVector& refP4, double caloEn)
  {
    double ux = TMath::Cos(refP4.phi())*TMath::Sin(refP4.theta());
    double uy = TMath::Sin(refP4.phi())*TMath::Sin(refP4.theta());
    double uz = TMath::Cos(refP4.theta());
    reco::Candidate::LorentzVector p4(ux*caloEn, uy*caloEn, uz*caloEn, caloEn);
    return p4;
  }
}

bool MuonRadiationFilter::filter(edm::Event& evt, const edm::EventSetup& es)
{
  if ( verbosity_ ) std::cout << "<MuonRadiationFilter::filter>:" << std::endl;

  std::vector<reco::CandidateBaseRef> selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);

  if ( muPlus.isNull() || muMinus.isNull() ) return false; // not selected Z --> mu+ mu- event: reject event

  edm::Handle<PFView> pfCandidatesNoPU;
  evt.getByLabel(srcPFCandsNoPU_, pfCandidatesNoPU);
  edm::Handle<PFView> pfCandidatesPU;
  evt.getByLabel(srcPFCandsPU_, pfCandidatesPU);

  double muPlusCaloEnECAL = compCaloEnECAL(muPlus->p4(), *pfCandidatesNoPU);
  if ( verbosity_ ) std::cout << "muPlusCaloEnECAL = " << muPlusCaloEnECAL << std::endl;
  reco::Candidate::LorentzVector caloP4muPlus = makeCaloEnP4(muPlus->p4(), muPlusCaloEnECAL);
  double muMinusCaloEnECAL = compCaloEnECAL(muMinus->p4(), *pfCandidatesNoPU);
  if ( verbosity_ ) std::cout << "muMinusCaloEnECAL = " << muMinusCaloEnECAL << std::endl;
  reco::Candidate::LorentzVector caloP4muMinus = makeCaloEnP4(muMinus->p4(), muMinusCaloEnECAL);

  bool isMuonRadiation = false;
  
  for ( PFView::const_iterator pfCandidate = pfCandidatesNoPU->begin();
	pfCandidate != pfCandidatesNoPU->end(); ++pfCandidate ) {
    if ( (*pfCandidate)->particleId() == reco::PFCandidate::gamma || (*pfCandidate)->particleId() == reco::PFCandidate::e ) { // CV: include converted photons
      double dRmuPlus = deltaR((*pfCandidate)->p4(), muPlus->p4());
      if ( checkMuonRadiation((*pfCandidate)->p4(), &caloP4muPlus,  dRmuPlus,  *pfCandidatesNoPU, *pfCandidatesPU, *muPlus, *muMinus) ) isMuonRadiation = true;
      double dRmuMinus = deltaR((*pfCandidate)->p4(), muMinus->p4());
      if ( checkMuonRadiation((*pfCandidate)->p4(), &caloP4muMinus, dRmuMinus, *pfCandidatesNoPU, *pfCandidatesPU, *muPlus, *muMinus) ) isMuonRadiation = true;
    }
  }
  
  if ( checkMuonRadiation(caloP4muPlus,  0, 0., *pfCandidatesNoPU, *pfCandidatesPU, *muPlus, *muMinus) ) isMuonRadiation = true;
  if ( checkMuonRadiation(caloP4muMinus, 0, 0., *pfCandidatesNoPU, *pfCandidatesPU, *muPlus, *muMinus) ) isMuonRadiation = true;
  
  if ( verbosity_ ) std::cout << "isMuonRadiation = " << isMuonRadiation << std::endl;

  if ( filter_ ) {
    if ( invert_ != isMuonRadiation ) return false; // reject events with muon -> muon + photon radiation
    else return true;
  } else {
    std::auto_ptr<bool> filter_result(new bool(invert_ != !isMuonRadiation));
    evt.put(filter_result);
    return true;
  }
}

double MuonRadiationFilter::compCaloEnECAL(const reco::Candidate::LorentzVector& refP4, const PFView& pfCandidates)
{
  double caloEnECAL = 0.;
  for ( PFView::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    double dR = deltaR(refP4, (*pfCandidate)->p4());
    if ( dR < dRvetoCone_ ) {
      caloEnECAL += (*pfCandidate)->ecalEnergy();
    }
  }
  return caloEnECAL;
}

void MuonRadiationFilter::compPFIso_raw(const reco::Candidate::LorentzVector& refP4, 
					const PFView& pfCandidates,
					const reco::Candidate::LorentzVector& muPlusP4, const reco::Candidate::LorentzVector& muMinusP4,
					double& pfChargedCandIsoSum, double& pfGammaIsoSum, double& pfNeutralHadronIsoSum)
{
  pfChargedCandIsoSum   = 0.;
  pfGammaIsoSum         = 0.;
  pfNeutralHadronIsoSum = 0.;
  for ( PFView::const_iterator pfCandidate = pfCandidates.begin();
	pfCandidate != pfCandidates.end(); ++pfCandidate ) {
    double dR = deltaR(refP4, (*pfCandidate)->p4());
    if ( dR < dRisoCone_ ) {
      bool isVeto = false;
      if ( dR < dRvetoCone_ ) {
	isVeto = true;
      } else {
	double dRmuPlus  = deltaR(muPlusP4,  (*pfCandidate)->p4());
	if ( dRmuPlus  < dRvetoCone_ ) isVeto = true;
	double dRmuMinus = deltaR(muMinusP4, (*pfCandidate)->p4());
	if ( dRmuMinus < dRvetoCone_ ) isVeto = true;
      }
      if ( isVeto ) continue;
      if ( TMath::Abs((*pfCandidate)->charge()) > 0.5 ) {
	if ( verbosity_ ) std::cout << " adding PFChargedCand: Pt = " << (*pfCandidate)->pt() << ", eta = " << (*pfCandidate)->eta() << ", phi = " << (*pfCandidate)->phi() << " (dR = " << dR << ")" << std::endl;
	pfChargedCandIsoSum += (*pfCandidate)->pt();
      } else if ( (*pfCandidate)->particleId() == reco::PFCandidate::gamma ) {
	if ( verbosity_ ) std::cout << " adding PFGamma: Pt = " << (*pfCandidate)->pt() << ", eta = " << (*pfCandidate)->eta() << ", phi = " << (*pfCandidate)->phi() << " (dR = " << dR << ")" << std::endl;
	pfGammaIsoSum += (*pfCandidate)->pt();
      } else if ( (*pfCandidate)->particleId() == reco::PFCandidate::h0 ) {
	if ( verbosity_ ) std::cout << " adding PFNeutralHadron: Pt = " << (*pfCandidate)->pt() << ", eta = " << (*pfCandidate)->eta() << ", phi = " << (*pfCandidate)->phi() << " (dR = " << dR << ")" << std::endl;
	pfNeutralHadronIsoSum += (*pfCandidate)->pt();
      } 
    }
  }
}

double MuonRadiationFilter::compPFIso_puCorr(const reco::Candidate::LorentzVector& refP4, 
					     const PFView& pfCandidatesNoPU, const PFView& pfCandidatesPU,
					     const reco::Candidate::LorentzVector& muPlusP4, const reco::Candidate::LorentzVector& muMinusP4)
{
  if ( verbosity_ ) std::cout << "computing isoSum(NoPU)" << std::endl;
  double pfChargedCandIsoSumNoPU, pfGammaIsoSumNoPU, pfNeutralHadronIsoSumNoPU;
  compPFIso_raw(refP4, pfCandidatesNoPU, muPlusP4, muMinusP4, pfChargedCandIsoSumNoPU, pfGammaIsoSumNoPU, pfNeutralHadronIsoSumNoPU);
  
  if ( verbosity_ ) std::cout << "computing isoSum(PU)" << std::endl;
  double pfChargedCandIsoSumPU, pfGammaIsoSumPU, pfNeutralHadronIsoSumPU;
  compPFIso_raw(refP4, pfCandidatesPU, muPlusP4, muMinusP4, pfChargedCandIsoSumPU, pfGammaIsoSumPU, pfNeutralHadronIsoSumPU);
  
  // apply delta-beta PU correction to isolation Pt sum
  double pfIso = pfChargedCandIsoSumNoPU + TMath::Max(0., pfGammaIsoSumNoPU + pfNeutralHadronIsoSumNoPU - 0.5*pfChargedCandIsoSumPU);
  if ( verbosity_ ) {
    std::cout << " pfChargedCandIsoSumNoPU = " << pfChargedCandIsoSumNoPU << std::endl;
    std::cout << " pfGammaIsoSumNoPU = " << pfGammaIsoSumNoPU << std::endl;
    std::cout << " pfNeutralHadronIsoSumNoPU = " << pfNeutralHadronIsoSumNoPU << std::endl;
    std::cout << " pfChargedCandIsoSumPU = " << pfChargedCandIsoSumPU << std::endl;
    std::cout << "--> isolation of PFGamma = " << pfIso << std::endl;
  }
  return pfIso;
}

namespace
{
  reco::Candidate::LorentzVector makeMuonP4(const reco::Candidate& muon_candidate, int verbosity)
  {
    const reco::Track* muonTrack = 0;
    if ( dynamic_cast<const reco::Muon*>(&muon_candidate) ) {
      const reco::Muon* muon = dynamic_cast<const reco::Muon*>(&muon_candidate);
      if      ( muon->innerTrack().isNonnull()  ) muonTrack = muon->innerTrack().get();
      else if ( muon->globalTrack().isNonnull() ) muonTrack = muon->globalTrack().get();
      else if ( muon->outerTrack().isNonnull()  ) muonTrack = muon->outerTrack().get();
    } else if( dynamic_cast<const reco::PFCandidate*>(&muon_candidate) ) {
      const reco::PFCandidate* pfCandidate = dynamic_cast<const reco::PFCandidate*>(&muon_candidate);
      if      ( pfCandidate->muonRef().isNonnull()     ) return makeMuonP4(*pfCandidate->muonRef(), verbosity);
      else if ( pfCandidate->trackRef().isNonnull()    ) muonTrack = pfCandidate->trackRef().get();
      else if ( pfCandidate->gsfTrackRef().isNonnull() ) muonTrack = pfCandidate->gsfTrackRef().get();
    } else if ( muon_candidate.hasMasterClone() ) {
      return makeMuonP4(*muon_candidate.masterClone(), verbosity);
    } else if ( muon_candidate.hasMasterClonePtr() ) {
      return makeMuonP4(*muon_candidate.masterClonePtr(), verbosity);
    }
    if ( muonTrack ) {
      double muonP  = muonTrack->p();
      double muonPx = muonP*TMath::Cos(muon_candidate.phi())*TMath::Sin(muon_candidate.theta());
      double muonPy = muonP*TMath::Sin(muon_candidate.phi())*TMath::Sin(muon_candidate.theta());
      double muonPz = muonP*TMath::Cos(muon_candidate.theta());      
      const double muonMass = 0.10566; // GeV
      double muonEn = TMath::Sqrt(muonP*muonP + muonMass*muonMass);
      reco::Candidate::LorentzVector muonP4(muonPx, muonPy, muonPz, muonEn);
      if ( verbosity ) {
	std::cout << "muon (original): Pt = " << muon_candidate.pt() << ", eta = " << muon_candidate.eta() << ", phi = " << muon_candidate.phi() << std::endl;
	std::cout << "muon (from track): Pt = " << muonP4.pt() << ", eta = " << muonP4.eta() << ", phi = " << muonP4.phi() << std::endl;
      }
      return muonP4;
    } else {
      return muon_candidate.p4();
    }
  }
}

bool MuonRadiationFilter::checkMuonRadiation(const reco::Candidate::LorentzVector& photonP4, const reco::Candidate::LorentzVector* caloP4mu, double dR, 
					     const PFView& pfCandidatesNoPU, const PFView& pfCandidatesPU,
					     const reco::Candidate& muPlus, const reco::Candidate& muMinus)
{
  bool isMuonRadiation = false;
  if ( dR < dRlowPt_ || dR < dRhighPt_ ) {
    reco::Candidate::LorentzVector muPlusP4_fromTrack  = makeMuonP4(muPlus,  verbosity_);
    reco::Candidate::LorentzVector muMinusP4_fromTrack = makeMuonP4(muMinus, verbosity_);
    double massWithoutPhoton = (muPlusP4_fromTrack + muMinusP4_fromTrack).mass();    
    reco::Candidate::LorentzVector photonP4sumLow = photonP4;
    if ( addCaloEnECALlowPt_ && caloP4mu ) photonP4sumLow += (*caloP4mu);
    if ( dR < dRlowPt_ && photonP4sumLow.pt() > minPtLow_ ) {
      double massWithPhoton = (muPlusP4_fromTrack + muMinusP4_fromTrack + photonP4sumLow).mass();
      if ( verbosity_ ) {
	std::cout << "low photon Pt threshold:" << std::endl;
	std::cout << " muon #1:" << "Pt = " << muPlusP4_fromTrack.pt() << ", eta = " << muPlusP4_fromTrack.eta() << ", phi = " << muPlusP4_fromTrack.phi() << std::endl;
	std::cout << " muon #2:" << "Pt = " << muMinusP4_fromTrack.pt() << ", eta = " << muMinusP4_fromTrack.eta() << ", phi = " << muMinusP4_fromTrack.phi() << std::endl;
	std::cout << " photon:" << "Pt = " << photonP4sumLow.pt() << ", eta = " << photonP4sumLow.eta() << ", phi = " << photonP4sumLow.phi() << " (dR = " << dR << ")" << std::endl;
	std::cout << " M(mu+mu) = " << massWithoutPhoton << ", M(mu+mu+gamma) = " << massWithPhoton << std::endl;
      }
      if ( (TMath::Abs(massWithPhoton - nomMassZ) < TMath::Abs(massWithoutPhoton - nomMassZ) && massWithPhoton < maxMass_) || !applyMassWindowSelectionLowPt_ ) {
	double pfIso = compPFIso_puCorr(photonP4sumLow, pfCandidatesNoPU, pfCandidatesPU, muPlus.p4(), muMinus.p4());
	if ( pfIso < (maxRelIso_*photonP4sumLow.pt()) ) isMuonRadiation = true;
      }
    }    
    reco::Candidate::LorentzVector photonP4sumHigh = photonP4;
    if ( addCaloEnECALhighPt_ && caloP4mu ) photonP4sumHigh += (*caloP4mu);
    if ( dR < dRhighPt_ && photonP4sumHigh.pt() > minPtHigh_ ) {
      double massWithPhoton = (muPlusP4_fromTrack + muMinusP4_fromTrack + photonP4sumHigh).mass();
      if ( verbosity_ ) {
	std::cout << "high photon Pt threshold:" << std::endl;
	std::cout << " muon #1:" << "Pt = " << muPlusP4_fromTrack.pt() << ", eta = " << muPlusP4_fromTrack.eta() << ", phi = " << muPlusP4_fromTrack.phi() << std::endl;
	std::cout << " muon #2:" << "Pt = " << muMinusP4_fromTrack.pt() << ", eta = " << muMinusP4_fromTrack.eta() << ", phi = " << muMinusP4_fromTrack.phi() << std::endl;
	std::cout << " photon:" << "Pt = " << photonP4sumHigh.pt() << ", eta = " << photonP4sumHigh.eta() << ", phi = " << photonP4sumHigh.phi() << " (dR = " << dR << ")" << std::endl;
	std::cout << " M(mu+mu) = " << massWithoutPhoton << ", M(mu+mu+gamma) = " << massWithPhoton << std::endl;
      }
      if ( (TMath::Abs(massWithPhoton - nomMassZ) < TMath::Abs(massWithoutPhoton - nomMassZ) && massWithPhoton < maxMass_) || !applyMassWindowSelectionHighPt_ ) {
	double pfIso = compPFIso_puCorr(photonP4sumHigh, pfCandidatesNoPU, pfCandidatesPU, muPlus.p4(), muMinus.p4());
	if ( pfIso < (maxRelIso_*photonP4sumHigh.pt()) ) isMuonRadiation = true;
      }
    }
  }
  return isMuonRadiation;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonRadiationFilter);
