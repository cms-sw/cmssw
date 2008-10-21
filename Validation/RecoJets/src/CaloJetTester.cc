// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// $Id: CaloJetTester.cc,v 1.7 2008/02/29 20:49:02 ksmith Exp $

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "RecoJets/JetAlgorithms/interface/JetMatchingTools.h"

#include "CaloJetTester.h"

using namespace edm;
using namespace reco;
using namespace std;

namespace {
  bool is_B (const reco::Jet& fJet) {return fabs (fJet.eta()) < 1.4;}
  bool is_E (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 1.4 && fabs (fJet.eta()) < 3.;}
  bool is_F (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 3.;}
}

CaloJetTester::CaloJetTester(const edm::ParameterSet& iConfig)
  : mInputCollection (iConfig.getParameter<edm::InputTag>( "src" )),
    mInputGenCollection (iConfig.getParameter<edm::InputTag>( "srcGen" )),
    mOutputFile (iConfig.getUntrackedParameter<string>("outputFile", "")),
    mMatchGenPtThreshold (iConfig.getParameter<double>("genPtThreshold")),
    mGenEnergyFractionThreshold (iConfig.getParameter<double>("genEnergyFractionThreshold")),
    mReverseEnergyFractionThreshold (iConfig.getParameter<double>("reverseEnergyFractionThreshold")),
    mRThreshold (iConfig.getParameter<double>("RThreshold"))
{
  mEta = mPhi = mE = mP = mPt = mMass = mConstituents
    = mEtaFirst = mPhiFirst = mEFirst = mPtFirst 
    = mMaxEInEmTowers = mMaxEInHadTowers 
    = mHadEnergyInHO = mHadEnergyInHB = mHadEnergyInHF = mHadEnergyInHE 
    = mEmEnergyInEB = mEmEnergyInEE = mEmEnergyInHF 
    = mEnergyFractionHadronic = mEnergyFractionEm 
    = mN90
    = mAllGenJetsPt = mMatchedGenJetsPt = mAllGenJetsEta = mMatchedGenJetsEta 
    = mGenJetMatchEnergyFraction = mReverseMatchEnergyFraction = mRMatch
    = mDeltaEta = mDeltaPhi = mEScale = mDeltaE
    = 0;
  
  DQMStore* dbe = &*edm::Service<DQMStore>();
  if (dbe) {
    dbe->setCurrentFolder("RecoJetsV/CaloJetTask_" + mInputCollection.label());
    mEta = dbe->book1D("Eta", "Eta", 100, -5, 5); 
    mPhi = dbe->book1D("Phi", "Phi", 70, -3.5, 3.5); 
    mE = dbe->book1D("E", "E", 100, 0, 500); 
    mP = dbe->book1D("P", "P", 100, 0, 500); 
    mPt = dbe->book1D("Pt", "Pt", 100, 0, 50); 
    mMass = dbe->book1D("Mass", "Mass", 100, 0, 25); 
    mConstituents = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100); 
    //
    mEtaFirst = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5); 
    mPhiFirst = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5); 
    mEFirst = dbe->book1D("EFirst", "EFirst", 100, 0, 1000); 
    mPtFirst = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500); 
    //
    mMaxEInEmTowers = dbe->book1D("MaxEInEmTowers", "MaxEInEmTowers", 100, 0, 100); 
    mMaxEInHadTowers = dbe->book1D("MaxEInHadTowers", "MaxEInHadTowers", 100, 0, 100); 
    mHadEnergyInHO = dbe->book1D("HadEnergyInHO", "HadEnergyInHO", 100, 0, 10); 
    mHadEnergyInHB = dbe->book1D("HadEnergyInHB", "HadEnergyInHB", 100, 0, 50); 
    mHadEnergyInHF = dbe->book1D("HadEnergyInHF", "HadEnergyInHF", 100, 0, 50); 
    mHadEnergyInHE = dbe->book1D("HadEnergyInHE", "HadEnergyInHE", 100, 0, 100); 
    mEmEnergyInEB = dbe->book1D("EmEnergyInEB", "EmEnergyInEB", 100, 0, 50); 
    mEmEnergyInEE = dbe->book1D("EmEnergyInEE", "EmEnergyInEE", 100, 0, 50); 
    mEmEnergyInHF = dbe->book1D("EmEnergyInHF", "EmEnergyInHF", 120, -20, 100); 
    mEnergyFractionHadronic = dbe->book1D("EnergyFractionHadronic", "EnergyFractionHadronic", 120, -0.1, 1.1); 
    mEnergyFractionEm = dbe->book1D("EnergyFractionEm", "EnergyFractionEm", 120, -0.1, 1.1); 
    mN90 = dbe->book1D("N90", "N90", 50, 0, 50); 
    //
    double log10PtMin = 0.5;
    double log10PtMax = 4.;
    int log10PtBins = 14;
    double etaMin = -5.;
    double etaMax = 5.;
    int etaBins = 50;
    mAllGenJetsPt = dbe->book1D("GenJetLOGpT", "GenJet LOG(pT_gen)", 
				log10PtBins, log10PtMin, log10PtMax);
    mMatchedGenJetsPt = dbe->book1D("MatchedGenJetLOGpT", "MatchedGenJet LOG(pT_gen)", 
				    log10PtBins, log10PtMin, log10PtMax);
    mAllGenJetsEta = dbe->book2D("GenJetEta", "GenJet Eta vs LOG(pT_gen)", 
				 log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax);
    mMatchedGenJetsEta = dbe->book2D("MatchedGenJetEta", "MatchedGenJet Eta vs LOG(pT_gen)", 
				     log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax);
    mGenJetMatchEnergyFraction  = dbe->book3D("GenJetMatchEnergyFraction", "GenJetMatchEnergyFraction vs LOG(pT_gen) vs eta", 
					      log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 101, 0, 1.01);
    mReverseMatchEnergyFraction  = dbe->book3D("ReverseMatchEnergyFraction", "ReverseMatchEnergyFraction vs LOG(pT_gen) vs eta", 
					       log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 101, 0, 1.01);
    mRMatch  = dbe->book3D("RMatch", "delta(R)(Gen-Calo) vs LOG(pT_gen) vs eta", 
			   log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 60, 0, 3);
    mDeltaEta = dbe->book3D("DeltaEta", "DeltaEta vs LOG(pT_gen) vs eta", 
			      log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, -1, 1);
    mDeltaPhi = dbe->book3D("DeltaPhi", "DeltaPhi vs LOG(pT_gen) vs eta", 
			      log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, -1, 1);
    mEScale = dbe->book3D("EScale", "EnergyScale vs LOG(pT_gen) vs eta", 
			    log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, 0, 2);
    mDeltaE = dbe->book3D("DeltaE", "DeltaE vs LOG(pT_gen) vs eta", 
			    log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 2000, -200, 200);
  }

  if (mOutputFile.empty ()) {
    LogInfo("OutputInfo") << " CaloJet histograms will NOT be saved";
  } 
  else {
    LogInfo("OutputInfo") << " CaloJethistograms will be saved to file:" << mOutputFile;
  }
}
   
CaloJetTester::~CaloJetTester()
{
}

void CaloJetTester::beginJob(const edm::EventSetup& c){
}

void CaloJetTester::endJob() {
 if (!mOutputFile.empty() && &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (mOutputFile);
}


void CaloJetTester::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  Handle<CaloJetCollection> caloJets;
  mEvent.getByLabel(mInputCollection, caloJets);
  CaloJetCollection::const_iterator jet = caloJets->begin ();
  int jetIndex = 0;
  for (; jet != caloJets->end (); jet++, jetIndex++) {
    if (mEta) mEta->Fill (jet->eta());
    if (mPhi) mPhi->Fill (jet->phi());
    if (mE) mE->Fill (jet->energy());
    if (mP) mP->Fill (jet->p());
    if (mPt) mPt->Fill (jet->pt());
    if (mMass) mMass->Fill (jet->mass());
    if (mConstituents) mConstituents->Fill (jet->nConstituents());
    if (jet == caloJets->begin ()) { // first jet
      if (mEtaFirst) mEtaFirst->Fill (jet->eta());
      if (mPhiFirst) mPhiFirst->Fill (jet->phi());
      if (mEFirst) mEFirst->Fill (jet->energy());
      if (mPtFirst) mPtFirst->Fill (jet->pt());
    }
    if (mMaxEInEmTowers) mMaxEInEmTowers->Fill (jet->maxEInEmTowers());
    if (mMaxEInHadTowers) mMaxEInHadTowers->Fill (jet->maxEInHadTowers());
    if (mHadEnergyInHO) mHadEnergyInHO->Fill (jet->hadEnergyInHO());
    if (mHadEnergyInHB) mHadEnergyInHB->Fill (jet->hadEnergyInHB());
    if (mHadEnergyInHF) mHadEnergyInHF->Fill (jet->hadEnergyInHF());
    if (mHadEnergyInHE) mHadEnergyInHE->Fill (jet->hadEnergyInHE());
    if (mEmEnergyInEB) mEmEnergyInEB->Fill (jet->emEnergyInEB());
    if (mEmEnergyInEE) mEmEnergyInEE->Fill (jet->emEnergyInEE());
    if (mEmEnergyInHF) mEmEnergyInHF->Fill (jet->emEnergyInHF());
    if (mEnergyFractionHadronic) mEnergyFractionHadronic->Fill (jet->energyFractionHadronic());
    if (mEnergyFractionEm) mEnergyFractionEm->Fill (jet->emEnergyFraction());
    if (mN90) mN90->Fill (jet->n90());
  }

  // now match CaloJets to GenJets
  JetMatchingTools jetMatching (mEvent);
  if (!(mInputGenCollection.label().empty())) {
    Handle<GenJetCollection> genJets;
    mEvent.getByLabel(mInputGenCollection, genJets);

    std::vector <std::vector <const reco::GenParticle*> > genJetConstituents (genJets->size());
    std::vector <std::vector <const reco::GenParticle*> > caloJetConstituents (caloJets->size());
    if (mRThreshold > 0) { 
    }
    else {
      for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
	genJetConstituents [iGenJet] = jetMatching.getGenParticles ((*genJets) [iGenJet]);
      }
      
      for (unsigned iCaloJet = 0; iCaloJet < caloJets->size(); ++iCaloJet) {
	caloJetConstituents [iCaloJet] = jetMatching.getGenParticles ((*caloJets) [iCaloJet], false);
      }
    }

    for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
      const GenJet& genJet = (*genJets) [iGenJet];
      double genJetPt = genJet.pt();
      if (fabs(genJet.eta()) > 5.) continue; // out of detector 
      if (genJetPt < mMatchGenPtThreshold) continue; // no low momentum 
      double logPtGen = log10 (genJetPt);
      mAllGenJetsPt->Fill (logPtGen);
      mAllGenJetsEta->Fill (logPtGen, genJet.eta());
      if (caloJets->size() <= 0) continue; // no CaloJets - nothing to match
      if (mRThreshold > 0) {
	unsigned iCaloJetBest = 0;
	double deltaRBest = 999.;
	for (unsigned iCaloJet = 0; iCaloJet < caloJets->size(); ++iCaloJet) {
	  double dR = deltaR (genJet.eta(), genJet.phi(), (*caloJets) [iCaloJet].eta(), (*caloJets) [iCaloJet].phi());
	  if (deltaRBest < mRThreshold && dR < mRThreshold && genJet.pt() > 5.) {
	    std::cout << "Yet another matched jet for GenJet pt=" << genJet.pt()
		      << " previous CaloJet pt/dr: " << (*caloJets) [iCaloJetBest].pt() << '/' << deltaRBest
		      << " new CaloJet pt/dr: " << (*caloJets) [iCaloJet].pt() << '/' << dR
		      << std::endl;
	  }
	  if (dR < deltaRBest) {
	    iCaloJetBest = iCaloJet;
	    deltaRBest = dR;
	  }
	}
	mRMatch->Fill (logPtGen, genJet.eta(), deltaRBest);
	if (deltaRBest < mRThreshold) { // Matched
	  fillMatchHists (genJet, (*caloJets) [iCaloJetBest]);
	}
      }
      else {
	unsigned iCaloJetBest = 0;
	double energyFractionBest = 0.;
	for (unsigned iCaloJet = 0; iCaloJet < caloJets->size(); ++iCaloJet) {
	  double energyFraction = jetMatching.overlapEnergyFraction (genJetConstituents [iGenJet], 
								     caloJetConstituents [iCaloJet]);
	  if (energyFraction > energyFractionBest) {
	    iCaloJetBest = iCaloJet;
	    energyFractionBest = energyFraction;
	  }
	}
	mGenJetMatchEnergyFraction->Fill (logPtGen, genJet.eta(), energyFractionBest);
	if (energyFractionBest > mGenEnergyFractionThreshold) { // good enough
	  double reverseEnergyFraction = jetMatching.overlapEnergyFraction (caloJetConstituents [iCaloJetBest], 
									    genJetConstituents [iGenJet]);
	  mReverseMatchEnergyFraction->Fill (logPtGen, genJet.eta(), reverseEnergyFraction);
	  if (reverseEnergyFraction > mReverseEnergyFractionThreshold) { // Matched
	    fillMatchHists (genJet, (*caloJets) [iCaloJetBest]);
	  }
	}
      }
    }
  }
}

void CaloJetTester::fillMatchHists (const reco::GenJet& fGenJet, const reco::CaloJet& fCaloJet) {
  double logPtGen = log10 (fGenJet.pt());
  mMatchedGenJetsPt->Fill (logPtGen);
  mMatchedGenJetsEta->Fill (logPtGen, fGenJet.eta());
  mDeltaEta->Fill (logPtGen, fGenJet.eta(), fCaloJet.eta()-fGenJet.eta());
  mDeltaPhi->Fill (logPtGen, fGenJet.eta(), fCaloJet.phi()-fGenJet.phi());
  mEScale->Fill (logPtGen, fGenJet.eta(), fCaloJet.energy()/fGenJet.energy());
  mDeltaE->Fill (logPtGen, fGenJet.eta(), fCaloJet.energy()-fGenJet.energy());
}
