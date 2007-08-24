// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// $Id: CaloJetTester.cc,v 1.4 2007/08/20 21:51:36 fedor Exp $

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/CaloJetfwd.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetfwd.h"

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
    mReverseEnergyFractionThreshold (iConfig.getParameter<double>("reverseEnergyFractionThreshold"))
{
  mEta = mPhi = mE = mP = mPt = mMass = mConstituents
    = mEtaFirst = mPhiFirst = mEFirst = mPtFirst 
    = mMaxEInEmTowers = mMaxEInHadTowers 
    = mHadEnergyInHO = mHadEnergyInHB = mHadEnergyInHF = mHadEnergyInHE 
    = mEmEnergyInEB = mEmEnergyInEE = mEmEnergyInHF 
    = mEnergyFractionHadronic = mEnergyFractionEm 
    = mN90
    = mAllGenJetsPt = mMatchedGenJetsPt = mAllGenJetsEta = mMatchedGenJetsEta 
    = mGenJetMatchEnergyFraction = mReverseMatchEnergyFraction 
    = mDeltaEta_B = mDeltaPhi_B = mEScale_B = mDeltaE_B
    = mDeltaEta_E = mDeltaPhi_E = mEScale_E = mDeltaE_E
    = mDeltaEta_F = mDeltaPhi_F = mEScale_F = mDeltaE_F
    = 0;
  
  DaqMonitorBEInterface* dbe = &*edm::Service<DaqMonitorBEInterface>();
  if (dbe) {
    dbe->setCurrentFolder("CaloJetTask_" + mInputCollection.label());
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
    mAllGenJetsPt = dbe->book1D("GenJetLOGpT", "GenJet LOG(pT_gen)", 16, 0, 4);
    mMatchedGenJetsPt = dbe->book1D("MatchedGenJetLOGpT", "MatchedGenJet LOG(pT_gen)", 16, 0, 4);
    mAllGenJetsEta = dbe->book2D("GenJetEta", "GenJet Eta vs LOG(pT_gen)", 16, 0, 4, 25, -5, 5);
    mMatchedGenJetsEta = dbe->book2D("MatchedGenJetEta", "MatchedGenJet Eta vs LOG(pT_gen)", 16, 0, 4, 25, -5, 5);
    mGenJetMatchEnergyFraction  = dbe->book2D("GenJetMatchEnergyFraction", "GenJetMatchEnergyFraction vs LOG(pT_gen)", 16, 0, 4, 101, 0, 1.01);
    mReverseMatchEnergyFraction  = dbe->book2D("ReverseMatchEnergyFraction", "ReverseMatchEnergyFraction vs LOG(pT_gen)", 16, 0, 4, 101, 0, 1.01);
    mDeltaEta_B = dbe->book2D("DeltaEta_B", "DeltaEta vs LOG(pT_gen) (|Eta_gen|<1.4)", 16, 0, 4, 100, -1, 1);
    mDeltaPhi_B = dbe->book2D("DeltaPhi_B", "DeltaPhi vs LOG(pT_gen) (|Eta_gen|<1.4", 16, 0, 4, 100, -1, 1);
    mEScale_B = dbe->book2D("EScale_B", "EnergyScale vs LOG(pT_gen) (|Eta_gen|<1.4", 16, 0, 4, 100, 0, 2);
    mDeltaE_B = dbe->book2D("DeltaE_B", "DeltaE vs LOG(pT_gen) (|Eta_gen|<1.4", 16, 0, 4, 2000, -200, 200);
    mDeltaEta_E = dbe->book2D("DeltaEta_E", "DeltaEta vs LOG(pT_gen) (1.4<=|Eta_gen|<3.)", 16, 0, 4, 100, -1, 1);
    mDeltaPhi_E = dbe->book2D("DeltaPhi_E", "DeltaPhi vs LOG(pT_gen) (1.4<=|Eta_gen|<3.", 16, 0, 4, 100, -1, 1);
    mEScale_E = dbe->book2D("EScale_E", "EnergyScale vs LOG(pT_gen) (1.4<=|Eta_gen|<3.", 16, 0, 4, 100, 0, 2);
    mDeltaE_E = dbe->book2D("DeltaE_E", "DeltaE vs LOG(pT_gen) (1.4<=|Eta_gen|<3.", 16, 0, 4, 2000, -200, 200);
    mDeltaEta_F = dbe->book2D("DeltaEta_F", "DeltaEta vs LOG(pT_gen) (|Eta_gen|>=3.)", 16, 0, 4, 100, -1, 1);
    mDeltaPhi_F = dbe->book2D("DeltaPhi_F", "DeltaPhi vs LOG(pT_gen) (|Eta_gen|>=3.", 16, 0, 4, 100, -1, 1);
    mEScale_F = dbe->book2D("EScale_F", "EnergyScale vs LOG(pT_gen) (|Eta_gen|>=3", 16, 0, 4, 100, 0, 2);
    mDeltaE_F = dbe->book2D("DeltaE_F", "DeltaE vs LOG(pT_gen) (|Eta_gen|>=3", 16, 0, 4, 2000, -200, 200);
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
 if (!mOutputFile.empty() && &*edm::Service<DaqMonitorBEInterface>()) edm::Service<DaqMonitorBEInterface>()->save (mOutputFile);
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

    std::vector <std::vector <const reco::GenParticleCandidate*> > genJetConstituents (genJets->size());
    for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
      genJetConstituents [iGenJet] = jetMatching.getGenParticles ((*genJets) [iGenJet]);
    }

    std::vector <std::vector <const reco::GenParticleCandidate*> > caloJetConstituents (caloJets->size());
    for (unsigned iCaloJet = 0; iCaloJet < caloJets->size(); ++iCaloJet) {
      caloJetConstituents [iCaloJet] = jetMatching.getGenParticles ((*caloJets) [iCaloJet], false);
    }

    for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {
      const GenJet& genJet = (*genJets) [iGenJet];
      double genJetPt = genJet.pt();
      if (fabs(genJet.eta()) > 5.) continue; // out of detector 
      if (genJetPt < mMatchGenPtThreshold) continue; // no low momentum 
      double logPtGen = log10 (genJetPt);
      mAllGenJetsPt->Fill (logPtGen);
      mAllGenJetsEta->Fill (logPtGen, genJet.eta());
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
      mGenJetMatchEnergyFraction->Fill (logPtGen, energyFractionBest);
      if (energyFractionBest > mGenEnergyFractionThreshold) { // good enough
	double reverseEnergyFraction = jetMatching.overlapEnergyFraction (caloJetConstituents [iCaloJetBest], 
									  genJetConstituents [iGenJet]);
	mReverseMatchEnergyFraction->Fill (logPtGen, reverseEnergyFraction);
	if (reverseEnergyFraction > mReverseEnergyFractionThreshold) { // good enough
	  // Matched!
	  const CaloJet& caloJet = (*caloJets) [iCaloJetBest];
	  mMatchedGenJetsPt->Fill (logPtGen);
	  mMatchedGenJetsEta->Fill (logPtGen, genJet.eta());
	  if (is_B (genJet)) mDeltaEta_B->Fill (logPtGen, caloJet.eta()-genJet.eta());
	  if (is_E (genJet)) mDeltaEta_E->Fill (logPtGen, caloJet.eta()-genJet.eta());
	  if (is_F (genJet)) mDeltaEta_F->Fill (logPtGen, caloJet.eta()-genJet.eta());
	  if (is_B (genJet)) mDeltaPhi_B->Fill (logPtGen, caloJet.phi()-genJet.phi());
	  if (is_E (genJet)) mDeltaPhi_B->Fill (logPtGen, caloJet.phi()-genJet.phi());
	  if (is_F (genJet)) mDeltaPhi_B->Fill (logPtGen, caloJet.phi()-genJet.phi());
	  if (is_B (genJet)) mEScale_B->Fill (logPtGen, caloJet.energy()/genJet.energy());
	  if (is_E (genJet)) mEScale_E->Fill (logPtGen, caloJet.energy()/genJet.energy());
	  if (is_F (genJet)) mEScale_F->Fill (logPtGen, caloJet.energy()/genJet.energy());
	  if (is_B (genJet)) mDeltaE_B->Fill (logPtGen, caloJet.energy()-genJet.energy());
	  if (is_E (genJet)) mDeltaE_E->Fill (logPtGen, caloJet.energy()-genJet.energy());
	  if (is_F (genJet)) mDeltaE_F->Fill (logPtGen, caloJet.energy()-genJet.energy());
	}
      }
    }
  }
}
