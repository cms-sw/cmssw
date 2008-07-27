// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by J F Novak July 10, 2008
// $Id: CaloJetTester.cc,v 1.3 2008/07/21 21:57:26 chlebana Exp $

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

//I don't know which of these I actually need yet
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/METCollection.h"

#include "RecoJets/JetAlgorithms/interface/JetMatchingTools.h"

#include "CaloJetTester.h"

#include <cmath>

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
  mEta = mEtaFineBin = mPhi = mPhiFineBin = mE = mE_80 = mE_3000
    = mP = mP_80 = mP_3000 = mPt = mPt_80 = mPt_3000
    = mMass = mMass_80 = mMass_3000 = mConstituents = mConstituents_80
    = mEtaFirst = mPhiFirst = mEFirst = mEFirst_80 = mEFirst_3000 = mPtFirst 
    = mMaxEInEmTowers = mMaxEInHadTowers 
    = mHadEnergyInHO = mHadEnergyInHB = mHadEnergyInHF = mHadEnergyInHE 
    = mEmEnergyInEB = mEmEnergyInEE = mEmEnergyInHF 
    = mEnergyFractionHadronic = mEnergyFractionEm 
    = mHFLong = mHFTotal = mHFLong_80 = mHFLong_3000 = mHFShort = mHFShort_80 = mHFShort_3000
    = mN90
    = mAllGenJetsPt = mMatchedGenJetsPt = mAllGenJetsEta = mMatchedGenJetsEta 
    = mGenJetMatchEnergyFraction = mReverseMatchEnergyFraction = mRMatch
    = mDeltaEta = mDeltaPhi = mEScale = mDeltaE
    = mHadEnergyProfile = mEmEnergyProfile = mJetEnergyProfile = mHadJetEnergyProfile = mEMJetEnergyProfile
    = mHadTiming = mEmTiming = mCaloMEx = mCaloMEy = mCaloMETSig = mCaloMET = mCaloMETPhi = mCaloSumET 
    = 0;
  
  DQMStore* dbe = &*edm::Service<DQMStore>();
  if (dbe) {
    dbe->setCurrentFolder("RecoJetsV/CaloJetTask_" + mInputCollection.label());
    mEta = dbe->book1D("Eta", "Eta", 100, -5, 5); 
    mEtaFineBin = dbe->book1D("EtaFineBin_Pt30", "EtaFineBin_Pt30", 500, -5, 5); 
    mPhi = dbe->book1D("Phi", "Phi", 70, -3.5, 3.5); 
    mPhiFineBin = dbe->book1D("PhiFineBin_Pt30", "PhiFineBin_Pt30", 350, -3.5, 3.5); 
    mE = dbe->book1D("E", "E", 100, 0, 500); 
    mE_80 = dbe->book1D("E_80", "E_80", 100, 0, 4500); 
    mE_3000 = dbe->book1D("E_3000", "E_3000", 100, 0, 6000); 
    mP = dbe->book1D("P", "P", 100, 0, 500); 
    mP_80 = dbe->book1D("P_80", "P_80", 100, 0, 4500); 
    mP_3000 = dbe->book1D("P_3000", "P_3000", 100, 0, 6000); 
    mPt = dbe->book1D("Pt", "Pt", 100, 0, 50); 
    mPt_80   = dbe->book1D("Pt_80", "Pt_80", 100, 0, 140); 
    mPt_3000 = dbe->book1D("Pt_3000", "Pt_3000", 100, 0, 4000); 
    mMass = dbe->book1D("Mass", "Mass", 100, 0, 25); 
    mMass_80 = dbe->book1D("Mass_80", "Mass_80", 100, 0, 120); 
    mMass_3000 = dbe->book1D("Mass_3000", "Mass_3000", 100, 0, 1500); 
    mConstituents = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100); 
    mConstituents_80 = dbe->book1D("Constituents_80", "# of Constituents_80", 100, 0, 40); 
    //
    mEtaFirst = dbe->book1D("EtaFirst", "EtaFirst", 100, -5, 5); 
    mPhiFirst = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5); 
    mEFirst = dbe->book1D("EFirst", "EFirst", 100, 0, 1000); 
    mEFirst_80 = dbe->book1D("EFirst_80", "EFirst_80", 100, 0, 180); 
    mEFirst_3000 = dbe->book1D("EFirst_3000", "EFirst_3000", 100, 0, 4000); 
    mPtFirst = dbe->book1D("PtFirst", "PtFirst", 100, 0, 500); 
    //
    mMjj = dbe->book1D("Mjj", "Mjj", 100, 0, 200); 
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
    mHFTotal = dbe->book1D("HFTotal", "HFTotal", 100, 0, 500);
    mHFTotal_80 = dbe->book1D("HFTotal_80", "HFTotal_80", 100, 0, 3000);
    mHFTotal_3000 = dbe->book1D("HFTotal_3000", "HFTotal_3000", 100, 0, 6000);
    mHFLong = dbe->book1D("HFLong", "HFLong", 100, 0, 500);
    mHFLong_80 = dbe->book1D("HFLong_80", "HFLong_80", 100, 0, 200);
    mHFLong_3000 = dbe->book1D("HFLong_3000", "HFLong_3000", 100, 0, 1500);
    mHFShort = dbe->book1D("HFShort", "HFShort", 100, 0, 500);
    mHFShort_80 = dbe->book1D("HFShort_80", "HFShort_80", 100, 0, 200);
    mHFShort_3000 = dbe->book1D("HFShort_3000", "HFShort_3000", 100, 0, 1500);
    mN90 = dbe->book1D("N90", "N90", 50, 0, 50); 
    //
    mCaloMEx = dbe->book1D("CaloMEx","CaloMEx",200,-150,150);
    mCaloMEy = dbe->book1D("CaloMEy","CaloMEy",200,-150,150);
    mCaloMETSig = dbe->book1D("CaloMETSig","CaloMETSig",100,0,20);
    mCaloMET = dbe->book1D("CaloMET","CaloMET",100,0,200);
    mCaloMETPhi = dbe->book1D("CaloMETPhi","CaloMETPhi",70, 3.5, 3.5);
    mCaloSumET = dbe->book1D("CaloSumET","CaloSumET",100,0,1000);
    //
    mHadTiming = dbe->book1D("HadTiming", "HadTiming", 25, 0, 25);
    mEmTiming = dbe->book1D("EMTiming", "EMTiming", 15, 0 , 15);
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
    //
    mHadEnergyProfile = dbe->bookProfile2D("HadEnergyProfile", "HadEnergyProfile", 82, -41, 41, 73, 0, 73, 100, 0, 10000, "s");
    mEmEnergyProfile = dbe->bookProfile2D("EmEnergyProfile", "EmEnergyProfile", 82, -41, 41, 73, 0, 73, 100, 0, 10000, "s");
    mJetEnergyProfile = dbe->bookProfile2D("JetEnergyProfile", "JetEnergyProfile", 200, -5, 5, 200, -3.1415987, 3.1415987, 100, 0, 10000, "s");
    mHadJetEnergyProfile = dbe->bookProfile2D("HadJetEnergyProfile", "HadJetEnergyProfile", 200, -5, 5, 200, -3.1415987, 3.1415987, 100, 0, 10000, "s");
    mEMJetEnergyProfile = dbe->bookProfile2D("EMJetEnergyProfile", "EMJetEnergyProfile", 200, -5, 5, 200, -3.1415987, 3.1415987, 100, 0, 10000, "s");
    //
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
   const CaloMET *calomet;
      // Get CaloMET
      edm::Handle<CaloMETCollection> calo;
      mEvent.getByLabel("met", calo);
      if (!calo.isValid()) {
        edm::LogInfo("OutputInfo") << " failed to retrieve data required by MET Task";
        edm::LogInfo("OutputInfo") << " MET Task cannot continue...!";
       } else {
        const CaloMETCollection *calometcol = calo.product();
       calomet = &(calometcol->front());

      double caloSumET = calomet->sumEt();
      double caloMETSig = calomet->mEtSig();
      double caloMET = calomet->pt();
      double caloMEx = calomet->px();
      double caloMEy = calomet->py();
      double caloMETPhi = calomet->phi();


      mCaloMEx->Fill(caloMEx);
      mCaloMEy->Fill(caloMEy);
      mCaloMET->Fill(caloMET);
      mCaloMETPhi->Fill(caloMETPhi);
      mCaloSumET->Fill(caloSumET);
      mCaloMETSig->Fill(caloMETSig);
  }

 //Get the CaloTower collection
  Handle<CaloTowerCollection> caloTowers;
  mEvent.getByLabel( "towerMaker", caloTowers );
  for( CaloTowerCollection::const_iterator cal = caloTowers->begin(); cal != caloTowers->end(); ++ cal ){
    // std::cout << "ieta/iphi = " <<  cal->ieta() << " / "  << cal->iphi() << std::endl;
    // std::cout << "hadEnergy/outEnergy = " <<  cal->hadEnergy() << " / " << cal->outerEnergy() << std::endl;
    //this next line causes crashing, pointers are probably wrong
    // std::cout << "hadEnergyHeOut/hadEnergyHeInn = " << cal->hadEnergyHeOuterLayer() << " / " << cal->hadEnergyHeInnerLayer() << std::endl;
    // std::cout << "emEnergy = " << cal->emEnergy() << std::endl;
    // std::cout << "ecalTime/hcalTime = " << cal->ecalTime() << " / " << cal->hcalTime() << std::endl;
    // std::cout << "zide = " << cal->zside() << std::endl;

    //To compensate for the index
    if (cal->ieta() >> 0 ){mHadEnergyProfile->Fill (cal->ieta()-1, cal->iphi(), cal->hadEnergy());
    mEmEnergyProfile->Fill (cal->ieta()-1, cal->iphi(), cal->emEnergy());}
    mHadEnergyProfile->Fill (cal->ieta(), cal->iphi(), cal->hadEnergy());
    mEmEnergyProfile->Fill (cal->ieta(), cal->iphi(), cal->emEnergy());

    mHadTiming->Fill (cal->hcalTime());
    mEmTiming->Fill (cal->ecalTime());
    
  }


  math::XYZTLorentzVector p4tmp[2];

  Handle<CaloJetCollection> caloJets;
  mEvent.getByLabel(mInputCollection, caloJets);
  CaloJetCollection::const_iterator jet = caloJets->begin ();
  int jetIndex = 0;
  int nJet = 0;
  for (; jet != caloJets->end (); jet++, jetIndex++) {
    if (mEta) mEta->Fill (jet->eta());
    if (jet->pt() > 30.) {
      if (mEtaFineBin) mEtaFineBin->Fill (jet->eta());
      if (mPhiFineBin) mPhiFineBin->Fill (jet->phi());
    }
    if (mPhi) mPhi->Fill (jet->phi());
    if (mE) mE->Fill (jet->energy());
    if (mE_80) mE_80->Fill (jet->energy());
    if (mE_3000) mE_3000->Fill (jet->energy());
    if (mP) mP->Fill (jet->p());
    if (mP_80) mP_80->Fill (jet->p());
    if (mP_3000) mP_3000->Fill (jet->p());
    if (mPt) mPt->Fill (jet->pt());
    if (mPt_80) mPt_80->Fill (jet->pt());
    if (mPt_3000) mPt_3000->Fill (jet->pt());
    if (mMass) mMass->Fill (jet->mass());
    if (mMass_80) mMass_80->Fill (jet->mass());
    if (mMass_3000) mMass_3000->Fill (jet->mass());
    if (mConstituents) mConstituents->Fill (jet->nConstituents());
    if (mConstituents_80) mConstituents_80->Fill (jet->nConstituents());
    if (jet == caloJets->begin ()) { // first jet
      if (mEtaFirst) mEtaFirst->Fill (jet->eta());
      if (mPhiFirst) mPhiFirst->Fill (jet->phi());
      if (mEFirst) mEFirst->Fill (jet->energy());
      if (mEFirst_80) mEFirst_80->Fill (jet->energy());
      if (mEFirst_3000) mEFirst_3000->Fill (jet->energy());
      if (mPtFirst) mPtFirst->Fill (jet->pt());
    }
    if (jetIndex == 0) {
      nJet++;
      p4tmp[0] = jet->p4();     
    }
    if (jetIndex == 1) {
      nJet++;
      p4tmp[1] = jet->p4();     
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

    if (mHFTotal)      mHFTotal->Fill (jet->hadEnergyInHF()+jet->emEnergyInHF());
    if (mHFTotal_80)   mHFTotal_80->Fill (jet->hadEnergyInHF()+jet->emEnergyInHF());
    if (mHFTotal_3000) mHFTotal_3000->Fill (jet->hadEnergyInHF()+jet->emEnergyInHF());
    if (mHFLong)       mHFLong->Fill (jet->hadEnergyInHF()*0.5+jet->emEnergyInHF());
    if (mHFLong_80)    mHFLong_80->Fill (jet->hadEnergyInHF()*0.5+jet->emEnergyInHF());
    if (mHFLong_3000)  mHFLong_3000->Fill (jet->hadEnergyInHF()*0.5+jet->emEnergyInHF());
    if (mHFShort)      mHFShort->Fill (jet->hadEnergyInHF()*0.5);
    if (mHFShort_80)   mHFShort_80->Fill (jet->hadEnergyInHF()*0.5);
    if (mHFShort_3000) mHFShort_3000->Fill (jet->hadEnergyInHF()*0.5);

    if (mN90) mN90->Fill (jet->n90());
    mJetEnergyProfile->Fill (jet->eta(), jet->phi(), jet->energy());
    mHadJetEnergyProfile->Fill (jet->eta(), jet->phi(), jet->hadEnergyInHO()+jet->hadEnergyInHB()+jet->hadEnergyInHF()+jet->hadEnergyInHE());
    mEMJetEnergyProfile->Fill (jet->eta(), jet->phi(), jet->emEnergyInEB()+jet->emEnergyInEE()+jet->emEnergyInHF());
  }
  if (nJet == 2) {
    if (mMjj) mMjj->Fill( (p4tmp[0]+p4tmp[1]).mass() );
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
