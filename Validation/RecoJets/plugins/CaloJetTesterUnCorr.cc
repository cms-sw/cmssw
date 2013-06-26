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

#include "RecoJets/JetProducers/interface/JetMatchingTools.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "CaloJetTesterUnCorr.h"

#include <cmath>

using namespace edm;
using namespace reco;
using namespace std;

namespace {
  bool is_B (const reco::Jet& fJet) {return fabs (fJet.eta()) < 1.3;}
  bool is_E (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 1.3 && fabs (fJet.eta()) < 3.;}
  bool is_F (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 3.;}
}

CaloJetTesterUnCorr::CaloJetTesterUnCorr(const edm::ParameterSet& iConfig)
  : mInputCollection (iConfig.getParameter<edm::InputTag>( "src" )),
    mInputGenCollection (iConfig.getParameter<edm::InputTag>( "srcGen" )),
    rho_tag_ (iConfig.getParameter<edm::InputTag>( "srcRho" )),
    mOutputFile (iConfig.getUntrackedParameter<std::string>("outputFile", "")),
    mMatchGenPtThreshold (iConfig.getParameter<double>("genPtThreshold")),
    mGenEnergyFractionThreshold (iConfig.getParameter<double>("genEnergyFractionThreshold")),
    mReverseEnergyFractionThreshold (iConfig.getParameter<double>("reverseEnergyFractionThreshold")),
    mRThreshold (iConfig.getParameter<double>("RThreshold")),
    mTurnOnEverything (iConfig.getUntrackedParameter<std::string>("TurnOnEverything",""))
{
    numberofevents
    = mEta = mEtaFineBin = mPhi = mPhiFineBin = mE = mE_80 
    = mP = mP_80 = mPt = mPt_80
    = mMass = mMass_80 = mConstituents = mConstituents_80
    = mEtaFirst = mPhiFirst = mPtFirst = mPtFirst_80 = mPtFirst_3000
    = mMjj = mMjj_3000 = mDelEta = mDelPhi = mDelPt 
    = mMaxEInEmTowers = mMaxEInHadTowers 
    = mHadEnergyInHO = mHadEnergyInHB = mHadEnergyInHF = mHadEnergyInHE 
    = mHadEnergyInHO_80 = mHadEnergyInHB_80 = mHadEnergyInHE_80 
    = mHadEnergyInHO_3000 
    = mEmEnergyInEB = mEmEnergyInEE = mEmEnergyInHF 
    = mEmEnergyInEB_80 = mEmEnergyInEE_80
    = mEnergyFractionHadronic_B = mEnergyFractionHadronic_E = mEnergyFractionHadronic_F
    = mEnergyFractionEm_B = mEnergyFractionEm_E = mEnergyFractionEm_F 
    = mHFLong = mHFTotal = mHFLong_80  = mHFShort = mHFShort_80 
    = mN90
      ///= mCaloMEx = mCaloMEx_3000 = mCaloMEy = mCaloMEy_3000 = mCaloMETSig = mCaloMETSig_3000
      //= mCaloMET = mCaloMET_3000 =  mCaloMETPhi = mCaloSumET  = mCaloSumET_3000   
    = mHadTiming = mEmTiming 
    = mNJetsEtaC = mNJetsEtaF = mNJets1 = mNJets2
      //= mAllGenJetsPt = mMatchedGenJetsPt = mAllGenJetsEta = mMatchedGenJetsEta 
      //= mGenJetMatchEnergyFraction = mReverseMatchEnergyFraction = mRMatch
      = mDeltaEta = mDeltaPhi //= mEScale = mlinEScale = mDeltaE
      //= mHadEnergyProfile = mEmEnergyProfile = mJetEnergyProfile = mHadJetEnergyProfile = mEMJetEnergyProfile
    = mEScale_pt10 = mEScaleFineBin
      //= mpTScaleB_s = mpTScaleE_s = mpTScaleF_s 
    = mpTScaleB_d = mpTScaleE_d = mpTScaleF_d
    = mpTScalePhiB_d = mpTScalePhiE_d = mpTScalePhiF_d
      //= mpTScale_30_200_s = mpTScale_200_600_s = mpTScale_600_1500_s = mpTScale_1500_3500_s
    = mpTScale_30_200_d = mpTScale_200_600_d = mpTScale_600_1500_d = mpTScale_1500_3500_d
      
    = mpTScale1DB_30_200    = mpTScale1DE_30_200    = mpTScale1DF_30_200 
    = mpTScale1DB_200_600   = mpTScale1DE_200_600   = mpTScale1DF_200_600 
    = mpTScale1DB_600_1500   = mpTScale1DE_600_1500   = mpTScale1DF_600_1500 
    = mpTScale1DB_1500_3500 = mpTScale1DE_1500_3500 = mpTScale1DF_1500_3500
    /*
    = mpTScale1D_30_200 = mpTScale1D_200_600 = mpTScale1D_600_1500 = mpTScale1D_1500_3500
    = mHBEne = mHBTime = mHEEne = mHETime = mHFEne = mHFTime = mHOEne = mHOTime
    = mEBEne = mEBTime = mEEEne = mEETime
      */

    = mPthat_80 = mPthat_3000 =mjetArea =mRho
    = 0;
  
  DQMStore* dbe = &*edm::Service<DQMStore>();
  if (dbe) {
    dbe->setCurrentFolder("JetMET/RecoJetsV/CaloJetTask_" + mInputCollection.label());
    //
    numberofevents    = dbe->book1D("numberofevents","numberofevents", 3, 0 , 2);
    //
    mEta              = dbe->book1D("Eta", "Eta", 120, -6, 6); 
    mEtaFineBin       = dbe->book1D("EtaFineBin_Pt10", "EtaFineBin_Pt10", 600, -6, 6);
    /*
    mEtaFineBin1p     = dbe->book1D("EtaFineBin1p_Pt10", "EtaFineBin1p_Pt10", 100, 0, 1.3); 
    mEtaFineBin2p     = dbe->book1D("EtaFineBin2p_Pt10", "EtaFineBin2p_Pt10", 100, 1.3, 3); 
    mEtaFineBin3p     = dbe->book1D("EtaFineBin3p_Pt10", "EtaFineBin3p_Pt10", 100, 3, 5); 
    mEtaFineBin1m     = dbe->book1D("EtaFineBin1m_Pt10", "EtaFineBin1m_Pt10", 100, -1.3, 0); 
    mEtaFineBin2m     = dbe->book1D("EtaFineBin2m_Pt10", "EtaFineBin2m_Pt10", 100, -3, -1.3); 
    mEtaFineBin3m     = dbe->book1D("EtaFineBin3m_Pt10", "EtaFineBin3m_Pt10", 100, -5, -3); 
    */
    //
    mPhi              = dbe->book1D("Phi", "Phi", 70, -3.5, 3.5); 
    mPhiFineBin       = dbe->book1D("PhiFineBin_Pt10", "PhiFineBin_Pt10", 350, -3.5, 3.5); 
    //
    mE                = dbe->book1D("E", "E", 100, 0, 500); 
    mE_80             = dbe->book1D("E_80", "E_80", 100, 0, 5000); 
    //
    mP                = dbe->book1D("P", "P", 100, 0, 500); 
    mP_80             = dbe->book1D("P_80", "P_80", 100, 0, 5000); 
    //
    mPt               = dbe->book1D("Pt", "Pt", 100, 0, 150); 
    mPt_80            = dbe->book1D("Pt_80", "Pt_80", 100, 0, 4000);  
    //
    mMass             = dbe->book1D("Mass", "Mass", 100, 0, 200); 
    mMass_80          = dbe->book1D("Mass_80", "Mass_80", 100, 0, 500);  
    //
    mConstituents     = dbe->book1D("Constituents", "# of Constituents", 100, 0, 100); 
    mConstituents_80  = dbe->book1D("Constituents_80", "# of Constituents_80", 40, 0, 40); 
    //
    mEtaFirst         = dbe->book1D("EtaFirst", "EtaFirst", 120, -6, 6); 
    mPhiFirst         = dbe->book1D("PhiFirst", "PhiFirst", 70, -3.5, 3.5);      
    mPtFirst          = dbe->book1D("PtFirst", "PtFirst", 100, 0, 50); 
    mPtFirst_80       = dbe->book1D("PtFirst_80", "PtFirst_80", 100, 0, 140);
    mPtFirst_3000     = dbe->book1D("PtFirst_3000", "PtFirst_3000", 100, 0, 4000);
    //
    mMjj              = dbe->book1D("Mjj", "Mjj", 100, 0, 2000); 
    mMjj_3000         = dbe->book1D("Mjj_3000", "Mjj_3000", 100, 0, 10000); 
    mDelEta           = dbe->book1D("DelEta", "DelEta", 100, -.5, .5); 
    mDelPhi           = dbe->book1D("DelPhi", "DelPhi", 100, -.5, .5); 
    mDelPt            = dbe->book1D("DelPt", "DelPt", 100, -1, 1); 
    //
    mMaxEInEmTowers   = dbe->book1D("MaxEInEmTowers", "MaxEInEmTowers", 100, 0, 100); 
    mMaxEInHadTowers  = dbe->book1D("MaxEInHadTowers", "MaxEInHadTowers", 100, 0, 100); 
    mHadEnergyInHO    = dbe->book1D("HadEnergyInHO", "HadEnergyInHO", 100, 0, 10); 
    mHadEnergyInHB    = dbe->book1D("HadEnergyInHB", "HadEnergyInHB", 100, 0, 150); 
    mHadEnergyInHF    = dbe->book1D("HadEnergyInHF", "HadEnergyInHF", 100, 0, 50); 
    mHadEnergyInHE    = dbe->book1D("HadEnergyInHE", "HadEnergyInHE", 100, 0, 150); 
    //
    mHadEnergyInHO_80    = dbe->book1D("HadEnergyInHO_80", "HadEnergyInHO_80", 100, 0, 50); 
    mHadEnergyInHB_80    = dbe->book1D("HadEnergyInHB_80", "HadEnergyInHB_80", 100, 0, 3000); 
    mHadEnergyInHE_80    = dbe->book1D("HadEnergyInHE_80", "HadEnergyInHE_80", 100, 0, 3000); 
    mHadEnergyInHO_3000  = dbe->book1D("HadEnergyInHO_3000", "HadEnergyInHO_3000", 100, 0, 500);  
    //
    mEmEnergyInEB     = dbe->book1D("EmEnergyInEB", "EmEnergyInEB", 100, 0, 150); 
    mEmEnergyInEE     = dbe->book1D("EmEnergyInEE", "EmEnergyInEE", 100, 0, 150); 
    mEmEnergyInHF     = dbe->book1D("EmEnergyInHF", "EmEnergyInHF", 120, -20, 100); 
    mEmEnergyInEB_80  = dbe->book1D("EmEnergyInEB_80", "EmEnergyInEB_80", 100, 0, 3000); 
    mEmEnergyInEE_80  = dbe->book1D("EmEnergyInEE_80", "EmEnergyInEE_80", 100, 0, 3000); 
    mEnergyFractionHadronic_B = dbe->book1D("EnergyFractionHadronic_B", "EnergyFractionHadronic_B", 120, -0.1, 1.1);
    mEnergyFractionHadronic_E = dbe->book1D("EnergyFractionHadronic_E", "EnergyFractionHadronic_E", 120, -0.1, 1.1);
    mEnergyFractionHadronic_F = dbe->book1D("EnergyFractionHadronic_F", "EnergyFractionHadronic_F", 120, -0.1, 1.1);
    mEnergyFractionEm_B = dbe->book1D("EnergyFractionEm_B", "EnergyFractionEm_B", 120, -0.1, 1.1);
    mEnergyFractionEm_E = dbe->book1D("EnergyFractionEm_E", "EnergyFractionEm_E", 120, -0.1, 1.1);
    mEnergyFractionEm_F = dbe->book1D("EnergyFractionEm_F", "EnergyFractionEm_F", 120, -0.1, 1.1); 
    //
    mHFTotal          = dbe->book1D("HFTotal", "HFTotal", 100, 0, 150);
    mHFTotal_80       = dbe->book1D("HFTotal_80", "HFTotal_80", 100, 0, 3000);

    mHFLong           = dbe->book1D("HFLong", "HFLong", 100, 0, 150);
    mHFLong_80        = dbe->book1D("HFLong_80", "HFLong_80", 100, 0, 3000);
 
    mHFShort          = dbe->book1D("HFShort", "HFShort", 100, 0, 150);
    mHFShort_80       = dbe->book1D("HFShort_80", "HFShort_80", 100, 0, 3000);

    //
    mN90              = dbe->book1D("N90", "N90", 50, 0, 50); 
    //
    mGenEta           = dbe->book1D("GenEta", "GenEta", 120, -6, 6);
    mGenPhi           = dbe->book1D("GenPhi", "GenPhi", 70, -3.5, 3.5);
    mGenPt            = dbe->book1D("GenPt", "GenPt", 100, 0, 150);
    mGenPt_80         = dbe->book1D("GenPt_80", "GenPt_80", 100, 0, 1500);
    //
    mGenEtaFirst      = dbe->book1D("GenEtaFirst", "GenEtaFirst", 120, -6, 6);
    mGenPhiFirst      = dbe->book1D("GenPhiFirst", "GenPhiFirst", 70, -3.5, 3.5);
    //
    /*
    mCaloMEx          = dbe->book1D("CaloMEx","CaloMEx",200,-150,150);
    mCaloMEx_3000     = dbe->book1D("CaloMEx_3000","CaloMEx_3000",100,-500,500);
    mCaloMEy          = dbe->book1D("CaloMEy","CaloMEy",200,-150,150);
    mCaloMEy_3000     = dbe->book1D("CaloMEy_3000","CaloMEy_3000",100,-500,500);
    mCaloMETSig       = dbe->book1D("CaloMETSig","CaloMETSig",100,0,15);
    mCaloMETSig_3000  = dbe->book1D("CaloMETSig_3000","CaloMETSig_3000",100,0,50);
    mCaloMET          = dbe->book1D("CaloMET","CaloMET",100,0,150);
    mCaloMET_3000     = dbe->book1D("CaloMET_3000","CaloMET_3000",100,0,1000);
    mCaloMETPhi       = dbe->book1D("CaloMETPhi","CaloMETPhi",70, -3.5, 3.5);
    mCaloSumET        = dbe->book1D("CaloSumET","CaloSumET",100,0,500);
    mCaloSumET_3000   = dbe->book1D("CaloSumET_3000","CaloSumET_3000",100,3000,8000);
    */
    //
    mHadTiming        = dbe->book1D("HadTiming", "HadTiming", 75, -50, 100);
    mEmTiming         = dbe->book1D("EMTiming", "EMTiming", 75, -50, 100);
    //
    mNJetsEtaC        = dbe->book1D("NJetsEtaC_Pt10", "NJetsEtaC_Pt10", 15, 0, 15);
    mNJetsEtaF        = dbe->book1D("NJetsEtaF_Pt10", "NJetsEtaF_Pt10", 15, 0, 15);
    //
    mNJets1           = dbe->bookProfile("NJets1", "NJets1", 100, 0, 200,  100, 0, 50, "s");
    mNJets2           = dbe->bookProfile("NJets2", "NJets2", 100, 0, 4000, 100, 0, 50, "s");
    //
    /*
    mHBEne     = dbe->book1D( "HBEne",  "HBEne", 1000, -20, 100 );
    mHBTime    = dbe->book1D( "HBTime", "HBTime", 200, -200, 200 );
    mHEEne     = dbe->book1D( "HEEne",  "HEEne", 1000, -20, 100 );
    mHETime    = dbe->book1D( "HETime", "HETime", 200, -200, 200 );
    mHOEne     = dbe->book1D( "HOEne",  "HOEne", 1000, -20, 100 );
    mHOTime    = dbe->book1D( "HOTime", "HOTime", 200, -200, 200 );
    mHFEne     = dbe->book1D( "HFEne",  "HFEne", 1000, -20, 100 );
    mHFTime    = dbe->book1D( "HFTime", "HFTime", 200, -200, 200 );
    mEBEne     = dbe->book1D( "EBEne",  "EBEne", 1000, -20, 100 );
    mEBTime    = dbe->book1D( "EBTime", "EBTime", 200, -200, 200 );
    mEEEne     = dbe->book1D( "EEEne",  "EEEne", 1000, -20, 100 );
    mEETime    = dbe->book1D( "EETime", "EETime", 200, -200, 200 );
    */
    //
    mPthat_80            = dbe->book1D("Pthat_80", "Pthat_80", 100, 0.0, 1000.0); 
    mPthat_3000          = dbe->book1D("Pthat_3000", "Pthat_3000", 100, 1000.0, 4000.0); 


    mjetArea = dbe->book1D("jetArea","jetArea",26,-0.5,12.5);
    mRho = dbe->book1D("Rho","Rho",20,0,20);

    //
    double log10PtMin = 0.5; //=3.1622766
    double log10PtMax = 3.75; //=5623.41325
    int log10PtBins = 26; 
    //double etaMin = -6.;
    //double etaMax = 6.;
    //int etaBins = 50;
    double etaRange[91] = {-6.0,-5.8,-5.6,-5.4,-5.2,-5.0,-4.8,-4.6,-4.4,-4.2,-4.0,-3.8,-3.6,-3.4,-3.2,-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0};

    //double linPtMin = 5;
    //double linPtMax = 155;
    //int linPtBins = 15;

   // int log10PtFineBins = 50;
    /*
    mAllGenJetsPt = dbe->book1D("GenJetLOGpT", "GenJet LOG(pT_gen)", 
				log10PtBins, log10PtMin, log10PtMax);
    mMatchedGenJetsPt = dbe->book1D("MatchedGenJetLOGpT", "MatchedGenJet LOG(pT_gen)", 
				    log10PtBins, log10PtMin, log10PtMax);
    mAllGenJetsEta = dbe->book2D("GenJetEta", "GenJet Eta vs LOG(pT_gen)", 
				 log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax);
    mMatchedGenJetsEta = dbe->book2D("MatchedGenJetEta", "MatchedGenJet Eta vs LOG(pT_gen)", 
				     log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax);
    */
    //
    if (mTurnOnEverything.compare("yes")==0) {
      /*
      mHadEnergyProfile = dbe->bookProfile2D("HadEnergyProfile", "HadEnergyProfile", 82, -41, 41, 73, 0, 73, 100, 0, 10000, "s");
      mEmEnergyProfile  = dbe->bookProfile2D("EmEnergyProfile", "EmEnergyProfile", 82, -41, 41, 73, 0, 73, 100, 0, 10000, "s");
      */
    }      
      /*
    mJetEnergyProfile = dbe->bookProfile2D("JetEnergyProfile", "JetEnergyProfile", 50, -5, 5, 36, -3.1415987, 3.1415987, 100, 0, 10000, "s");
    mHadJetEnergyProfile = dbe->bookProfile2D("HadJetEnergyProfile", "HadJetEnergyProfile", 50, -5, 5, 36, -3.1415987, 3.1415987, 100, 0, 10000, "s");
    mEMJetEnergyProfile = dbe->bookProfile2D("EMJetEnergyProfile", "EMJetEnergyProfile", 50, -5, 5, 36, -3.1415987, 3.1415987, 100, 0, 10000, "s");
      */
    //
    if (mTurnOnEverything.compare("yes")==0) {
      /*
    mGenJetMatchEnergyFraction  = dbe->book3D("GenJetMatchEnergyFraction", "GenJetMatchEnergyFraction vs LOG(pT_gen) vs eta", 
					      log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 101, 0, 1.01);
    mReverseMatchEnergyFraction  = dbe->book3D("ReverseMatchEnergyFraction", "ReverseMatchEnergyFraction vs LOG(pT_gen) vs eta", 
					       log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 101, 0, 1.01);
    mRMatch  = dbe->book3D("RMatch", "delta(R)(Gen-Calo) vs LOG(pT_gen) vs eta", 
			   log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 30, 0, 3);
      */
/*
    mDeltaEta = dbe->book3D("DeltaEta", "DeltaEta vs LOG(pT_gen) vs eta", 
			      log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, -1, 1);
    mDeltaPhi = dbe->book3D("DeltaPhi", "DeltaPhi vs LOG(pT_gen) vs eta", 
			      log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, -1, 1);
*/  
  /*
    mEScale = dbe->book3D("EScale", "EnergyScale vs LOG(pT_gen) vs eta", 
			    log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 100, 0, 2);
    mlinEScale = dbe->book3D("linEScale", "EnergyScale vs LOG(pT_gen) vs eta", 
			    linPtBins, linPtMin, linPtMax, etaBins, etaMin, etaMax, 100, 0, 2);
    mDeltaE = dbe->book3D("DeltaE", "DeltaE vs LOG(pT_gen) vs eta", 
			    log10PtBins, log10PtMin, log10PtMax, etaBins, etaMin, etaMax, 2000, -200, 200);
    */
    //
   /*
    mEScale_pt10 = dbe->book3D("EScale_pt10", "EnergyScale vs LOG(pT_gen) vs eta", 
			    log10PtBins, log10PtMin, log10PtMax, 90,etaRange, 100, 0, 2);
    mEScaleFineBin = dbe->book3D("EScaleFineBins", "EnergyScale vs LOG(pT_gen) vs eta", 
			    log10PtFineBins, log10PtMin, log10PtMax, 90,etaRange, 100, 0, 2);
*/
    }
    //mpTScaleB_s = dbe->bookProfile("pTScaleB_s", "pTScale_s_0<|eta|<1.3",
    //				    log10PtBins, log10PtMin, log10PtMax, 0, 2, "s");
    //mpTScaleE_s = dbe->bookProfile("pTScaleE_s", "pTScale_s_1.3<|eta|<3.0",
    //				    log10PtBins, log10PtMin, log10PtMax, 0, 2, "s");
  // mpTScaleF_s = dbe->bookProfile("pTScaleF_s", "pTScale_s_3.0<|eta|<5.0",
    //				    log10PtBins, log10PtMin, log10PtMax, 0, 2, "s");
    mpTScaleB_d = dbe->bookProfile("pTScaleB_d", "pTScale_d_0<|eta|<1.5",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScaleE_d = dbe->bookProfile("pTScaleE_d", "pTScale_d_1.5<|eta|<3.0",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScaleF_d = dbe->bookProfile("pTScaleF_d", "pTScale_d_3.0<|eta|<6.0",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScalePhiB_d = dbe->bookProfile("pTScalePhiB_d", "pTScalePhi_d_0<|eta|<1.5",
				   70, -3.5, 3.5, 0, 2, " ");
    mpTScalePhiE_d = dbe->bookProfile("pTScalePhiE_d", "pTScalePhi_d_1.5<|eta|<3.0",
				   70, -3.5, 3.5, 0, 2, " ");
    mpTScalePhiF_d = dbe->bookProfile("pTScalePhiF_d", "pTScalePhi_d_3.0<|eta|<6.0",
				   70, -3.5, 3.5, 0, 2, " ");
//mpTScale_30_200_s    = dbe->bookProfile("pTScale_30_200_s", "pTScale_s_30<pT<200",
//					  etaBins, etaMin, etaMax, 0., 2., "s");
//mpTScale_200_600_s   = dbe->bookProfile("pTScale_200_600_s", "pTScale_s_200<pT<600",
//					  etaBins, etaMin, etaMax, 0., 2., "s");
//mpTScale_600_1500_s   = dbe->bookProfile("pTScale_600_1500_s", "pTScale_s_600<pT<1500",
//					  etaBins, etaMin, etaMax, 0., 2., "s");
//mpTScale_1500_3500_s = dbe->bookProfile("pTScale_1500_3500_s", "pTScale_s_1500<pt<3500",
//                                          etaBins, etaMin, etaMax, 0., 2., "s");
    mpTScale_30_200_d    = dbe->bookProfile("pTScale_30_200_d", "pTScale_d_30<pT<200",
					  90,etaRange, 0., 2., " ");
    mpTScale_200_600_d   = dbe->bookProfile("pTScale_200_600_d", "pTScale_d_200<pT<600",
					  90,etaRange, 0., 2., " ");
    mpTScale_600_1500_d   = dbe->bookProfile("pTScale_600_1500_d", "pTScale_d_600<pT<1500",
					  90,etaRange, 0., 2., " ");
    mpTScale_1500_3500_d = dbe->bookProfile("pTScale_1500_3500_d", "pTScale_d_1500<pt<3500",
                                          90,etaRange, 0., 2., " ");

    mpTScale1DB_30_200 = dbe->book1D("pTScale1DB_30_200", "pTScale_distribution_for_0<|eta|<1.5_30_200",
				   100, 0, 2);
    mpTScale1DE_30_200 = dbe->book1D("pTScale1DE_30_200", "pTScale_distribution_for_1.5<|eta|<3.0_30_200",
				   50, 0, 2);
    mpTScale1DF_30_200 = dbe->book1D("pTScale1DF_30_200", "pTScale_distribution_for_3.0<|eta|<6.0_30_200",
				   50, 0, 2);

    mpTScale1DB_200_600 = dbe->book1D("pTScale1DB_200_600", "pTScale_distribution_for_0<|eta|<1.5_200_600",
				   100, 0, 2);
    mpTScale1DE_200_600 = dbe->book1D("pTScale1DE_200_600", "pTScale_distribution_for_1.5<|eta|<3.0_200_600",
				   50, 0, 2);
    mpTScale1DF_200_600 = dbe->book1D("pTScale1DF_200_600", "pTScale_distribution_for_3.0<|eta|<6.0_200_600",
				   50, 0, 2);

    mpTScale1DB_600_1500 = dbe->book1D("pTScale1DB_600_1500", "pTScale_distribution_for_0<|eta|<1.5_600_1500",
				   100, 0, 2);
    mpTScale1DE_600_1500 = dbe->book1D("pTScale1DE_600_1500", "pTScale_distribution_for_1.5<|eta|<3.0_600_1500",
				   50, 0, 2);
    mpTScale1DF_600_1500 = dbe->book1D("pTScale1DF_600_1500", "pTScale_distribution_for_3.0<|eta|<6.0_600_1500",
				   50, 0, 2);

    mpTScale1DB_1500_3500 = dbe->book1D("pTScale1DB_1500_3500", "pTScale_distribution_for_0<|eta|<1.5_1500_3500",
				   100, 0, 2);
    mpTScale1DE_1500_3500 = dbe->book1D("pTScale1DE_1500_3500", "pTScale_distribution_for_1.5<|eta|<3.0_1500_3500",
				   50, 0, 2);
    mpTScale1DF_1500_3500 = dbe->book1D("pTScale1DF_1500_3500", "pTScale_distribution_for_3.0<|eta|<6.0_1500_3500",
				   50, 0, 2);
/*
    mpTScale1D_30_200    = dbe->book1D("pTScale1D_30_200", "pTScale_distribution_for_30<pT<200",
					    100, 0, 2);
    mpTScale1D_200_600    = dbe->book1D("pTScale1D_200_600", "pTScale_distribution_for_200<pT<600",
					    100, 0, 2);
    mpTScale1D_600_1500    = dbe->book1D("pTScale1D_600_1500", "pTScale_distribution_for_600<pT<1500",
					    100, 0, 2);
    mpTScale1D_1500_3500 = dbe->book1D("pTScale1D_1500_3500", "pTScale_distribution_for_1500<pt<3500",
					    100, 0, 2);
*/
  }
  
  if (mOutputFile.empty ()) {
    LogInfo("OutputInfo") << " CaloJet histograms will NOT be saved";
  } 
  else {
    LogInfo("OutputInfo") << " CaloJethistograms will be saved to file:" << mOutputFile;
  }
}
   
CaloJetTesterUnCorr::~CaloJetTesterUnCorr()
{
}

void CaloJetTesterUnCorr::beginJob(){
}

void CaloJetTesterUnCorr::endJob() {
 if (!mOutputFile.empty() && &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (mOutputFile);
}


void CaloJetTesterUnCorr::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  double countsfornumberofevents = 1;
  numberofevents->Fill(countsfornumberofevents);
  // *********************************
  // *** Get pThat
  // *********************************
if (!mEvent.isRealData()){
  edm::Handle<HepMCProduct> evt;
  mEvent.getByLabel("generator", evt);
  if (evt.isValid()) {
  HepMC::GenEvent * myGenEvent = new HepMC::GenEvent(*(evt->GetEvent()));
  
  double pthat = myGenEvent->event_scale();

  mPthat_80->Fill(pthat);
  mPthat_3000->Fill(pthat);

  delete myGenEvent; 
  }
}
  // ***********************************
  // *** Get CaloMET
  // ***********************************
/*
  const CaloMET *calomet;
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
    mCaloMEx_3000->Fill(caloMEx);
    mCaloMEy->Fill(caloMEy);
    mCaloMEy_3000->Fill(caloMEy);
    mCaloMET->Fill(caloMET);
    mCaloMET_3000->Fill(caloMET);
    mCaloMETPhi->Fill(caloMETPhi);
    mCaloSumET->Fill(caloSumET);
    mCaloSumET_3000->Fill(caloSumET);
    mCaloMETSig->Fill(caloMETSig);
    mCaloMETSig_3000->Fill(caloMETSig);
    
  }
*/
  // ***********************************
  // *** Get the CaloTower collection
  // ***********************************
  Handle<CaloTowerCollection> caloTowers;
  mEvent.getByLabel( "towerMaker", caloTowers );
  if (caloTowers.isValid()) {
  for( CaloTowerCollection::const_iterator cal = caloTowers->begin(); cal != caloTowers->end(); ++ cal ){

    //To compensate for the index
    if (mTurnOnEverything.compare("yes")==0) {
      /*
      if (cal->ieta() >> 0 ){mHadEnergyProfile->Fill (cal->ieta()-1, cal->iphi(), cal->hadEnergy());
      mEmEnergyProfile->Fill (cal->ieta()-1, cal->iphi(), cal->emEnergy());}
      mHadEnergyProfile->Fill (cal->ieta(), cal->iphi(), cal->hadEnergy()); 
      mEmEnergyProfile->Fill (cal->ieta(), cal->iphi(), cal->emEnergy());
      */
      }

    mHadTiming->Fill (cal->hcalTime());
    mEmTiming->Fill (cal->ecalTime());    
  }
  }
  
  // ***********************************
  // *** Get the RecHits collection
  // ***********************************
  try {
    std::vector<edm::Handle<HBHERecHitCollection> > colls;
    mEvent.getManyByType(colls);
    std::vector<edm::Handle<HBHERecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HBHERecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        //      std::cout << *j << std::endl;
	/*
        if (j->id().subdet() == HcalBarrel) {
          mHBEne->Fill(j->energy()); 
          mHBTime->Fill(j->time()); 
        }
        if (j->id().subdet() == HcalEndcap) {
          mHEEne->Fill(j->energy()); 
          mHETime->Fill(j->time()); 
        }
	*/
      }
    }
  } catch (...) {
    edm::LogInfo("OutputInfo") << " No HB/HE RecHits.";
  }

  try {
    std::vector<edm::Handle<HFRecHitCollection> > colls;
    mEvent.getManyByType(colls);
    std::vector<edm::Handle<HFRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HFRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        //      std::cout << *j << std::endl;
	/*
        if (j->id().subdet() == HcalForward) {
          mHFEne->Fill(j->energy()); 
          mHFTime->Fill(j->time()); 
        }
	*/
      }
    }
  } catch (...) {
    edm::LogInfo("OutputInfo") << " No HF RecHits.";
  }

  try {
    std::vector<edm::Handle<HORecHitCollection> > colls;
    mEvent.getManyByType(colls);
    std::vector<edm::Handle<HORecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (HORecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
	/*
        if (j->id().subdet() == HcalOuter) {
          //mHOEne->Fill(j->energy()); 
          //mHOTime->Fill(j->time()); 
        }
	*/
      }
    }
  } catch (...) {
    edm::LogInfo("OutputInfo") << " No HO RecHits.";
  }
  try {
    std::vector<edm::Handle<EBRecHitCollection> > colls;
    mEvent.getManyByType(colls);
    std::vector<edm::Handle<EBRecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (EBRecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        //      if (j->id() == EcalBarrel) {
	//mEBEne->Fill(j->energy()); 
	//mEBTime->Fill(j->time()); 
	//    }
        //      std::cout << *j << std::endl;
        //      std::cout << j->id() << std::endl;
      }
    }
  } catch (...) {
    edm::LogInfo("OutputInfo") << " No EB RecHits.";
  }

  try {
    std::vector<edm::Handle<EERecHitCollection> > colls;
    mEvent.getManyByType(colls);
    std::vector<edm::Handle<EERecHitCollection> >::iterator i;
    for (i=colls.begin(); i!=colls.end(); i++) {
      for (EERecHitCollection::const_iterator j=(*i)->begin(); j!=(*i)->end(); j++) {
        //      if (j->id().subdet() == EcalEndcap) {
	//mEEEne->Fill(j->energy()); 
	//mEETime->Fill(j->time()); 
	//    }
	//      std::cout << *j << std::endl;
      }
    }
  } catch (...) {
    edm::LogInfo("OutputInfo") << " No EE RecHits.";
  }


  //***********************************
  //*** Get the Jet Rho
  //***********************************
  edm::Handle<double> pRho;
  mEvent.getByLabel(rho_tag_,pRho);
  if( pRho.isValid() ) {
    double rho_ = *pRho;
    if(mRho) mRho->Fill(rho_);
  }

  //***********************************
  //*** Get the Jet collection
  //***********************************
  math::XYZTLorentzVector p4tmp[2];
  Handle<CaloJetCollection> caloJets;
  mEvent.getByLabel(mInputCollection, caloJets);
  if (!caloJets.isValid()) return;
  CaloJetCollection::const_iterator jet = caloJets->begin ();
  int jetIndex = 0;
  int nJet = 0;
  int nJetF = 0;
  int nJetC = 0;
  for (; jet != caloJets->end (); jet++, jetIndex++) {
    

    if (jet->pt() > 10.) {
      if (fabs(jet->eta()) > 1.5) 
	nJetF++;
      else 
	nJetC++;	  
    }
    if (jet->pt() > 10.) {
      if (mEta) mEta->Fill (jet->eta());
      if (mEtaFineBin) mEtaFineBin->Fill (jet->eta());
      //if (mEtaFineBin1p) mEtaFineBin1p->Fill (jet->eta());
      //if (mEtaFineBin2p) mEtaFineBin2p->Fill (jet->eta());
      //if (mEtaFineBin3p) mEtaFineBin3p->Fill (jet->eta());
      //if (mEtaFineBin1m) mEtaFineBin1m->Fill (jet->eta());
      //if (mEtaFineBin2m) mEtaFineBin2m->Fill (jet->eta());
      //if (mEtaFineBin3m) mEtaFineBin3m->Fill (jet->eta());
      if (mPhiFineBin) mPhiFineBin->Fill (jet->phi());
    }
    if (mjetArea) mjetArea->Fill(jet->jetArea());
    if (mPhi) mPhi->Fill (jet->phi());
    if (mE) mE->Fill (jet->energy());
    if (mE_80) mE_80->Fill (jet->energy());
    if (mP) mP->Fill (jet->p());
    if (mP_80) mP_80->Fill (jet->p());
    if (mPt) mPt->Fill (jet->pt());
    if (mPt_80) mPt_80->Fill (jet->pt());
    if (mMass) mMass->Fill (jet->mass());
    if (mMass_80) mMass_80->Fill (jet->mass());
    if (mConstituents) mConstituents->Fill (jet->nConstituents());
    if (mConstituents_80) mConstituents_80->Fill (jet->nConstituents());
    if (jet == caloJets->begin ()) { // first jet
      if (mEtaFirst) mEtaFirst->Fill (jet->eta());
      if (mPhiFirst) mPhiFirst->Fill (jet->phi());
      if (mPtFirst) mPtFirst->Fill (jet->pt());
      if (mPtFirst_80) mPtFirst_80->Fill (jet->pt());
      if (mPtFirst_3000) mPtFirst_3000->Fill (jet->pt());
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
    if (mHadEnergyInHO_80)   mHadEnergyInHO_80->Fill (jet->hadEnergyInHO());
    if (mHadEnergyInHO_3000) mHadEnergyInHO_3000->Fill (jet->hadEnergyInHO());
    if (mHadEnergyInHB) mHadEnergyInHB->Fill (jet->hadEnergyInHB());
    if (mHadEnergyInHB_80)   mHadEnergyInHB_80->Fill (jet->hadEnergyInHB());
    if (mHadEnergyInHF) mHadEnergyInHF->Fill (jet->hadEnergyInHF());
    if (mHadEnergyInHE) mHadEnergyInHE->Fill (jet->hadEnergyInHE());
    if (mHadEnergyInHE_80)   mHadEnergyInHE_80->Fill (jet->hadEnergyInHE());
    if (mEmEnergyInEB) mEmEnergyInEB->Fill (jet->emEnergyInEB());
    if (mEmEnergyInEB_80)   mEmEnergyInEB_80->Fill (jet->emEnergyInEB());
    if (mEmEnergyInEE) mEmEnergyInEE->Fill (jet->emEnergyInEE());
    if (mEmEnergyInEE_80)   mEmEnergyInEE_80->Fill (jet->emEnergyInEE());
    if (mEmEnergyInHF) mEmEnergyInHF->Fill (jet->emEnergyInHF());
    if (fabs(jet->eta())<1.5) mEnergyFractionHadronic_B->Fill (jet->energyFractionHadronic());
    if (fabs(jet->eta())>1.5 && fabs(jet->eta())<3.0) mEnergyFractionHadronic_E->Fill (jet->energyFractionHadronic());
    if (fabs(jet->eta())>3.0 && fabs(jet->eta())<6.0) mEnergyFractionHadronic_F->Fill (jet->energyFractionHadronic());
    if (fabs(jet->eta())<1.5) mEnergyFractionEm_B->Fill (jet->emEnergyFraction());
    if (fabs(jet->eta())>1.5 && fabs(jet->eta())<3.0) mEnergyFractionEm_E->Fill (jet->emEnergyFraction());
    if (fabs(jet->eta())>3.0 && fabs(jet->eta())<6.0) mEnergyFractionEm_F->Fill (jet->emEnergyFraction());
    if (mHFTotal)      mHFTotal->Fill (jet->hadEnergyInHF()+jet->emEnergyInHF());
    if (mHFTotal_80)   mHFTotal_80->Fill (jet->hadEnergyInHF()+jet->emEnergyInHF());
    if (mHFLong)       mHFLong->Fill (jet->hadEnergyInHF()*0.5+jet->emEnergyInHF());
    if (mHFLong_80)    mHFLong_80->Fill (jet->hadEnergyInHF()*0.5+jet->emEnergyInHF());
    if (mHFShort)      mHFShort->Fill (jet->hadEnergyInHF()*0.5);
    if (mHFShort_80)   mHFShort_80->Fill (jet->hadEnergyInHF()*0.5);



    if (mN90) mN90->Fill (jet->n90());
    /*
    mJetEnergyProfile->Fill (jet->eta(), jet->phi(), jet->energy());
    mHadJetEnergyProfile->Fill (jet->eta(), jet->phi(), jet->hadEnergyInHO()+jet->hadEnergyInHB()+jet->hadEnergyInHF()+jet->hadEnergyInHE());
    mEMJetEnergyProfile->Fill (jet->eta(), jet->phi(), jet->emEnergyInEB()+jet->emEnergyInEE()+jet->emEnergyInHF());
    */
  }



  if (mNJetsEtaC) mNJetsEtaC->Fill( nJetC );
  if (mNJetsEtaF) mNJetsEtaF->Fill( nJetF );

  if (nJet == 2) {
    if (mMjj) mMjj->Fill( (p4tmp[0]+p4tmp[1]).mass() );
    if (mMjj_3000) mMjj_3000->Fill( (p4tmp[0]+p4tmp[1]).mass() );
  }

  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int     njet = 0;
    float ptStep = (istep * (200./100.));

    for ( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {
      if ( cal->pt() > ptStep ) njet++;
    }
    mNJets1->Fill( ptStep, njet );
  }

  for (int istep = 0; istep < 100; ++istep) {
    int     njet = 0;
    float ptStep = (istep * (4000./100.));
    for ( CaloJetCollection::const_iterator cal = caloJets->begin(); cal != caloJets->end(); ++ cal ) {
      if ( cal->pt() > ptStep ) njet++;
    }
    mNJets2->Fill( ptStep, njet );
  }

if (!mEvent.isRealData()){
  // Gen jet analysis
  Handle<GenJetCollection> genJets;
  mEvent.getByLabel(mInputGenCollection, genJets);
  if (!genJets.isValid()) return;
  GenJetCollection::const_iterator gjet = genJets->begin ();
  int gjetIndex = 0;
  for (; gjet != genJets->end (); gjet++, gjetIndex++) {
    if (mGenEta) mGenEta->Fill (gjet->eta());
    if (mGenPhi) mGenPhi->Fill (gjet->phi());
    if (mGenPt) mGenPt->Fill (gjet->pt());
    if (mGenPt_80) mGenPt_80->Fill (gjet->pt());
    if (gjet == genJets->begin ()) { // first jet
      if (mGenEtaFirst) mGenEtaFirst->Fill (gjet->eta());
      if (mGenPhiFirst) mGenPhiFirst->Fill (gjet->phi());
    }
  }


  // now match CaloJets to GenJets
  JetMatchingTools jetMatching (mEvent);
  if (!(mInputGenCollection.label().empty())) {
    //    Handle<GenJetCollection> genJets;
    //    mEvent.getByLabel(mInputGenCollection, genJets);

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

    for (unsigned iGenJet = 0; iGenJet < genJets->size(); ++iGenJet) {               //****************************************************************
    //for (unsigned iGenJet = 0; iGenJet < 1; ++iGenJet) {                           // only FIRST Jet !!!!
      const GenJet& genJet = (*genJets) [iGenJet];
      double genJetPt = genJet.pt();

      //std::cout << iGenJet <<". Genjet: pT = " << genJetPt << "GeV" << std::endl;  //  *****************************************************

      if (fabs(genJet.eta()) > 6.) continue; // out of detector 
      if (genJetPt < mMatchGenPtThreshold) continue; // no low momentum 
      //double logPtGen = log10 (genJetPt);
      //mAllGenJetsPt->Fill (logPtGen);
      //mAllGenJetsEta->Fill (logPtGen, genJet.eta());
      if (caloJets->size() <= 0) continue; // no CaloJets - nothing to match
      if (mRThreshold > 0) {
	unsigned iCaloJetBest = 0;
	double deltaRBest = 999.;
	for (unsigned iCaloJet = 0; iCaloJet < caloJets->size(); ++iCaloJet) {
	  double dR = deltaR (genJet.eta(), genJet.phi(), (*caloJets) [iCaloJet].eta(), (*caloJets) [iCaloJet].phi());
	  if (deltaRBest < mRThreshold && dR < mRThreshold && genJet.pt() > 5.) {
	    /*
	    std::cout << "Yet another matched jet for GenJet pt=" << genJet.pt()
		      << " previous CaloJet pt/dr: " << (*caloJets) [iCaloJetBest].pt() << '/' << deltaRBest
		      << " new CaloJet pt/dr: " << (*caloJets) [iCaloJet].pt() << '/' << dR
		      << std::endl;
	    */
	  }
	  if (dR < deltaRBest) {
	    iCaloJetBest = iCaloJet;
	    deltaRBest = dR;
	  }
	}
	if (mTurnOnEverything.compare("yes")==0) {
	  //mRMatch->Fill (logPtGen, genJet.eta(), deltaRBest);
	}
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
	if (mTurnOnEverything.compare("yes")==0) {
	  //mGenJetMatchEnergyFraction->Fill (logPtGen, genJet.eta(), energyFractionBest);
	}
	if (energyFractionBest > mGenEnergyFractionThreshold) { // good enough
	  double reverseEnergyFraction = jetMatching.overlapEnergyFraction (caloJetConstituents [iCaloJetBest], 
									    genJetConstituents [iGenJet]);
	  if (mTurnOnEverything.compare("yes")==0) {
	    //mReverseMatchEnergyFraction->Fill (logPtGen, genJet.eta(), reverseEnergyFraction);
	  }
	  if (reverseEnergyFraction > mReverseEnergyFractionThreshold) { // Matched
	    fillMatchHists (genJet, (*caloJets) [iCaloJetBest]);
	  }
	}
      }
    }
  }
}

}/////Gen Close

void CaloJetTesterUnCorr::fillMatchHists (const reco::GenJet& fGenJet, const reco::CaloJet& fCaloJet) {
  double logPtGen = log10 (fGenJet.pt());
  double PtGen = fGenJet.pt();
  double PtCalo = fCaloJet.pt();
  //mMatchedGenJetsPt->Fill (logPtGen);
  //mMatchedGenJetsEta->Fill (logPtGen, fGenJet.eta());

  double PtThreshold = 10.;

  if (mTurnOnEverything.compare("yes")==0) {
    mDeltaEta->Fill (logPtGen, fGenJet.eta(), fCaloJet.eta()-fGenJet.eta());
    mDeltaPhi->Fill (logPtGen, fGenJet.eta(), fCaloJet.phi()-fGenJet.phi());
    //mEScale->Fill (logPtGen, fGenJet.eta(), fCaloJet.energy()/fGenJet.energy());
    //mlinEScale->Fill (fGenJet.pt(), fGenJet.eta(), fCaloJet.energy()/fGenJet.energy());
    //mDeltaE->Fill (logPtGen, fGenJet.eta(), fCaloJet.energy()-fGenJet.energy());

    mEScaleFineBin->Fill (logPtGen, fGenJet.eta(), fCaloJet.energy()/fGenJet.energy());
  
    if (fGenJet.pt()>PtThreshold) {
      mEScale_pt10->Fill (logPtGen, fGenJet.eta(), fCaloJet.energy()/fGenJet.energy());

    }

  }
  if (fCaloJet.pt() > PtThreshold) {
    mDelEta->Fill (fGenJet.eta()-fCaloJet.eta());
    mDelPhi->Fill (fGenJet.phi()-fCaloJet.phi());
    mDelPt->Fill  ((fGenJet.pt()-fCaloJet.pt())/fGenJet.pt());
  }

  if (fabs(fGenJet.eta())<1.5) {

    //mpTScaleB_s->Fill (log10(PtGen), PtCalo/PtGen);
    mpTScaleB_d->Fill (log10(PtGen), PtCalo/PtGen);
    mpTScalePhiB_d->Fill (fGenJet.phi(), PtCalo/PtGen);
    
    if (PtGen>30.0 && PtGen<200.0) {
      mpTScale1DB_30_200->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>200.0 && PtGen<600.0) {
      mpTScale1DB_200_600->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>600.0 && PtGen<1500.0) {
      mpTScale1DB_600_1500->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>1500.0 && PtGen<3500.0) {
      mpTScale1DB_1500_3500->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    
  }

  if (fabs(fGenJet.eta())>1.5 && fabs(fGenJet.eta())<3.0) {

    //mpTScaleE_s->Fill (log10(PtGen), PtCalo/PtGen);
    mpTScaleE_d->Fill (log10(PtGen), PtCalo/PtGen);
    mpTScalePhiE_d->Fill (fGenJet.phi(), PtCalo/PtGen);
    
    if (PtGen>30.0 && PtGen<200.0) {
      mpTScale1DE_30_200->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>200.0 && PtGen<600.0) {
      mpTScale1DE_200_600->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>600.0 && PtGen<1500.0) {
      mpTScale1DE_600_1500->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>1500.0 && PtGen<3500.0) {
      mpTScale1DE_1500_3500->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    
  }

  if (fabs(fGenJet.eta())>3.0 && fabs(fGenJet.eta())<6.0) {

    //mpTScaleF_s->Fill (log10(PtGen), PtCalo/PtGen);
    mpTScaleF_d->Fill (log10(PtGen), PtCalo/PtGen);
    mpTScalePhiF_d->Fill (fGenJet.phi(), PtCalo/PtGen);
    
    if (PtGen>30.0 && PtGen<200.0) {
      mpTScale1DF_30_200->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>200.0 && PtGen<600.0) {
      mpTScale1DF_200_600->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>600.0 && PtGen<1500.0) {
      mpTScale1DF_600_1500->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    if (PtGen>1500.0 && PtGen<3500.0) {
      mpTScale1DF_1500_3500->Fill (fCaloJet.pt()/fGenJet.pt());
    }
    
  }

  if (fGenJet.pt()>30.0 && fGenJet.pt()<200.0) {
    //mpTScale_30_200_s->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    mpTScale_30_200_d->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    //mpTScale1D_30_200->Fill (fCaloJet.pt()/fGenJet.pt());
  }

  if (fGenJet.pt()>200.0 && fGenJet.pt()<600.0) {
    //mpTScale_200_600_s->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    mpTScale_200_600_d->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    //mpTScale1D_200_600->Fill (fCaloJet.pt()/fGenJet.pt());
  }

  if (fGenJet.pt()>600.0 && fGenJet.pt()<1500.0) {
    //mpTScale_600_1500_s->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    mpTScale_600_1500_d->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    //mpTScale1D_600_1500->Fill (fCaloJet.pt()/fGenJet.pt());
  }

  if (fGenJet.pt()>1500.0 && fGenJet.pt()<3500.0) {
    //mpTScale_1500_3500_s->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    mpTScale_1500_3500_d->Fill (fGenJet.eta(),fCaloJet.pt()/fGenJet.pt());
    //mpTScale1D_1500_3500->Fill (fCaloJet.pt()/fGenJet.pt());
  }



}

