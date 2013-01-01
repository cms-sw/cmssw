
// Producer for validation histograms for CaloJet objects
// F. Ratnikov, Sept. 7, 2006
// Modified by J F Novak July 10, 2008
// $Id: JPTJetTester.cc,v 1.23 2012/12/02 23:38:40 kovitang Exp $

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/JetReco/interface/JPTJet.h"
#include "DataFormats/JetReco/interface/JPTJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"

#include "RecoJets/JetProducers/interface/JetMatchingTools.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "JPTJetTester.h"

#include <cmath>

#include "JetMETCorrections/Objects/interface/JetCorrector.h"

using namespace edm;
using namespace reco;
using namespace std;

namespace {
  bool is_B (const reco::Jet& fJet) {return fabs (fJet.eta()) < 1.3;}
  bool is_E (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 1.3 && fabs (fJet.eta()) < 3.;}
  bool is_F (const reco::Jet& fJet) {return fabs (fJet.eta()) >= 3.;}
}

JPTJetTester::JPTJetTester(const edm::ParameterSet& iConfig)
  : mInputCollection (iConfig.getParameter<edm::InputTag>( "src" )),
    mInputGenCollection (iConfig.getParameter<edm::InputTag>( "srcGen" )),
    mOutputFile (iConfig.getUntrackedParameter<std::string>("outputFile", "")),
    mMatchGenPtThreshold (iConfig.getParameter<double>("genPtThreshold")),
    mGenEnergyFractionThreshold (iConfig.getParameter<double>("genEnergyFractionThreshold")),
    mReverseEnergyFractionThreshold (iConfig.getParameter<double>("reverseEnergyFractionThreshold")),
    mRThreshold (iConfig.getParameter<double>("RThreshold")),
     JetCorrectionService  (iConfig.getParameter<std::string>  ("JetCorrectionService"  )),
    mTurnOnEverything (iConfig.getUntrackedParameter<std::string>("TurnOnEverything",""))
{
    numberofevents
    = mEta = mEtaFineBin = mPhi = mPhiFineBin = mE = mE_80 
    = mP = mP_80  = mPt = mPt_80 
    = mMass = mMass_80 
    = mEtaFirst = mPhiFirst  = mPtFirst = mPtFirst_80 = mPtFirst_3000
    = mMjj = mMjj_3000 = mDelEta = mDelPhi = mDelPt 
    = mHadTiming = mEmTiming 
    = mNJetsEtaC = mNJetsEtaF = mNJets1 = mNJets2
      = mDeltaEta = mDeltaPhi 
    = mHadEnergyProfile = mEmEnergyProfile = mJetEnergyProfile 
    = mEScale_pt10 = mEScaleFineBin
    = mpTScaleB_d = mpTScaleE_d = mpTScaleF_d
      = mpTScalePhiB_d = mpTScalePhiE_d = mpTScalePhiF_d
    = mpTScale_30_200_d = mpTScale_200_600_d = mpTScale_600_1500_d = mpTScale_1500_3500_d
      
    = mpTScale1DB_30_200    = mpTScale1DE_30_200    = mpTScale1DF_30_200 
    = mpTScale1DB_200_600   = mpTScale1DE_200_600   = mpTScale1DF_200_600 
    = mpTScale1DB_600_1500   = mpTScale1DE_600_1500   = mpTScale1DF_600_1500 
    = mpTScale1DB_1500_3500 = mpTScale1DE_1500_3500 = mpTScale1DF_1500_3500
    = mPthat_80 = mPthat_3000

      //Corr Jet
    = mCorrJetPt = mCorrJetPt =mCorrJetPt_80  =mCorrJetEta =mCorrJetPhi =mpTRatio =mpTResponse
      = mpTRatioB_d = mpTRatioE_d = mpTRatioF_d
      = mpTRatio_30_200_d = mpTRatio_200_600_d = mpTRatio_600_1500_d = mpTRatio_1500_3500_d
      = mpTResponseB_d = mpTResponseE_d = mpTResponseF_d
      = mpTResponse_30_200_d = mpTResponse_200_600_d = mpTResponse_600_1500_d = mpTResponse_1500_3500_d
      = mpTResponse_30_d 
      = nvtx_0_30 = nvtx_0_60 
      = mpTResponse_nvtx_0_5 = mpTResponse_nvtx_5_10 =mpTResponse_nvtx_10_15 
      = mpTResponse_nvtx_15_20 = mpTResponse_nvtx_20_30 = mpTResponse_nvtx_30_inf
      = mpTScale_a_nvtx_0_5 = mpTScale_b_nvtx_0_5 = mpTScale_c_nvtx_0_5 
      = mpTScale_a_nvtx_5_10 = mpTScale_b_nvtx_5_10 = mpTScale_c_nvtx_5_10
      = mpTScale_a_nvtx_10_15 = mpTScale_b_nvtx_10_15 = mpTScale_c_nvtx_10_15
      = mpTScale_a_nvtx_15_20 = mpTScale_b_nvtx_15_20 = mpTScale_c_nvtx_15_20
      = mpTScale_a_nvtx_20_30 = mpTScale_b_nvtx_20_30 = mpTScale_c_nvtx_20_30
      = mpTScale_a_nvtx_30_inf = mpTScale_b_nvtx_30_inf = mpTScale_c_nvtx_30_inf
      = mpTScale_nvtx_0_5 = mpTScale_nvtx_5_10 = mpTScale_nvtx_10_15 
      = mpTScale_nvtx_15_20 = mpTScale_nvtx_20_30 = mpTScale_nvtx_30_inf
      = mNJetsEtaF_30
      = mpTScale_a = mpTScale_b = mpTScale_c = mpTScale_pT
      = 0;
  
  DQMStore* dbe = &*edm::Service<DQMStore>();
  if (dbe) {
    dbe->setCurrentFolder("JetMET/RecoJetsV/JPTJetTask_" + mInputCollection.label());
    //
    numberofevents    = dbe->book1D("numberofevents","numberofevents", 3, 0 , 2);
    //
    mEta              = dbe->book1D("Eta", "Eta", 120, -6, 6); 
    mEtaFineBin       = dbe->book1D("EtaFineBin_Pt10", "EtaFineBin_Pt10", 600, -6, 6); 
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

    //
    mGenEta           = dbe->book1D("GenEta", "GenEta", 120, -6, 6);
    mGenPhi           = dbe->book1D("GenPhi", "GenPhi", 70, -3.5, 3.5);
    mGenPt            = dbe->book1D("GenPt", "GenPt", 100, 0, 150);
    mGenPt_80         = dbe->book1D("GenPt_80", "GenPt_80", 100, 0, 1500);
    //
    mGenEtaFirst      = dbe->book1D("GenEtaFirst", "GenEtaFirst", 100, -5, 5);
    mGenPhiFirst      = dbe->book1D("GenPhiFirst", "GenPhiFirst", 70, -3.5, 3.5);
    //
    //
    mHadTiming        = dbe->book1D("HadTiming", "HadTiming", 75, -50, 100);
    mEmTiming         = dbe->book1D("EMTiming", "EMTiming", 75, -50, 100);
    //
    mNJetsEtaC        = dbe->book1D("NJetsEtaC_Pt10", "NJetsEtaC_Pt10", 15, 0, 15);
    mNJetsEtaF        = dbe->book1D("NJetsEtaF_Pt10", "NJetsEtaF_Pt10", 15, 0, 15);
    mNJetsEtaF_30        = dbe->book1D("NJetsEtaF_Pt30", "NJetsEtaF_Pt30", 15, 0, 15);
    //
    mNJets1           = dbe->bookProfile("NJets1", "NJets1", 100, 0, 200,  100, 0, 50, "s");
    mNJets2           = dbe->bookProfile("NJets2", "NJets2", 100, 0, 4000, 100, 0, 50, "s");
    //
    //
    mPthat_80            = dbe->book1D("Pthat_80", "Pthat_80", 100, 0.0, 1000.0); 
    mPthat_3000          = dbe->book1D("Pthat_3000", "Pthat_3000", 100, 1000.0, 4000.0); 

    //Corr
    mCorrJetPt  = dbe->book1D("CorrPt", "CorrPt", 100, 0, 150);
    mCorrJetPt_80 = dbe->book1D("CorrPt_80", "CorrPt_80", 100, 0, 4000);
    mCorrJetEta = dbe->book1D("CorrEta", "CorrEta", 120, -6, 6);
    mCorrJetPhi = dbe->book1D("CorrPhi", "CorrPhi", 70, -3.5, 3.5);
    //mjetArea = dbe->book1D("jetArea","jetArea",25,0,2.5);

    //nvtx
    nvtx_0_30 = dbe->book1D("nvtx_0_30","nvtx_0_30",31,-0.5,30.5);
    nvtx_0_60 = dbe->book1D("nvtx_0_60","nvtx_0_60",61,-0.5,60.5);

    //pT scale with nvtx
    mpTScale_a_nvtx_0_5 = dbe->book1D("mpTScale_a_nvtx_0_5", "pTScale_a_nvtx_0_5_0<|eta|<1.3_60_120",100, 0, 2);
    mpTScale_b_nvtx_0_5 = dbe->book1D("mpTScale_b_nvtx_0_5", "pTScale_b_nvtx_0_5_0<|eta|<1.3_200_300",100, 0, 2);
    mpTScale_c_nvtx_0_5 = dbe->book1D("mpTScale_c_nvtx_0_5", "pTScale_c_nvtx_0_5_0<|eta|<1.3_600_900",100, 0, 2);
    mpTScale_a_nvtx_5_10 = dbe->book1D("mpTScale_a_nvtx_5_10", "pTScale_a_nvtx_5_10_0<|eta|<1.3_60_120",100, 0, 2);
    mpTScale_b_nvtx_5_10 = dbe->book1D("mpTScale_b_nvtx_5_10", "pTScale_b_nvtx_5_10_0<|eta|<1.3_200_300",100, 0, 2);
    mpTScale_c_nvtx_5_10 = dbe->book1D("mpTScale_c_nvtx_5_10", "pTScale_c_nvtx_5_10_0<|eta|<1.3_600_900",100, 0, 2);
    mpTScale_a_nvtx_10_15 = dbe->book1D("mpTScale_a_nvtx_10_15", "pTScale_a_nvtx_10_15_0<|eta|<1.3_60_120",100, 0, 2);
    mpTScale_b_nvtx_10_15 = dbe->book1D("mpTScale_b_nvtx_10_15", "pTScale_b_nvtx_10_15_0<|eta|<1.3_200_300",100, 0, 2);
    mpTScale_c_nvtx_10_15 = dbe->book1D("mpTScale_c_nvtx_10_15", "pTScale_c_nvtx_10_15_0<|eta|<1.3_600_900",100, 0, 2);
    mpTScale_a_nvtx_15_20 = dbe->book1D("mpTScale_a_nvtx_15_20", "pTScale_a_nvtx_15_20_0<|eta|<1.3_60_120",100, 0, 2);
    mpTScale_b_nvtx_15_20 = dbe->book1D("mpTScale_b_nvtx_15_20", "pTScale_b_nvtx_15_20_0<|eta|<1.3_200_300",100, 0, 2);
    mpTScale_c_nvtx_15_20 = dbe->book1D("mpTScale_c_nvtx_15_20", "pTScale_c_nvtx_15_20_0<|eta|<1.3_600_900",100, 0, 2);
    mpTScale_a_nvtx_20_30 = dbe->book1D("mpTScale_a_nvtx_20_30", "pTScale_a_nvtx_20_30_0<|eta|<1.3_60_120",100, 0, 2);
    mpTScale_b_nvtx_20_30 = dbe->book1D("mpTScale_b_nvtx_20_30", "pTScale_b_nvtx_20_30_0<|eta|<1.3_200_300",100, 0, 2);
    mpTScale_c_nvtx_20_30 = dbe->book1D("mpTScale_c_nvtx_20_30", "pTScale_c_nvtx_20_30_0<|eta|<1.3_600_900",100, 0, 2);
    mpTScale_a_nvtx_30_inf = dbe->book1D("mpTScale_a_nvtx_30_inf", "pTScale_a_nvtx_30_inf_0<|eta|<1.3_60_120",100, 0, 2);
    mpTScale_b_nvtx_30_inf = dbe->book1D("mpTScale_b_nvtx_30_inf", "pTScale_b_nvtx_30_inf_0<|eta|<1.3_200_300",100, 0, 2);
    mpTScale_c_nvtx_30_inf = dbe->book1D("mpTScale_c_nvtx_30_inf", "pTScale_c_nvtx_30_inf_0<|eta|<1.3_600_900",100, 0, 2);
    mpTScale_a = dbe->book1D("mpTScale_a", "pTScale_a_60_120",100, 0, 2);
    mpTScale_b = dbe->book1D("mpTScale_b", "pTScale_b_200_300",100, 0, 2);
    mpTScale_c = dbe->book1D("mpTScale_c", "pTScale_c_600_900",100, 0, 2);
    //
    double log10PtMin = 0.5; //=3.1622766
    double log10PtMax = 3.75; //=5623.41325
    int log10PtBins = 26; 
    double etaRange[91] = {-6.0,-5.8,-5.6,-5.4,-5.2,-5.0,-4.8,-4.6,-4.4,-4.2,-4.0,-3.8,-3.6,-3.4,-3.2,-3.0,-2.9,-2.8,-2.7,-2.6,-2.5,-2.4,-2.3,-2.2,-2.1,-2.0,-1.9,-1.8,-1.7,-1.6,-1.5,-1.4,-1.3,-1.2,-1.1,-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.2,3.4,3.6,3.8,4.0,4.2,4.4,4.6,4.8,5.0,5.2,5.4,5.6,5.8,6.0};


   // int log10PtFineBins = 50;
    //
    if (mTurnOnEverything.compare("yes")==0) {
    }

    //
    if (mTurnOnEverything.compare("yes")==0) {
  }
    mpTScaleB_d = dbe->bookProfile("pTScaleB_d", "pTScale_d_0<|eta|<1.5",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScaleE_d = dbe->bookProfile("pTScaleE_d", "pTScale_d_1.5<|eta|<3.0",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScaleF_d = dbe->bookProfile("pTScaleF_d", "pTScale_d_3.0<|eta|<6.0",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScalePhiB_d = dbe->bookProfile("pTScalePhiB_d", "pTScalePhi_d_0<|eta|<1.5",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScalePhiE_d = dbe->bookProfile("pTScalePhiE_d", "pTScalePhi_d_1.5<|eta|<3.0",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
    mpTScalePhiF_d = dbe->bookProfile("pTScalePhiF_d", "pTScalePhi_d_3.0<|eta|<6.0",
				   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
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

  mpTScale_nvtx_0_5  = dbe->bookProfile("pTScale_nvtx_0_5", "pTScale_nvtx_0_5_0<|eta|<1.3",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
   mpTScale_nvtx_5_10  = dbe->bookProfile("pTScale_nvtx_5_10", "pTScale_nvtx_5_10_0<|eta|<1.3",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
   mpTScale_nvtx_10_15  = dbe->bookProfile("pTScale_nvtx_10_15", "pTScale_nvtx_10_15_0<|eta|<1.3",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
   mpTScale_nvtx_15_20  = dbe->bookProfile("pTScale_nvtx_15_20", "pTScale_nvtx_15_20_0<|eta|<1.3",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
   mpTScale_nvtx_20_30  = dbe->bookProfile("pTScale_nvtx_20_30", "pTScale_nvtx_20_30_0<|eta|<1.3",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
   mpTScale_nvtx_30_inf  = dbe->bookProfile("pTScale_nvtx_30_inf", "pTScale_nvtx_30_inf_0<|eta|<1.3",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
 mpTScale_pT = dbe->bookProfile("pTScale_pT", "pTScale_vs_pT",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 2, " ");
///////////Corr profile//////////////
    mpTRatio = dbe->bookProfile("pTRatio", "pTRatio",
                                log10PtBins, log10PtMin, log10PtMax, 100, 0.,5., " ");
    mpTRatioB_d = dbe->bookProfile("pTRatioB_d", "pTRatio_d_0<|eta|<1.5",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");
    mpTRatioE_d = dbe->bookProfile("pTRatioE_d", "pTRatio_d_1.5<|eta|<3.0",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");
    mpTRatioF_d = dbe->bookProfile("pTRatioF_d", "pTRatio_d_3.0<|eta|<6.0",
                                   log10PtBins, log10PtMin, log10PtMax, 0, 5, " ");
    mpTRatio_30_200_d    = dbe->bookProfile("pTRatio_30_200_d", "pTRatio_d_30<pT<200",
                                          90,etaRange, 0., 5., " ");
    mpTRatio_200_600_d   = dbe->bookProfile("pTRatio_200_600_d", "pTRatio_d_200<pT<600",
                                          90,etaRange, 0., 5., " ");
    mpTRatio_600_1500_d   = dbe->bookProfile("pTRatio_600_1500_d", "pTRatio_d_600<pT<1500",
                                          90,etaRange, 0., 5., " ");
    mpTRatio_1500_3500_d = dbe->bookProfile("pTRatio_1500_3500_d", "pTRatio_d_1500<pt<3500",
                                          90,etaRange, 0., 5., " ");
    mpTResponse = dbe->bookProfile("pTResponse", "pTResponse",
				log10PtBins, log10PtMin, log10PtMax, 100, 0.8,1.2, " ");
    mpTResponseB_d = dbe->bookProfile("pTResponseB_d", "pTResponse_d_0<|eta|<1.5",
				      log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
    mpTResponseE_d = dbe->bookProfile("pTResponseE_d", "pTResponse_d_1.5<|eta|<3.0",
                                   log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
    mpTResponseF_d = dbe->bookProfile("pTResponseF_d", "pTResponse_d_3.0<|eta|<6.0",
                                   log10PtBins, log10PtMin, log10PtMax, 0.8, 1.2, " ");
    mpTResponse_30_200_d    = dbe->bookProfile("pTResponse_30_200_d", "pTResponse_d_30<pT<200",
                                          90,etaRange, 0.8, 1.2, " ");
    mpTResponse_200_600_d   = dbe->bookProfile("pTResponse_200_600_d", "pTResponse_d_200<pT<600",
                                          90,etaRange, 0.8, 1.2, " ");
    mpTResponse_600_1500_d   = dbe->bookProfile("pTResponse_600_1500_d", "pTResponse_d_600<pT<1500",
                                          90,etaRange, 0.8, 1.2, " ");
    mpTResponse_1500_3500_d = dbe->bookProfile("pTResponse_1500_3500_d", "pTResponse_d_1500<pt<3500",
					       90,etaRange, 0.8, 1.2, " ");
    mpTResponse_30_d = dbe->bookProfile("pTResponse_30_d", "pTResponse_d_pt>30",
					       90,etaRange, 0.8, 1.2, " ");

    mpTResponse_nvtx_0_5 = dbe->bookProfile("pTResponse_nvtx_0_5", "pTResponse_nvtx_0_5", log10PtBins, log10PtMin, log10PtMax, 100, 0.8,1.2, " ");
   mpTResponse_nvtx_5_10 = dbe->bookProfile("pTResponse_nvtx_5_10", "pTResponse_nvtx_5_10", log10PtBins, log10PtMin, log10PtMax, 100, 0.8,1.2, " ");
   mpTResponse_nvtx_10_15 = dbe->bookProfile("pTResponse_nvtx_10_15", "pTResponse_nvtx_10_15", log10PtBins, log10PtMin, log10PtMax, 100, 0.8,1.2, " ");
   mpTResponse_nvtx_15_20 = dbe->bookProfile("pTResponse_nvtx_15_20", "pTResponse_nvtx_15_20", log10PtBins, log10PtMin, log10PtMax, 100, 0.8,1.2, " ");
   mpTResponse_nvtx_20_30 = dbe->bookProfile("pTResponse_nvtx_20_30", "pTResponse_nvtx_20_30", log10PtBins, log10PtMin, log10PtMax, 100, 0.8,1.2, " ");
   mpTResponse_nvtx_30_inf = dbe->bookProfile("pTResponse_nvtx_30_inf", "pTResponse_nvtx_30_inf", log10PtBins, log10PtMin, log10PtMax, 100, 0.8,1.2, " ");


  } // if (dbe)

  if (mOutputFile.empty ()) {
    LogInfo("OutputInfo") << " JPTJet histograms will NOT be saved";
  } 
  else {
    LogInfo("OutputInfo") << " JPTJethistograms will be saved to file:" << mOutputFile;
  }
}
   
JPTJetTester::~JPTJetTester()
{
}

void JPTJetTester::beginJob(){
}

void JPTJetTester::endJob() {
 if (!mOutputFile.empty() && &*edm::Service<DQMStore>()) edm::Service<DQMStore>()->save (mOutputFile);
}


void JPTJetTester::analyze(const edm::Event& mEvent, const edm::EventSetup& mSetup)
{
  double countsfornumberofevents = 1;
  numberofevents->Fill(countsfornumberofevents);

  //get primary vertices
  edm::Handle<vector<reco::Vertex> >pvHandle;
  try {
    mEvent.getByLabel( "offlinePrimaryVertices", pvHandle );
  } catch ( cms::Exception & e ) {
    //cout <<prefix<<"error: " << e.what() << endl;
  }
  vector<reco::Vertex> goodVertices;
  for (unsigned i = 0; i < pvHandle->size(); i++) {
    if ( (*pvHandle)[i].ndof() > 4 &&
       ( fabs((*pvHandle)[i].z()) <= 24. ) &&
       ( fabs((*pvHandle)[i].position().rho()) <= 2.0 ) )
       goodVertices.push_back((*pvHandle)[i]);
  }
  
  nvtx_0_30->Fill(goodVertices.size());
  nvtx_0_60->Fill(goodVertices.size());

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
      }
    }
  } catch (...) {
    edm::LogInfo("OutputInfo") << " No EE RecHits.";
  }

  //***********************************
  //*** Get the Jet collection
  //***********************************
  math::XYZTLorentzVector p4tmp[2];
  Handle<JPTJetCollection> jptJets;
  mEvent.getByLabel(mInputCollection, jptJets);
  if (!jptJets.isValid()) return;
  JPTJetCollection::const_iterator jet = jptJets->begin ();
  int jetIndex = 0;
  int nJet = 0;
  int nJetF = 0;
  int nJetC = 0;
  int nJetF_30 =0;
  for (; jet != jptJets->end (); jet++, jetIndex++) {
    if (jet->pt() > 10.) {
      if (fabs(jet->eta()) > 1.5) 
	nJetF++;
      else 
	nJetC++;	  
    }
    if (jet->pt() > 30.) nJetF_30++;
    if (jet->pt() > 10.) {
      if (mEta) mEta->Fill (jet->eta());
      if (mEtaFineBin) mEtaFineBin->Fill (jet->eta());
      if (mPhiFineBin) mPhiFineBin->Fill (jet->phi());
    }
    //if (mjetArea) mjetArea->Fill(jet->jetArea());
    if (mPhi) mPhi->Fill (jet->phi());
    if (mE) mE->Fill (jet->energy());
    if (mE_80) mE_80->Fill (jet->energy());
    if (mP) mP->Fill (jet->p());
    if (mP_80) mP_80->Fill (jet->p());
    if (mPt) mPt->Fill (jet->pt());
    if (mPt_80) mPt_80->Fill (jet->pt());
    if (mMass) mMass->Fill (jet->mass());
    if (mMass_80) mMass_80->Fill (jet->mass());
    if (jet == jptJets->begin ()) { // first jet
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

  }

  if (mNJetsEtaC) mNJetsEtaC->Fill( nJetC );
  if (mNJetsEtaF) mNJetsEtaF->Fill( nJetF );
  if (mNJetsEtaF_30) mNJetsEtaF_30->Fill( nJetF_30 );

  if (nJet == 2) {
    if (mMjj) mMjj->Fill( (p4tmp[0]+p4tmp[1]).mass() );
    if (mMjj_3000) mMjj_3000->Fill( (p4tmp[0]+p4tmp[1]).mass() );
  }

 // Correction jets
  const JetCorrector* corrector = JetCorrector::getJetCorrector (JetCorrectionService,mSetup);

  for (JPTJetCollection::const_iterator jet = jptJets->begin(); jet !=jptJets ->end(); jet++) 
  {
 
      JPTJet  correctedJet = *jet;
      //double scale = corrector->correction(jet->p4()); 
      //double scale = corrector->correction(*jet);
      double scale = corrector->correction(*jet,mEvent,mSetup);
      correctedJet.scaleEnergy(scale); 
      if(correctedJet.pt()>30){
      mCorrJetPt->Fill(correctedJet.pt());
      mCorrJetPt_80->Fill(correctedJet.pt());  
      if (correctedJet.pt()>10) mCorrJetEta->Fill(correctedJet.eta());
      mCorrJetPhi->Fill(correctedJet.phi());
      mpTRatio->Fill(log10(jet->pt()),correctedJet.pt()/jet->pt());

       if (fabs(jet->eta())<1.5) {
           mpTRatioB_d->Fill(log10(jet->pt()), correctedJet.pt()/jet->pt());
      }

     if (fabs(jet->eta())>1.5 && fabs(jet->eta())<3.0) {
        mpTRatioE_d->Fill (log10(jet->pt()), correctedJet.pt()/jet->pt());
     }
     if (fabs(jet->eta())>3.0 && fabs(jet->eta())<6.0) {
        mpTRatioF_d->Fill (log10(jet->pt()), correctedJet.pt()/jet->pt());
    }
     if (jet->pt()>30.0 && jet->pt()<200.0) {
    mpTRatio_30_200_d->Fill (jet->eta(),correctedJet.pt()/jet->pt());
  }
   if (jet->pt()>200.0 && jet->pt()<600.0) {
    mpTRatio_200_600_d->Fill (jet->eta(),correctedJet.pt()/jet->pt());
  }
   if (jet->pt()>600.0 && jet->pt()<1500.0) {
    mpTRatio_600_1500_d->Fill (jet->eta(),correctedJet.pt()/jet->pt());
  }
   if (jet->pt()>1500.0 && jet->pt()<3500.0) {
    mpTRatio_1500_3500_d->Fill (jet->eta(),correctedJet.pt()/jet->pt());
  }

      }
  }
  // Count Jets above Pt cut
  for (int istep = 0; istep < 100; ++istep) {
    int     njet = 0;
    float ptStep = (istep * (200./100.));

    for ( JPTJetCollection::const_iterator jpt = jptJets->begin(); jpt != jptJets->end(); ++ jpt ) {
      if ( jpt->pt() > ptStep ) njet++;
    }
    mNJets1->Fill( ptStep, njet );
  }

  for (int istep = 0; istep < 100; ++istep) {
    int     njet = 0;
    float ptStep = (istep * (4000./100.));
    for ( JPTJetCollection::const_iterator jpt = jptJets->begin(); jpt != jptJets->end(); ++ jpt ) {
      if ( jpt->pt() > ptStep ) njet++;
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


  // now match JPTJets to GenJets
  JetMatchingTools jetMatching (mEvent);
  if (!(mInputGenCollection.label().empty())) {
    //    Handle<GenJetCollection> genJets;
    //    mEvent.getByLabel(mInputGenCollection, genJets);

    std::vector <std::vector <const reco::GenParticle*> > genJetConstituents (genJets->size());
    std::vector <std::vector <const reco::GenParticle*> > jptJetConstituents (jptJets->size());
    
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
      if (jptJets->size() <= 0) continue; // no JPTJets - nothing to match
      if (mRThreshold > 0) {
	unsigned iJPTJetBest = 0;
	double deltaRBest = 999.;
	for (unsigned iJPTJet = 0; iJPTJet < jptJets->size(); ++iJPTJet) {
	  double dR = deltaR (genJet.eta(), genJet.phi(), (*jptJets) [iJPTJet].eta(), (*jptJets) [iJPTJet].phi());
	  if (deltaRBest < mRThreshold && dR < mRThreshold && genJet.pt() > 5.) {
	    /*
	    std::cout << "Yet another matched jet for GenJet pt=" << genJet.pt()
		      << " previous JPTJet pt/dr: " << (*jptJets) [iJPTJetBest].pt() << '/' << deltaRBest
		      << " new JPTJet pt/dr: " << (*jptJets) [iJPTJet].pt() << '/' << dR
		      << std::endl;
	    */
	  }
	  if (dR < deltaRBest) {
	    iJPTJetBest = iJPTJet;
	    deltaRBest = dR;
	  }
	}
	if (mTurnOnEverything.compare("yes")==0) {
	  //mRMatch->Fill (logPtGen, genJet.eta(), deltaRBest);
	}
	if (deltaRBest < mRThreshold) { // Matched
	  fillMatchHists (genJet, (*jptJets) [iJPTJetBest],goodVertices);
	}

		///////////pT Response///////////////
	double CorrdeltaRBest = 999.;
	double CorrJetPtBest = 0;
	for (JPTJetCollection::const_iterator jet = jptJets->begin(); jet !=jptJets ->end(); jet++) {
	  JPTJet  correctedJet = *jet;
	  //double scale = corrector->correction(jet->p4());
	  //double scale = corrector->correction(*jet);
          double scale = corrector->correction(*jet,mEvent,mSetup);
	  correctedJet.scaleEnergy(scale);
	  double CorrJetPt = correctedJet.pt();
	  if(CorrJetPt>30){
	  double CorrdR = deltaR (genJet.eta(), genJet.phi(), correctedJet.eta(), correctedJet.phi());
	  if (CorrdR < CorrdeltaRBest) {
	    CorrdeltaRBest = CorrdR;
	    CorrJetPtBest = CorrJetPt;
	  }
	}
	}
	if (deltaRBest < mRThreshold) { // Matched
	  mpTResponse->Fill(log10(genJet.pt()),CorrJetPtBest/genJet.pt());
	  
	  if (fabs(genJet.eta())<1.5) {
	    mpTResponseB_d->Fill(log10(genJet.pt()), CorrJetPtBest/genJet.pt());
	  }	
	  
	  if (fabs(genJet.eta())>1.5 && fabs(genJet.eta())<3.0) {
	    mpTResponseE_d->Fill (log10(genJet.pt()), CorrJetPtBest/genJet.pt());   
	  }
	  
	  if (fabs(genJet.eta())>3.0 && fabs(genJet.eta())<6.0) {
        mpTResponseF_d->Fill (log10(genJet.pt()), CorrJetPtBest/genJet.pt());
	  }
	  if (genJet.pt()>30.0 && genJet.pt()<200.0) {
	    mpTResponse_30_200_d->Fill (genJet.eta(),CorrJetPtBest/genJet.pt());
	  }
	  if (genJet.pt()>200.0 && genJet.pt()<600.0) {
	    mpTResponse_200_600_d->Fill (genJet.eta(),CorrJetPtBest/genJet.pt());
	  }
	  if (genJet.pt()>600.0 && genJet.pt()<1500.0) {
	    mpTResponse_600_1500_d->Fill (genJet.eta(),CorrJetPtBest/genJet.pt());
	  }
	  if (genJet.pt()>1500.0 && genJet.pt()<3500.0) {
	    mpTResponse_1500_3500_d->Fill (genJet.eta(),CorrJetPtBest/genJet.pt());
	  }
	  if (genJet.pt()>30.0) {
	    mpTResponse_30_d->Fill (genJet.eta(),CorrJetPtBest/genJet.pt());
	  }

	  if(goodVertices.size()<=5) mpTResponse_nvtx_0_5->Fill(log10(genJet.pt()),CorrJetPtBest/genJet.pt());
	  if(goodVertices.size()>5 && goodVertices.size()<=10) mpTResponse_nvtx_5_10->Fill(log10(genJet.pt()),CorrJetPtBest/genJet.pt());
	  if(goodVertices.size()>10 && goodVertices.size()<=15) mpTResponse_nvtx_10_15->Fill(log10(genJet.pt()),CorrJetPtBest/genJet.pt());	  
	  if(goodVertices.size()>15 && goodVertices.size()<=20) mpTResponse_nvtx_15_20->Fill(log10(genJet.pt()),CorrJetPtBest/genJet.pt());
	  if(goodVertices.size()>20 && goodVertices.size()<=30) mpTResponse_nvtx_20_30->Fill(log10(genJet.pt()),CorrJetPtBest/genJet.pt());
	  if(goodVertices.size()>30) mpTResponse_nvtx_30_inf->Fill(log10(genJet.pt()),CorrJetPtBest/genJet.pt());


	}
	///////////////////////////////////
      }
      
    }
  }
}

}////Gen Close

void JPTJetTester::fillMatchHists (const reco::GenJet& fGenJet, const reco::JPTJet& fJPTJet,std::vector<reco::Vertex> goodVertices) {
  double logPtGen = log10 (fGenJet.pt());
  double PtGen = fGenJet.pt();
  double PtJpt = fJPTJet.pt();

  double PtThreshold = 10.;

  if (mTurnOnEverything.compare("yes")==0) {
    mDeltaEta->Fill (logPtGen, fGenJet.eta(), fJPTJet.eta()-fGenJet.eta());
    mDeltaPhi->Fill (logPtGen, fGenJet.eta(), fJPTJet.phi()-fGenJet.phi());

    mEScaleFineBin->Fill (logPtGen, fGenJet.eta(), fJPTJet.energy()/fGenJet.energy());
  
    if (fGenJet.pt()>PtThreshold) {
      mEScale_pt10->Fill (logPtGen, fGenJet.eta(), fJPTJet.energy()/fGenJet.energy());

    }

  }
  if (fJPTJet.pt() > PtThreshold) {
    mDelEta->Fill (fGenJet.eta()-fJPTJet.eta());
    mDelPhi->Fill (fGenJet.phi()-fJPTJet.phi());
    mDelPt->Fill  ((fGenJet.pt()-fJPTJet.pt())/fGenJet.pt());
  }

  if (fabs(fGenJet.eta())<1.5) {

    mpTScaleB_d->Fill (log10(PtGen), PtJpt/PtGen);
    mpTScalePhiB_d->Fill (fGenJet.phi(), PtJpt/PtGen);
    
    if (PtGen>30.0 && PtGen<200.0) {
      mpTScale1DB_30_200->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>200.0 && PtGen<600.0) {
      mpTScale1DB_200_600->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>600.0 && PtGen<1500.0) {
      mpTScale1DB_600_1500->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>1500.0 && PtGen<3500.0) {
      mpTScale1DB_1500_3500->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    
  }

  if (fabs(fGenJet.eta())>1.5 && fabs(fGenJet.eta())<3.0) {
    mpTScaleE_d->Fill (log10(PtGen), PtJpt/PtGen);
    mpTScalePhiE_d->Fill (fGenJet.phi(), PtJpt/PtGen);
    
    if (PtGen>30.0 && PtGen<200.0) {
      mpTScale1DE_30_200->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>200.0 && PtGen<600.0) {
      mpTScale1DE_200_600->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>600.0 && PtGen<1500.0) {
      mpTScale1DE_600_1500->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>1500.0 && PtGen<3500.0) {
      mpTScale1DE_1500_3500->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    
  }

  if (fabs(fGenJet.eta())>3.0 && fabs(fGenJet.eta())<6.0) {

    mpTScaleF_d->Fill (log10(PtGen), PtJpt/PtGen);
    mpTScalePhiF_d->Fill (fGenJet.phi(), PtJpt/PtGen);
    
    if (PtGen>30.0 && PtGen<200.0) {
      mpTScale1DF_30_200->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>200.0 && PtGen<600.0) {
      mpTScale1DF_200_600->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>600.0 && PtGen<1500.0) {
      mpTScale1DF_600_1500->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    if (PtGen>1500.0 && PtGen<3500.0) {
      mpTScale1DF_1500_3500->Fill (fJPTJet.pt()/fGenJet.pt());
    }
    
  }

  if (fGenJet.pt()>30.0 && fGenJet.pt()<200.0) {
    //mpTScale_30_200_s->Fill (fGenJet.eta(),fJPTJet.pt()/fGenJet.pt());
    mpTScale_30_200_d->Fill (fGenJet.eta(),fJPTJet.pt()/fGenJet.pt());
    //mpTScale1D_30_200->Fill (fJPTJet.pt()/fGenJet.pt());
  }

  if (fGenJet.pt()>200.0 && fGenJet.pt()<600.0) {
    mpTScale_200_600_d->Fill (fGenJet.eta(),fJPTJet.pt()/fGenJet.pt());
  }

  if (fGenJet.pt()>600.0 && fGenJet.pt()<1500.0) {
    mpTScale_600_1500_d->Fill (fGenJet.eta(),fJPTJet.pt()/fGenJet.pt());
  }

  if (fGenJet.pt()>1500.0 && fGenJet.pt()<3500.0) {
    mpTScale_1500_3500_d->Fill (fGenJet.eta(),fJPTJet.pt()/fGenJet.pt());
  }

  if (fabs(fGenJet.eta())<1.3) {
    if(fGenJet.pt()>60.0 && fGenJet.pt()<120.0) {
     if(goodVertices.size()<=5) mpTScale_a_nvtx_0_5->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>200.0 && fGenJet.pt()<300.0) {
     if(goodVertices.size()<=5) mpTScale_b_nvtx_0_5->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>600.0 && fGenJet.pt()<900.0) {
     if(goodVertices.size()<=5) mpTScale_c_nvtx_0_5->Fill( PtJpt/PtGen);
    }

    if(fGenJet.pt()>60.0 && fGenJet.pt()<120.0) {
     if(goodVertices.size()>5 && goodVertices.size()<=10) mpTScale_a_nvtx_5_10->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>200.0 && fGenJet.pt()<300.0) {
     if(goodVertices.size()>5 && goodVertices.size()<=10) mpTScale_b_nvtx_5_10->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>600.0 && fGenJet.pt()<900.0) {
     if(goodVertices.size()>5 && goodVertices.size()<=10) mpTScale_c_nvtx_5_10->Fill( PtJpt/PtGen);
    }

    if(fGenJet.pt()>60.0 && fGenJet.pt()<120.0) {
     if(goodVertices.size()>10 && goodVertices.size()<=15) mpTScale_a_nvtx_10_15->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>200.0 && fGenJet.pt()<300.0) {
     if(goodVertices.size()>10 && goodVertices.size()<=15) mpTScale_b_nvtx_10_15->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>600.0 && fGenJet.pt()<900.0) {
     if(goodVertices.size()>10 && goodVertices.size()<=15) mpTScale_c_nvtx_10_15->Fill( PtJpt/PtGen);
    }

    if(fGenJet.pt()>60.0 && fGenJet.pt()<120.0) {
     if(goodVertices.size()>15 && goodVertices.size()<=20) mpTScale_a_nvtx_15_20->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>200.0 && fGenJet.pt()<300.0) {
     if(goodVertices.size()>15 && goodVertices.size()<=20) mpTScale_b_nvtx_15_20->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>600.0 && fGenJet.pt()<900.0) {
     if(goodVertices.size()>15 && goodVertices.size()<=20) mpTScale_c_nvtx_15_20->Fill( PtJpt/PtGen);
    }

if(fGenJet.pt()>60.0 && fGenJet.pt()<120.0) {
     if(goodVertices.size()>20 && goodVertices.size()<=30) mpTScale_a_nvtx_20_30->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>200.0 && fGenJet.pt()<300.0) {
     if(goodVertices.size()>20 && goodVertices.size()<=30) mpTScale_b_nvtx_20_30->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>600.0 && fGenJet.pt()<900.0) {
     if(goodVertices.size()>20 && goodVertices.size()<=30) mpTScale_c_nvtx_20_30->Fill( PtJpt/PtGen);
    }

if(fGenJet.pt()>60.0 && fGenJet.pt()<120.0) {
     if(goodVertices.size()>30) mpTScale_a_nvtx_30_inf->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>200.0 && fGenJet.pt()<300.0) {
     if(goodVertices.size()>30) mpTScale_b_nvtx_30_inf->Fill( PtJpt/PtGen);
    }
    if(fGenJet.pt()>600.0 && fGenJet.pt()<900.0) {
     if(goodVertices.size()>30) mpTScale_c_nvtx_30_inf->Fill( PtJpt/PtGen);
    }

    if(goodVertices.size()<=5) mpTScale_nvtx_0_5->Fill(log10(PtGen),PtJpt/PtGen);
    if(goodVertices.size()>5 && goodVertices.size()<=10) mpTScale_nvtx_5_10->Fill(log10(PtGen),PtJpt/PtGen);
    if(goodVertices.size()>10 && goodVertices.size()<=15) mpTScale_nvtx_10_15->Fill(log10(PtGen),PtJpt/PtGen);
    if(goodVertices.size()>15 && goodVertices.size()<=20) mpTScale_nvtx_15_20->Fill(log10(PtGen), PtJpt/PtGen);
    if(goodVertices.size()>20 && goodVertices.size()<=30) mpTScale_nvtx_20_30->Fill(log10(PtGen), PtJpt/PtGen);
    if(goodVertices.size()>30) mpTScale_nvtx_30_inf->Fill(log10(PtGen), PtJpt/PtGen);
}
  if (fabs(fGenJet.eta())<1.3) {
  if(fGenJet.pt()>60.0 && fGenJet.pt()<120.0) mpTScale_a->Fill(PtJpt/PtGen);
  if(fGenJet.pt()>200.0 && fGenJet.pt()<300.0) mpTScale_b->Fill(PtJpt/PtGen);
  if(fGenJet.pt()>600.0 && fGenJet.pt()<900.0) mpTScale_c->Fill(PtJpt/PtGen);
  }
  mpTScale_pT->Fill (log10(PtGen), PtJpt/PtGen);

}

double JPTJetTester::getSumPt(const reco::TrackRefVector& tracks){

  double sumpt = 0.;
  
  for (reco::TrackRefVector::const_iterator itrack = tracks.begin(); itrack != tracks.end(); ++itrack){
    const reco::Track& track = **itrack;
    sumpt += track.pt(); 
  }
  
  return sumpt;

}
