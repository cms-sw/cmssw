// -*- C++ -*-
//
// Package:    Validation/RecoMET
// Class:      METTesterPostProcessor
// 
// Original Author:  "Matthias Weber"
//         Created:  Sun Feb 22 14:35:25 CET 2015
//

#include "Validation/RecoJets/plugins/JetTesterPostProcessor.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

// Some switches
//
// constructors and destructor
//
JetTesterPostProcessor::JetTesterPostProcessor(const edm::ParameterSet& iConfig)
{
  inputJetLabelRECO_=iConfig.getParameter<edm::InputTag>("JetTypeRECO");
  inputJetLabelMiniAOD_=iConfig.getParameter<edm::InputTag>("JetTypeMiniAOD");
}


JetTesterPostProcessor::~JetTesterPostProcessor()
{ 
}


// ------------ method called right after a run ends ------------
void 
JetTesterPostProcessor::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_)
{
  std::vector<std::string> subDirVec;
  std::string RunDir="JetMET/JetValidation/";
  iget_.setCurrentFolder(RunDir);
  jet_dirs=iget_.getSubdirs();
  bool found_reco_dir=false;
  bool found_miniaod_dir=false;
  //loop over jet subdirectories
  for (int i=0; i<int(jet_dirs.size()); i++) {
    ibook_.setCurrentFolder(jet_dirs[i]);  
    if(jet_dirs[i]==(RunDir+inputJetLabelRECO_.label())){
      found_reco_dir=true;
    }
    if(jet_dirs[i]==(RunDir+inputJetLabelMiniAOD_.label())){
      found_miniaod_dir=true;
    }
  }
  if(found_miniaod_dir && found_reco_dir){
    std::string rundir_reco=RunDir+inputJetLabelRECO_.label();
    std::string rundir_miniaod=RunDir+inputJetLabelMiniAOD_.label();

    MonitorElement* mGenPt_Reco=iget_.get(rundir_reco+"/"+"GenPt");
    MonitorElement* mGenPhi_Reco=iget_.get(rundir_reco+"/"+"GenPhi");
    MonitorElement* mGenEta_Reco=iget_.get(rundir_reco+"/"+"GenEta");
    MonitorElement* mPt_Reco=iget_.get(rundir_reco+"/"+"Pt");
    MonitorElement* mPhi_Reco=iget_.get(rundir_reco+"/"+"Phi");
    MonitorElement* mEta_Reco=iget_.get(rundir_reco+"/"+"Eta");
    MonitorElement* mCorrJetPt_Reco=iget_.get(rundir_reco+"/"+"CorrJetPt");
    MonitorElement* mCorrJetPhi_Reco=iget_.get(rundir_reco+"/"+"CorrJetPhi");
    MonitorElement* mCorrJetEta_Reco=iget_.get(rundir_reco+"/"+"CorrJetEta");
    /*
    map_string_vec.push_back("Pt");
    map_string_vec.push_back("Phi");
    map_string_vec.push_back("Eta");
    map_string_vec.push_back("CorrJetPt");
    map_string_vec.push_back("CorrJetPhi");
    map_string_vec.push_back("CorrJetEta");
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"Pt" ,mPt_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"Phi" ,mPhi_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"Eta" ,mEta_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"CorrJetPt"  ,mCorrJetPt_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"CorrJetPhi" ,mCorrJetPhi_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"CorrJetEta" ,mCorrJetEta_Reco));
    */
    MonitorElement* mPtCorrOverReco_Eta_20_40_Reco=iget_.get(rundir_reco+"/"+"PtCorrOverReco_Eta_20_40");
    MonitorElement* mPtCorrOverReco_Eta_200_600_Reco=iget_.get(rundir_reco+"/"+"PtCorrOverReco_Eta_200_600");
    MonitorElement* mPtCorrOverReco_Eta_1500_3500_Reco=iget_.get(rundir_reco+"/"+"PtCorrOverReco_Eta_1500_3500");
    MonitorElement* mPtCorrOverGen_GenEta_40_200_Reco=iget_.get(rundir_reco+"/"+"PtCorrOverGen_GenEta_40_200");
    MonitorElement* mPtCorrOverGen_GenEta_600_1500_Reco=iget_.get(rundir_reco+"/"+"PtCorrOverGen_GenEta_600_1500");
    MonitorElement* mDeltaEta_Reco=iget_.get(rundir_reco+"/"+"DeltaEta");
    MonitorElement* mDeltaPhi_Reco=iget_.get(rundir_reco+"/"+"DeltaPhi");
    MonitorElement* mDeltaPt_Reco=iget_.get(rundir_reco+"/"+"DeltaPt");
    MonitorElement* mMjj_Reco=iget_.get(rundir_reco+"/"+"Mjj");
    MonitorElement* mNJets40_Reco=iget_.get(rundir_reco+"/"+"NJets");
    MonitorElement* mchargedHadronMultiplicity_Reco=iget_.get(rundir_reco+"/"+"chargedHadronMultiplicity");
    MonitorElement* mneutralHadronMultiplicity_Reco=iget_.get(rundir_reco+"/"+"neutralHadronMultiplicity");
    MonitorElement* mphotonMultiplicity_Reco=iget_.get(rundir_reco+"/"+"photonMultiplicity");
    MonitorElement* mphotonEnergyFraction_Reco=iget_.get(rundir_reco+"/"+"photonEnergyFraction");
    MonitorElement* mneutralHadronEnergyFraction_Reco=iget_.get(rundir_reco+"/"+"neutralHadronEnergyFraction");
    MonitorElement* mchargedHadronEnergyFraction_Reco=iget_.get(rundir_reco+"/"+"chargedHadronEnergyFraction");

    std::vector<MonitorElement*>ME_Reco;
    ME_Reco.push_back(mGenPt_Reco);
    ME_Reco.push_back(mGenPhi_Reco);
    ME_Reco.push_back(mGenEta_Reco);
    ME_Reco.push_back(mPt_Reco);
    ME_Reco.push_back(mPhi_Reco);
    ME_Reco.push_back(mEta_Reco);
    ME_Reco.push_back(mCorrJetPt_Reco);
    ME_Reco.push_back(mCorrJetPhi_Reco);
    ME_Reco.push_back(mCorrJetEta_Reco);
    ME_Reco.push_back(mPtCorrOverReco_Eta_20_40_Reco);
    ME_Reco.push_back(mPtCorrOverReco_Eta_200_600_Reco);
    ME_Reco.push_back(mPtCorrOverReco_Eta_1500_3500_Reco);
    ME_Reco.push_back(mPtCorrOverGen_GenEta_40_200_Reco);
    ME_Reco.push_back(mPtCorrOverGen_GenEta_600_1500_Reco);
    ME_Reco.push_back(mDeltaEta_Reco);
    ME_Reco.push_back(mDeltaPhi_Reco);
    ME_Reco.push_back(mDeltaPt_Reco);
    ME_Reco.push_back(mMjj_Reco);
    ME_Reco.push_back(mNJets40_Reco);
    ME_Reco.push_back(mchargedHadronMultiplicity_Reco);
    ME_Reco.push_back(mneutralHadronMultiplicity_Reco);
    ME_Reco.push_back(mphotonMultiplicity_Reco);
    ME_Reco.push_back(mphotonEnergyFraction_Reco);
    ME_Reco.push_back(mneutralHadronEnergyFraction_Reco);
    ME_Reco.push_back(mchargedHadronEnergyFraction_Reco);

    MonitorElement* mGenPt_MiniAOD=iget_.get(rundir_miniaod+"/"+"GenPt");
    MonitorElement* mGenPhi_MiniAOD=iget_.get(rundir_miniaod+"/"+"GenPhi");
    MonitorElement* mGenEta_MiniAOD=iget_.get(rundir_miniaod+"/"+"GenEta");
    MonitorElement* mPt_MiniAOD=iget_.get(rundir_miniaod+"/"+"Pt");
    MonitorElement* mPhi_MiniAOD=iget_.get(rundir_miniaod+"/"+"Phi");
    MonitorElement* mEta_MiniAOD=iget_.get(rundir_miniaod+"/"+"Eta");
    MonitorElement* mCorrJetPt_MiniAOD=iget_.get(rundir_miniaod+"/"+"CorrJetPt");
    MonitorElement* mCorrJetPhi_MiniAOD=iget_.get(rundir_miniaod+"/"+"CorrJetPhi");
    MonitorElement* mCorrJetEta_MiniAOD=iget_.get(rundir_miniaod+"/"+"CorrJetEta");
    MonitorElement* mPtCorrOverReco_Eta_20_40_MiniAOD=iget_.get(rundir_miniaod+"/"+"PtCorrOverReco_Eta_20_40");
    MonitorElement* mPtCorrOverReco_Eta_200_600_MiniAOD=iget_.get(rundir_miniaod+"/"+"PtCorrOverReco_Eta_200_600");
    MonitorElement* mPtCorrOverReco_Eta_1500_3500_MiniAOD=iget_.get(rundir_miniaod+"/"+"PtCorrOverReco_Eta_1500_3500");
    MonitorElement* mPtCorrOverGen_GenEta_40_200_MiniAOD=iget_.get(rundir_miniaod+"/"+"PtCorrOverGen_GenEta_40_200");
    MonitorElement* mPtCorrOverGen_GenEta_600_1500_MiniAOD=iget_.get(rundir_miniaod+"/"+"PtCorrOverGen_GenEta_600_1500");
    MonitorElement* mDeltaEta_MiniAOD=iget_.get(rundir_miniaod+"/"+"DeltaEta");
    MonitorElement* mDeltaPhi_MiniAOD=iget_.get(rundir_miniaod+"/"+"DeltaPhi");
    MonitorElement* mDeltaPt_MiniAOD=iget_.get(rundir_miniaod+"/"+"DeltaPt");
    MonitorElement* mMjj_MiniAOD=iget_.get(rundir_miniaod+"/"+"Mjj");
    MonitorElement* mNJets40_MiniAOD=iget_.get(rundir_miniaod+"/"+"NJets");
    MonitorElement* mchargedHadronMultiplicity_MiniAOD=iget_.get(rundir_miniaod+"/"+"chargedHadronMultiplicity");
    MonitorElement* mneutralHadronMultiplicity_MiniAOD=iget_.get(rundir_miniaod+"/"+"neutralHadronMultiplicity");
    MonitorElement* mphotonMultiplicity_MiniAOD=iget_.get(rundir_miniaod+"/"+"photonMultiplicity");
    MonitorElement* mphotonEnergyFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"photonEnergyFraction");
    MonitorElement* mneutralHadronEnergyFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"neutralHadronEnergyFraction");
    MonitorElement* mchargedHadronEnergyFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"chargedHadronEnergyFraction");

    std::vector<MonitorElement*>ME_MiniAOD;
    ME_MiniAOD.push_back(mGenPt_MiniAOD);
    ME_MiniAOD.push_back(mGenPhi_MiniAOD);
    ME_MiniAOD.push_back(mGenEta_MiniAOD);
    ME_MiniAOD.push_back(mPt_MiniAOD);
    ME_MiniAOD.push_back(mPhi_MiniAOD);
    ME_MiniAOD.push_back(mEta_MiniAOD);
    ME_MiniAOD.push_back(mCorrJetPt_MiniAOD);
    ME_MiniAOD.push_back(mCorrJetPhi_MiniAOD);
    ME_MiniAOD.push_back(mCorrJetEta_MiniAOD);
    ME_MiniAOD.push_back(mPtCorrOverReco_Eta_20_40_MiniAOD);
    ME_MiniAOD.push_back(mPtCorrOverReco_Eta_200_600_MiniAOD);
    ME_MiniAOD.push_back(mPtCorrOverReco_Eta_1500_3500_MiniAOD);
    ME_MiniAOD.push_back(mPtCorrOverGen_GenEta_40_200_MiniAOD);
    ME_MiniAOD.push_back(mPtCorrOverGen_GenEta_600_1500_MiniAOD);
    ME_MiniAOD.push_back(mDeltaEta_MiniAOD);
    ME_MiniAOD.push_back(mDeltaPhi_MiniAOD);
    ME_MiniAOD.push_back(mDeltaPt_MiniAOD);
    ME_MiniAOD.push_back(mMjj_MiniAOD);
    ME_MiniAOD.push_back(mNJets40_MiniAOD);
    ME_MiniAOD.push_back(mchargedHadronMultiplicity_MiniAOD);
    ME_MiniAOD.push_back(mneutralHadronMultiplicity_MiniAOD);
    ME_MiniAOD.push_back(mphotonMultiplicity_MiniAOD);
    ME_MiniAOD.push_back(mphotonEnergyFraction_MiniAOD);
    ME_MiniAOD.push_back(mneutralHadronEnergyFraction_MiniAOD);
    ME_MiniAOD.push_back(mchargedHadronEnergyFraction_MiniAOD);

    ibook_.setCurrentFolder(RunDir+"MiniAOD_over_RECO");
    mGenPt_MiniAOD_over_Reco=ibook_.book1D("GenPt_MiniAOD_over_RECO",(TH1F*)mGenPt_Reco->getRootObject());
    mGenPhi_MiniAOD_over_Reco=ibook_.book1D("GenPhi_MiniAOD_over_RECO",(TH1F*)mGenPhi_Reco->getRootObject());
    mGenEta_MiniAOD_over_Reco=ibook_.book1D("GenEta_MiniAOD_over_RECO",(TH1F*)mGenEta_Reco->getRootObject());
    mPt_MiniAOD_over_Reco=ibook_.book1D("Pt_MiniAOD_over_RECO",(TH1F*)mPt_Reco->getRootObject());
    mPhi_MiniAOD_over_Reco=ibook_.book1D("Phi_MiniAOD_over_RECO",(TH1F*)mPhi_Reco->getRootObject());
    mEta_MiniAOD_over_Reco=ibook_.book1D("Eta_MiniAOD_over_RECO",(TH1F*)mEta_Reco->getRootObject());
    mCorrJetPt_MiniAOD_over_Reco=ibook_.book1D("CorrJetPt_MiniAOD_over_RECO",(TH1F*)mCorrJetPt_Reco->getRootObject());
    mCorrJetPhi_MiniAOD_over_Reco=ibook_.book1D("CorrJetPhi_MiniAOD_over_RECO",(TH1F*)mCorrJetPhi_Reco->getRootObject());
    mCorrJetEta_MiniAOD_over_Reco=ibook_.book1D("CorrJetEta_MiniAOD_over_RECO",(TH1F*)mCorrJetEta_Reco->getRootObject());

   //if eta range changed here need change in JetTester as well
   float etarange[91] = {-6.0, -5.8, -5.6, -5.4, -5.2, -5.0, -4.8, -4.6, -4.4, -4.2,
		     -4.0, -3.8, -3.6, -3.4, -3.2, -3.0, -2.9, -2.8, -2.7, -2.6,
		     -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6,
		     -1.5, -1.4, -1.3, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6,
		     -0.5, -0.4, -0.3, -0.2, -0.1,
		     0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		     1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
		     2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9,
		     3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8,
		     5.0, 5.2, 5.4, 5.6, 5.8, 6.0};

    mPtCorrOverReco_Eta_20_40_MiniAOD_over_Reco=ibook_.book1D("PtCorrOverReco_Eta_20_40_MiniAOD_over_RECO","20<genpt<40",90,etarange);
    mPtCorrOverReco_Eta_200_600_MiniAOD_over_Reco=ibook_.book1D("PtCorrOverReco_Eta_200_600_MiniAOD_over_RECO","200<genpt<600",90,etarange);
    mPtCorrOverReco_Eta_1500_3500_MiniAOD_over_Reco=ibook_.book1D("PtCorrOverReco_Eta_1500_3500_MiniAOD_over_RECO","1500<genpt<3500",90,etarange);
    mPtCorrOverGen_GenEta_40_200_MiniAOD_over_Reco=ibook_.book1D("PtCorrOverGen_GenEta_40_200_MiniAOD_over_RECO","40<genpt<200",90,etarange);
    mPtCorrOverGen_GenEta_600_1500_MiniAOD_over_Reco=ibook_.book1D("PtCorrOverGen_GenEta_600_1500_MiniAOD_over_RECO","600<genpt<1500",90,etarange);
    mDeltaPt_MiniAOD_over_Reco=ibook_.book1D("DeltaPt_MiniAOD_over_RECO",(TH1F*)mDeltaPt_Reco->getRootObject());
    mDeltaPhi_MiniAOD_over_Reco=ibook_.book1D("DeltaPhi_MiniAOD_over_RECO",(TH1F*)mDeltaPhi_Reco->getRootObject());
    mDeltaEta_MiniAOD_over_Reco=ibook_.book1D("DeltaEta_MiniAOD_over_RECO",(TH1F*)mDeltaEta_Reco->getRootObject());
    mMjj_MiniAOD_over_Reco=ibook_.book1D("Mjj_MiniAOD_over_RECO",(TH1F*)mMjj_Reco->getRootObject());
    mNJets40_MiniAOD_over_Reco=ibook_.book1D("NJets_MiniAOD_over_RECO",(TH1F*)mNJets40_Reco->getRootObject());
    mchargedHadronMultiplicity_MiniAOD_over_Reco=ibook_.book1D("chargedHadronMultiplicity_MiniAOD_over_RECO",(TH1F*)mchargedHadronMultiplicity_Reco->getRootObject());
    mneutralHadronMultiplicity_MiniAOD_over_Reco=ibook_.book1D("neutralHadronMultiplicity_MiniAOD_over_RECO",(TH1F*)mneutralHadronMultiplicity_Reco->getRootObject());
    mphotonMultiplicity_MiniAOD_over_Reco=ibook_.book1D("photonMultiplicity_MiniAOD_over_RECO",(TH1F*)mphotonMultiplicity_Reco->getRootObject());
    mchargedHadronEnergyFraction_MiniAOD_over_Reco=ibook_.book1D("chargedHadronEnergyFraction_MiniAOD_over_RECO",(TH1F*)mchargedHadronEnergyFraction_Reco->getRootObject());
    mneutralHadronEnergyFraction_MiniAOD_over_Reco=ibook_.book1D("neutralHadronEnergyFraction_MiniAOD_over_RECO",(TH1F*)mneutralHadronEnergyFraction_Reco->getRootObject());
    mphotonEnergyFraction_MiniAOD_over_Reco=ibook_.book1D("photonEnergyFraction_MiniAOD_over_RECO",(TH1F*)mphotonEnergyFraction_Reco->getRootObject());

    std::vector<MonitorElement*>ME_MiniAOD_over_Reco;
    ME_MiniAOD_over_Reco.push_back(mGenPt_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mGenPhi_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mGenEta_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPt_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPhi_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mEta_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mCorrJetPt_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mCorrJetPhi_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mCorrJetEta_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPtCorrOverReco_Eta_20_40_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPtCorrOverReco_Eta_200_600_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPtCorrOverReco_Eta_1500_3500_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPtCorrOverGen_GenEta_40_200_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPtCorrOverGen_GenEta_600_1500_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mDeltaEta_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mDeltaPhi_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mDeltaPt_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMjj_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mNJets40_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mchargedHadronMultiplicity_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mneutralHadronMultiplicity_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mphotonMultiplicity_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mphotonEnergyFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mneutralHadronEnergyFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mchargedHadronEnergyFraction_MiniAOD_over_Reco);
    for(unsigned int j=0;j<ME_MiniAOD_over_Reco.size();j++){
      MonitorElement* monReco=ME_Reco[j];if(monReco && monReco->getRootObject()){
	MonitorElement* monMiniAOD=ME_MiniAOD[j];if(monMiniAOD && monMiniAOD->getRootObject()){
	  MonitorElement* monMiniAOD_over_RECO=ME_MiniAOD_over_Reco[j];if(monMiniAOD_over_RECO && monMiniAOD_over_RECO->getRootObject()){
	    for(int i=0;i<=(monMiniAOD_over_RECO->getNbinsX()+1);i++){
	      if(monReco->getBinContent(i)!=0){
		monMiniAOD_over_RECO->setBinContent(i,monMiniAOD->getBinContent(i)/monReco->getBinContent(i));
	      }else if (monMiniAOD->getBinContent(i)!=0){
		monMiniAOD_over_RECO->setBinContent(i,-0.5);
	      }
	    }
	  }
	}
      }
    }
  }
}
