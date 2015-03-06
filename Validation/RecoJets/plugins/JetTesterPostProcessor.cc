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

    MonitorElement* mPt_Reco=iget_.get(rundir_reco+"/"+"Pt");
    MonitorElement* mPhi_Reco=iget_.get(rundir_reco+"/"+"Phi");
    MonitorElement* mEta_Reco=iget_.get(rundir_reco+"/"+"Eta");
    MonitorElement* mCorrJetPt_Reco=iget_.get(rundir_reco+"/"+"CorrJetPt");
    MonitorElement* mCorrJetPhi_Reco=iget_.get(rundir_reco+"/"+"CorrJetPhi");
    MonitorElement* mCorrJetEta_Reco=iget_.get(rundir_reco+"/"+"CorrJetEta");

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
    
    map_string_vec.push_back("PtCorrOverReco_Eta_20_40");
    map_string_vec.push_back("PtCorrOverReco_Eta_200_600");
    map_string_vec.push_back("PtCorrOverReco_Eta_1500_3500");
    map_string_vec.push_back("PtCorrOverGen_GenEta_40_200");
    map_string_vec.push_back("PtCorrOverGen_GenEta_600_1500");
    map_string_vec.push_back("DeltaEta");
    map_string_vec.push_back("DeltaPhi");
    map_string_vec.push_back("DeltaPt");
    map_string_vec.push_back("Mjj");
    map_string_vec.push_back("NJets");
    map_string_vec.push_back("chargedHadronMultiplicity");
    map_string_vec.push_back("neutralHadronMultiplicity");
    map_string_vec.push_back("photonMultiplicity");
    map_string_vec.push_back("chargedHadronEnergyFraction");
    map_string_vec.push_back("neutralHadronEnergyFraction");
    map_string_vec.push_back("photonEnergyFraction");

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"PtCorrOverReco_Eta_20_40" ,mPtCorrOverReco_Eta_20_40_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"PtCorrOverReco_Eta_200_600" ,mPtCorrOverReco_Eta_200_600_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"PtCorrOverReco_Eta_1500_3500" ,mPtCorrOverReco_Eta_1500_3500_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"PtCorrOverGen_GenEta_40_200" ,mPtCorrOverGen_GenEta_40_200_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"PtCorrOverGen_GenEta_600_1500" ,mPtCorrOverGen_GenEta_600_1500_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"DeltaEta" ,mDeltaEta_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"DeltaPhi" ,mDeltaPhi_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"DeltaPt" ,mDeltaPt_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"Mjj" ,mMjj_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"NJets" ,mNJets40_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"chargedHadronMultiplicity" ,mchargedHadronMultiplicity_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"neutralHadronMultiplicity" ,mneutralHadronMultiplicity_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"photonMultiplicity" ,mphotonMultiplicity_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"chargedHadronEnergyFraction" ,mchargedHadronEnergyFraction_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"neutralHadronEnergyFraction" ,mneutralHadronEnergyFraction_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_reco+"/"+"photonEnergyFraction" ,mphotonEnergyFraction_Reco));

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

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"Pt" ,mPt_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"Phi" ,mPhi_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"Eta" ,mEta_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"CorrJetPt"  ,mCorrJetPt_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"CorrJetPhi" ,mCorrJetPhi_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"CorrJetEta" ,mCorrJetEta_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"PtCorrOverReco_Eta_20_40" ,mPtCorrOverReco_Eta_20_40_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"PtCorrOverReco_Eta_200_600" ,mPtCorrOverReco_Eta_200_600_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"PtCorrOverReco_Eta_1500_3500" ,mPtCorrOverReco_Eta_1500_3500_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"PtCorrOverGen_GenEta_40_200" ,mPtCorrOverGen_GenEta_40_200_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"PtCorrOverGen_GenEta_600_1500" ,mPtCorrOverGen_GenEta_600_1500_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"DeltaEta" ,mDeltaEta_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"DeltaPhi" ,mDeltaPhi_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"DeltaPt" ,mDeltaPt_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"Mjj" ,mMjj_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"NJets" ,mNJets40_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"chargedHadronMultiplicity" ,mchargedHadronMultiplicity_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"neutralHadronMultiplicity" ,mneutralHadronMultiplicity_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"photonMultiplicity" ,mphotonMultiplicity_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"chargedHadronEnergyFraction" ,mchargedHadronEnergyFraction_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"neutralHadronEnergyFraction" ,mneutralHadronEnergyFraction_MiniAOD));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(rundir_miniaod+"/"+"photonEnergyFraction" ,mphotonEnergyFraction_MiniAOD));

    ibook_.setCurrentFolder(RunDir+"MiniAOD_over_RECO");
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

    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"Pt" ,mPt_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"Phi" ,mPhi_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"Eta" ,mEta_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"CorrJetPt"  ,mCorrJetPt_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"CorrJetPhi" ,mCorrJetPhi_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"CorrJetEta" ,mCorrJetEta_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"PtCorrOverReco_Eta_20_40" ,mPtCorrOverReco_Eta_20_40_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"PtCorrOverReco_Eta_200_600" ,mPtCorrOverReco_Eta_200_600_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"PtCorrOverReco_Eta_1500_3500" ,mPtCorrOverReco_Eta_1500_3500_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"PtCorrOverGen_GenEta_40_200" ,mPtCorrOverGen_GenEta_40_200_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"PtCorrOverGen_GenEta_600_1500" ,mPtCorrOverGen_GenEta_600_1500_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"DeltaEta" ,mDeltaEta_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"DeltaPhi" ,mDeltaPhi_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"DeltaPt" ,mDeltaPt_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"Mjj" ,mMjj_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"NJets" ,mNJets40_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"chargedHadronMultiplicity" ,mchargedHadronMultiplicity_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"neutralHadronMultiplicity" ,mneutralHadronMultiplicity_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"photonMultiplicity" ,mphotonMultiplicity_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"chargedHadronEnergyFraction" ,mchargedHadronEnergyFraction_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"neutralHadronEnergyFraction" ,mneutralHadronEnergyFraction_MiniAOD_over_Reco));
    map_of_MEs.insert(std::pair<std::string,MonitorElement*>(RunDir+"MiniAOD_over_RECO"+"/"+"photonEnergyFraction" ,mphotonEnergyFraction_MiniAOD_over_Reco));

    for(unsigned int j=0;j<map_string_vec.size();j++){
      MonitorElement* monReco=map_of_MEs[rundir_reco+"/"+map_string_vec[j]];if(monReco && monReco->getRootObject()){
	MonitorElement* monMiniAOD=map_of_MEs[rundir_miniaod+"/"+map_string_vec[j]];if(monMiniAOD && monMiniAOD->getRootObject()){
	  MonitorElement* monMiniAOD_over_RECO=map_of_MEs[RunDir+"MiniAOD_over_RECO"+"/"+map_string_vec[j]];if(monMiniAOD_over_RECO && monMiniAOD_over_RECO->getRootObject()){
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
