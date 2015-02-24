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
    for(int i=0;i<=(mPt_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPt_Reco->getBinContent(i)!=0){
	mPt_MiniAOD_over_Reco->setBinContent(i,mPt_MiniAOD->getBinContent(i)/mPt_Reco->getBinContent(i));
      }else if(mPt_MiniAOD->getBinContent(i)!=0){
	mPt_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPhi_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPhi_Reco->getBinContent(i)!=0){
	mPhi_MiniAOD_over_Reco->setBinContent(i,mPhi_MiniAOD->getBinContent(i)/mPhi_Reco->getBinContent(i));
      }else if(mPhi_MiniAOD->getBinContent(i)!=0){
	mPhi_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mEta_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mEta_Reco->getBinContent(i)!=0){
	mEta_MiniAOD_over_Reco->setBinContent(i,mEta_MiniAOD->getBinContent(i)/mEta_Reco->getBinContent(i));
      }else if(mEta_MiniAOD->getBinContent(i)!=0){
	mEta_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mCorrJetPt_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mCorrJetPt_Reco->getBinContent(i)!=0){
	mCorrJetPt_MiniAOD_over_Reco->setBinContent(i,mCorrJetPt_MiniAOD->getBinContent(i)/mCorrJetPt_Reco->getBinContent(i));
      }else if(mCorrJetPt_MiniAOD->getBinContent(i)!=0){
	mCorrJetPt_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mCorrJetPhi_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mCorrJetPhi_Reco->getBinContent(i)!=0){
	mCorrJetPhi_MiniAOD_over_Reco->setBinContent(i,mCorrJetPhi_MiniAOD->getBinContent(i)/mCorrJetPhi_Reco->getBinContent(i));
      }else if(mCorrJetPhi_MiniAOD->getBinContent(i)!=0){
	mCorrJetPhi_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mCorrJetEta_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mCorrJetEta_Reco->getBinContent(i)!=0){
	mCorrJetEta_MiniAOD_over_Reco->setBinContent(i,mCorrJetEta_MiniAOD->getBinContent(i)/mCorrJetEta_Reco->getBinContent(i));
      }else if(mCorrJetEta_MiniAOD->getBinContent(i)!=0){
	mCorrJetEta_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPtCorrOverReco_Eta_20_40_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPtCorrOverReco_Eta_20_40_Reco->getBinContent(i)!=0){
	double value=mPtCorrOverReco_Eta_20_40_MiniAOD->getBinContent(i)/mPtCorrOverReco_Eta_20_40_Reco->getBinContent(i);
	mPtCorrOverReco_Eta_20_40_MiniAOD_over_Reco->setBinContent(i,value);
      }else if(mPtCorrOverReco_Eta_20_40_MiniAOD->getBinContent(i)!=0){
	mPtCorrOverReco_Eta_20_40_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    } 
    for(int i=0;i<=(mPtCorrOverReco_Eta_200_600_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPtCorrOverReco_Eta_200_600_Reco->getBinContent(i)!=0){
	mPtCorrOverReco_Eta_200_600_MiniAOD_over_Reco->setBinContent(i,mPtCorrOverReco_Eta_200_600_MiniAOD->getBinContent(i)/mPtCorrOverReco_Eta_200_600_Reco->getBinContent(i));
      }else if(mPtCorrOverReco_Eta_200_600_MiniAOD->getBinContent(i)!=0){
	mPtCorrOverReco_Eta_200_600_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPtCorrOverReco_Eta_1500_3500_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPtCorrOverReco_Eta_1500_3500_Reco->getBinContent(i)!=0){
	mPtCorrOverReco_Eta_1500_3500_MiniAOD_over_Reco->setBinContent(i,mPtCorrOverReco_Eta_1500_3500_MiniAOD->getBinContent(i)/mPtCorrOverReco_Eta_1500_3500_Reco->getBinContent(i));
      }else if(mPtCorrOverReco_Eta_1500_3500_MiniAOD->getBinContent(i)!=0){
	mPtCorrOverReco_Eta_1500_3500_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    } 
    for(int i=0;i<=(mPtCorrOverGen_GenEta_40_200_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPtCorrOverGen_GenEta_40_200_Reco->getBinContent(i)!=0){
	mPtCorrOverGen_GenEta_40_200_MiniAOD_over_Reco->setBinContent(i,mPtCorrOverGen_GenEta_40_200_MiniAOD->getBinContent(i)/mPtCorrOverGen_GenEta_40_200_Reco->getBinContent(i));
      }else if(mPtCorrOverGen_GenEta_40_200_MiniAOD->getBinContent(i)!=0){
	mPtCorrOverGen_GenEta_40_200_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPtCorrOverGen_GenEta_600_1500_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPtCorrOverGen_GenEta_600_1500_Reco->getBinContent(i)!=0){
	mPtCorrOverGen_GenEta_600_1500_MiniAOD_over_Reco->setBinContent(i,mPtCorrOverGen_GenEta_600_1500_MiniAOD->getBinContent(i)/mPtCorrOverGen_GenEta_600_1500_Reco->getBinContent(i));
      }else if(mPtCorrOverGen_GenEta_600_1500_MiniAOD->getBinContent(i)!=0){
	mPtCorrOverGen_GenEta_600_1500_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mDeltaPt_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mDeltaPt_Reco->getBinContent(i)!=0){
	mDeltaPt_MiniAOD_over_Reco->setBinContent(i,mDeltaPt_MiniAOD->getBinContent(i)/mDeltaPt_Reco->getBinContent(i));
      }else if(mDeltaPt_MiniAOD->getBinContent(i)!=0){
	mDeltaPt_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mDeltaPhi_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mDeltaPhi_Reco->getBinContent(i)!=0){
	mDeltaPhi_MiniAOD_over_Reco->setBinContent(i,mDeltaPhi_MiniAOD->getBinContent(i)/mDeltaPhi_Reco->getBinContent(i));
      }else if(mDeltaPhi_MiniAOD->getBinContent(i)!=0){
	mDeltaPhi_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mDeltaEta_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mDeltaEta_Reco->getBinContent(i)!=0){
	mDeltaEta_MiniAOD_over_Reco->setBinContent(i,mDeltaEta_MiniAOD->getBinContent(i)/mDeltaEta_Reco->getBinContent(i));
      }else if(mDeltaEta_MiniAOD->getBinContent(i)!=0){
	mDeltaEta_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMjj_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMjj_Reco->getBinContent(i)!=0){
	mMjj_MiniAOD_over_Reco->setBinContent(i,mMjj_MiniAOD->getBinContent(i)/mMjj_Reco->getBinContent(i));
      }else if(mMjj_MiniAOD->getBinContent(i)!=0){
	mMjj_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mNJets40_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mNJets40_Reco->getBinContent(i)!=0){
	mNJets40_MiniAOD_over_Reco->setBinContent(i,mNJets40_MiniAOD->getBinContent(i)/mNJets40_Reco->getBinContent(i));
      }else if(mNJets40_MiniAOD->getBinContent(i)!=0){
	mNJets40_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mchargedHadronMultiplicity_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mchargedHadronMultiplicity_Reco->getBinContent(i)!=0){
	mchargedHadronMultiplicity_MiniAOD_over_Reco->setBinContent(i,mchargedHadronMultiplicity_MiniAOD->getBinContent(i)/mchargedHadronMultiplicity_Reco->getBinContent(i));
      }else if(mchargedHadronMultiplicity_MiniAOD->getBinContent(i)!=0){
	mchargedHadronMultiplicity_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mneutralHadronMultiplicity_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mneutralHadronMultiplicity_Reco->getBinContent(i)!=0){
	mneutralHadronMultiplicity_MiniAOD_over_Reco->setBinContent(i,mneutralHadronMultiplicity_MiniAOD->getBinContent(i)/mneutralHadronMultiplicity_Reco->getBinContent(i));
      }else if(mneutralHadronMultiplicity_MiniAOD->getBinContent(i)!=0){
	mneutralHadronMultiplicity_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mphotonMultiplicity_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mphotonMultiplicity_Reco->getBinContent(i)!=0){
	mphotonMultiplicity_MiniAOD_over_Reco->setBinContent(i,mphotonMultiplicity_MiniAOD->getBinContent(i)/mphotonMultiplicity_Reco->getBinContent(i));
      }else if(mphotonMultiplicity_MiniAOD->getBinContent(i)!=0){
	mphotonMultiplicity_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mchargedHadronEnergyFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mchargedHadronEnergyFraction_Reco->getBinContent(i)!=0){
	mchargedHadronEnergyFraction_MiniAOD_over_Reco->setBinContent(i,mchargedHadronEnergyFraction_MiniAOD->getBinContent(i)/mchargedHadronEnergyFraction_Reco->getBinContent(i));
      }else if(mchargedHadronEnergyFraction_MiniAOD->getBinContent(i)!=0){
	mchargedHadronEnergyFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mneutralHadronEnergyFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mneutralHadronEnergyFraction_Reco->getBinContent(i)!=0){
	mneutralHadronEnergyFraction_MiniAOD_over_Reco->setBinContent(i,mneutralHadronEnergyFraction_MiniAOD->getBinContent(i)/mneutralHadronEnergyFraction_Reco->getBinContent(i));
      }else if(mneutralHadronEnergyFraction_MiniAOD->getBinContent(i)!=0){
	mneutralHadronEnergyFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mphotonEnergyFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mphotonEnergyFraction_Reco->getBinContent(i)!=0){
	mphotonEnergyFraction_MiniAOD_over_Reco->setBinContent(i,mphotonEnergyFraction_MiniAOD->getBinContent(i)/mphotonEnergyFraction_Reco->getBinContent(i));
      }else if(mphotonEnergyFraction_MiniAOD->getBinContent(i)!=0){
	mphotonEnergyFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
  }
}
