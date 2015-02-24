// -*- C++ -*-
//
// Package:    Validation/RecoMET
// Class:      METTesterPostProcessor
// 
// Original Author:  "Matthias Weber"
//         Created:  Sun Feb 22 14:35:25 CET 2015
//

#include "Validation/RecoMET/plugins/METTesterPostProcessor.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Run.h"

// Some switches
//
// constructors and destructor
//
METTesterPostProcessor::METTesterPostProcessor(const edm::ParameterSet& iConfig)
{
  inputMETLabelRECO_=iConfig.getParameter<edm::InputTag>("METTypeRECO");
  inputMETLabelMiniAOD_=iConfig.getParameter<edm::InputTag>("METTypeMiniAOD");
}


METTesterPostProcessor::~METTesterPostProcessor()
{ 
}


// ------------ method called right after a run ends ------------
void 
METTesterPostProcessor::dqmEndJob(DQMStore::IBooker& ibook_, DQMStore::IGetter& iget_)
{
  std::vector<std::string> subDirVec;
  std::string RunDir="JetMET/METValidation/";
  iget_.setCurrentFolder(RunDir);
  met_dirs=iget_.getSubdirs();
  //bin definition for resolution plot
  int nBins = 10;
  float bins[] = {0.,20.,40.,60.,80.,100.,150.,200.,300.,400.,500.};
  bool found_reco_dir=false;
  bool found_miniaod_dir=false;
  //loop over met subdirectories
  for (int i=0; i<int(met_dirs.size()); i++) {
    ibook_.setCurrentFolder(met_dirs[i]);  
    mMETDifference_GenMETTrue_METResolution = ibook_.book1D("METResolution_GenMETTrue_InMETBins","METResolution_GenMETTrue_InMETBins",nBins, bins); 
    FillMETRes(met_dirs[i],iget_);
    if(met_dirs[i]==(RunDir+inputMETLabelRECO_.label())){
      found_reco_dir=true;
    }
    if(met_dirs[i]==(RunDir+inputMETLabelMiniAOD_.label())){
      found_miniaod_dir=true;
    }
  }
  if(found_miniaod_dir && found_reco_dir){
    std::string rundir_reco=RunDir+inputMETLabelRECO_.label();
    std::string rundir_miniaod=RunDir+inputMETLabelMiniAOD_.label();
    MonitorElement* mMET_Reco=iget_.get(rundir_reco+"/"+"MET");
    MonitorElement* mMETPhi_Reco=iget_.get(rundir_reco+"/"+"METPhi");
    MonitorElement* mSumET_Reco=iget_.get(rundir_reco+"/"+"SumET");
    MonitorElement* mMETDifference_GenMETTrue_Reco=iget_.get(rundir_reco+"/"+"METDifference_GenMETTrue");
    MonitorElement* mMETDeltaPhi_GenMETTrue_Reco=iget_.get(rundir_reco+"/"+"METDeltaPhi_GenMETTrue"); 
    MonitorElement* mPFPhotonEtFraction_Reco=iget_.get(rundir_reco+"/"+"photonEtFraction"); 
    MonitorElement* mPFNeutralHadronEtFraction_Reco=iget_.get(rundir_reco+"/"+"neutralHadronEtFraction"); 
    MonitorElement* mPFChargedHadronEtFraction_Reco=iget_.get(rundir_reco+"/"+"chargedHadronEtFraction"); 
    MonitorElement* mPFHFHadronEtFraction_Reco=iget_.get(rundir_reco+"/"+"HFHadronEtFraction"); 
    MonitorElement* mPFHFEMEtFraction_Reco=iget_.get(rundir_reco+"/"+"HFEMEtFraction"); 
    MonitorElement* mMETDifference_GenMETTrue_MET20to40_Reco=iget_.get(rundir_reco+"/"+"METResolution_GenMETTrue_MET20to40");
    MonitorElement* mMETDifference_GenMETTrue_MET100to150_Reco=iget_.get(rundir_reco+"/"+"METResolution_GenMETTrue_MET100to150");
    MonitorElement* mMETDifference_GenMETTrue_MET300to400_Reco=iget_.get(rundir_reco+"/"+"METResolution_GenMETTrue_MET300to400");

    MonitorElement* mMET_MiniAOD=iget_.get(rundir_miniaod+"/"+"MET");
    MonitorElement* mMETPhi_MiniAOD=iget_.get(rundir_miniaod+"/"+"METPhi");
    MonitorElement* mSumET_MiniAOD=iget_.get(rundir_miniaod+"/"+"SumET");
    MonitorElement* mMETDifference_GenMETTrue_MiniAOD=iget_.get(rundir_miniaod+"/"+"METDifference_GenMETTrue");
    MonitorElement* mMETDeltaPhi_GenMETTrue_MiniAOD=iget_.get(rundir_miniaod+"/"+"METDeltaPhi_GenMETTrue"); 
    MonitorElement* mPFPhotonEtFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"photonEtFraction"); 
    MonitorElement* mPFNeutralHadronEtFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"neutralHadronEtFraction"); 
    MonitorElement* mPFChargedHadronEtFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"chargedHadronEtFraction"); 
    MonitorElement* mPFHFHadronEtFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"HFHadronEtFraction"); 
    MonitorElement* mPFHFEMEtFraction_MiniAOD=iget_.get(rundir_miniaod+"/"+"HFEMEtFraction"); 
    MonitorElement* mMETDifference_GenMETTrue_MET20to40_MiniAOD=iget_.get(rundir_miniaod+"/"+"METResolution_GenMETTrue_MET20to40");
    MonitorElement* mMETDifference_GenMETTrue_MET100to150_MiniAOD=iget_.get(rundir_miniaod+"/"+"METResolution_GenMETTrue_MET100to150");
    MonitorElement* mMETDifference_GenMETTrue_MET300to400_MiniAOD=iget_.get(rundir_miniaod+"/"+"METResolution_GenMETTrue_MET300to400");

    ibook_.setCurrentFolder(RunDir+"MiniAOD_over_RECO");
    mMET_MiniAOD_over_Reco=ibook_.book1D("MET_MiniAOD_over_RECO",(TH1F*)mMET_Reco->getRootObject());
    mMETPhi_MiniAOD_over_Reco=ibook_.book1D("METPhi_MiniAOD_over_RECO",(TH1F*)mMETPhi_Reco->getRootObject());
    mSumET_MiniAOD_over_Reco=ibook_.book1D("SumET_MiniAOD_over_RECO",(TH1F*)mSumET_Reco->getRootObject());
    mMETDifference_GenMETTrue_MiniAOD_over_Reco=ibook_.book1D("METDifference_GenMETTrue_MiniAOD_over_RECO",(TH1F*)mMETDifference_GenMETTrue_Reco->getRootObject());
    mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco=ibook_.book1D("METDeltaPhi_GenMETTrue_MiniAOD_over_RECO",(TH1F*)mMETDeltaPhi_GenMETTrue_Reco->getRootObject());
    mPFPhotonEtFraction_MiniAOD_over_Reco=ibook_.book1D("photonEtFraction_MiniAOD_over_RECO",(TH1F*)mPFPhotonEtFraction_Reco->getRootObject());
    mPFNeutralHadronEtFraction_MiniAOD_over_Reco=ibook_.book1D("neutralHadronEtFraction_MiniAOD_over_RECO",(TH1F*)mPFNeutralHadronEtFraction_Reco->getRootObject());
    mPFChargedHadronEtFraction_MiniAOD_over_Reco=ibook_.book1D("chargedHadronEtFraction_MiniAOD_over_RECO",(TH1F*)mPFChargedHadronEtFraction_Reco->getRootObject());
    mPFHFHadronEtFraction_MiniAOD_over_Reco=ibook_.book1D("HFHadronEtFraction_MiniAOD_over_RECO",(TH1F*)mPFHFHadronEtFraction_Reco->getRootObject());
    mPFHFEMEtFraction_MiniAOD_over_Reco=ibook_.book1D("HFEMEtEtFraction_MiniAOD_over_RECO",(TH1F*)mPFHFEMEtFraction_Reco->getRootObject());
    mMETDifference_GenMETTrue_MET20to40_MiniAOD_over_Reco=ibook_.book1D("METResolution_GenMETTrue_MET20to40_MiniAOD_over_RECO",(TH1F*)mMETDifference_GenMETTrue_MET20to40_Reco->getRootObject());
    mMETDifference_GenMETTrue_MET100to150_MiniAOD_over_Reco=ibook_.book1D("METResolution_GenMETTrue_MET100to150_MiniAOD_over_RECO",(TH1F*)mMETDifference_GenMETTrue_MET100to150_Reco->getRootObject());
    mMETDifference_GenMETTrue_MET300to400_MiniAOD_over_Reco=ibook_.book1D("METResolution_GenMETTrue_MET300to400_MiniAOD_over_RECO",(TH1F*)mMETDifference_GenMETTrue_MET300to400_Reco->getRootObject());
    for(int i=0;i<=(mMET_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMET_Reco->getBinContent(i)!=0){
	mMET_MiniAOD_over_Reco->setBinContent(i,mMET_MiniAOD->getBinContent(i)/mMET_Reco->getBinContent(i));
      }else if(mMET_MiniAOD->getBinContent(i)!=0){
	mMET_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMETPhi_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMETPhi_Reco->getBinContent(i)!=0){
	mMETPhi_MiniAOD_over_Reco->setBinContent(i,mMETPhi_MiniAOD->getBinContent(i)/mMETPhi_Reco->getBinContent(i));
      }else if(mMETPhi_MiniAOD->getBinContent(i)!=0){
	mMETPhi_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mSumET_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mSumET_Reco->getBinContent(i)!=0){
	mSumET_MiniAOD_over_Reco->setBinContent(i,mSumET_MiniAOD->getBinContent(i)/mSumET_Reco->getBinContent(i));
      }else if(mSumET_MiniAOD->getBinContent(i)!=0){
	mSumET_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMETDifference_GenMETTrue_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMETDifference_GenMETTrue_Reco->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MiniAOD_over_Reco->setBinContent(i,mMETDifference_GenMETTrue_MiniAOD->getBinContent(i)/mMETDifference_GenMETTrue_Reco->getBinContent(i));
      }else if(mMETDifference_GenMETTrue_MiniAOD->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMETDeltaPhi_GenMETTrue_Reco->getBinContent(i)!=0){
	mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco->setBinContent(i,mMETDeltaPhi_GenMETTrue_MiniAOD->getBinContent(i)/mMETDeltaPhi_GenMETTrue_Reco->getBinContent(i));
      }else if(mMETDeltaPhi_GenMETTrue_MiniAOD->getBinContent(i)!=0){
	mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPFPhotonEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPFPhotonEtFraction_Reco->getBinContent(i)!=0){
	mPFPhotonEtFraction_MiniAOD_over_Reco->setBinContent(i,mPFPhotonEtFraction_MiniAOD->getBinContent(i)/mPFPhotonEtFraction_Reco->getBinContent(i));
      }else if(mPFPhotonEtFraction_MiniAOD->getBinContent(i)!=0){
	mPFPhotonEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPFNeutralHadronEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPFNeutralHadronEtFraction_Reco->getBinContent(i)!=0){
	mPFNeutralHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,mPFNeutralHadronEtFraction_MiniAOD->getBinContent(i)/mPFNeutralHadronEtFraction_Reco->getBinContent(i));
      }else if(mPFNeutralHadronEtFraction_MiniAOD->getBinContent(i)!=0){
	mPFNeutralHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPFChargedHadronEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPFChargedHadronEtFraction_Reco->getBinContent(i)!=0){
	mPFChargedHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,mPFChargedHadronEtFraction_MiniAOD->getBinContent(i)/mPFChargedHadronEtFraction_Reco->getBinContent(i));
      }else if(mPFChargedHadronEtFraction_MiniAOD->getBinContent(i)!=0){
	mPFChargedHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPFHFHadronEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPFHFHadronEtFraction_Reco->getBinContent(i)!=0){
	mPFHFHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,mPFHFHadronEtFraction_MiniAOD->getBinContent(i)/mPFHFHadronEtFraction_Reco->getBinContent(i));
      }else if(mPFHFHadronEtFraction_MiniAOD->getBinContent(i)!=0){
	mPFHFHadronEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mPFHFEMEtFraction_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mPFHFEMEtFraction_Reco->getBinContent(i)!=0){
	mPFHFEMEtFraction_MiniAOD_over_Reco->setBinContent(i,mPFHFEMEtFraction_MiniAOD->getBinContent(i)/mPFHFEMEtFraction_Reco->getBinContent(i));
      }else if(mPFHFEMEtFraction_MiniAOD->getBinContent(i)!=0){
	mPFHFEMEtFraction_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMETDifference_GenMETTrue_MET20to40_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMETDifference_GenMETTrue_MET20to40_Reco->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MET20to40_MiniAOD_over_Reco->setBinContent(i,mMETDifference_GenMETTrue_MET20to40_MiniAOD->getBinContent(i)/mMETDifference_GenMETTrue_MET20to40_Reco->getBinContent(i));
      }else if(mMETDifference_GenMETTrue_MET20to40_MiniAOD->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MET20to40_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMETDifference_GenMETTrue_MET100to150_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMETDifference_GenMETTrue_MET100to150_Reco->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MET100to150_MiniAOD_over_Reco->setBinContent(i,mMETDifference_GenMETTrue_MET100to150_MiniAOD->getBinContent(i)/mMETDifference_GenMETTrue_MET100to150_Reco->getBinContent(i));
      }else if(mMETDifference_GenMETTrue_MET100to150_MiniAOD->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MET100to150_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
    for(int i=0;i<=(mMETDifference_GenMETTrue_MET300to400_MiniAOD_over_Reco->getNbinsX()+1);i++){
      if(mMETDifference_GenMETTrue_MET300to400_Reco->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MET300to400_MiniAOD_over_Reco->setBinContent(i,mMETDifference_GenMETTrue_MET300to400_MiniAOD->getBinContent(i)/mMETDifference_GenMETTrue_MET300to400_Reco->getBinContent(i));
      }else if(mMETDifference_GenMETTrue_MET300to400_MiniAOD->getBinContent(i)!=0){
	mMETDifference_GenMETTrue_MET300to400_MiniAOD_over_Reco->setBinContent(i,-0.5);
      }
    }
  }
}


void METTesterPostProcessor::FillMETRes(std::string metdir, DQMStore::IGetter & iget)
{

  mMETDifference_GenMETTrue_MET0to20=0;
  mMETDifference_GenMETTrue_MET20to40=0;
  mMETDifference_GenMETTrue_MET40to60=0;
  mMETDifference_GenMETTrue_MET60to80=0;
  mMETDifference_GenMETTrue_MET80to100=0;
  mMETDifference_GenMETTrue_MET100to150=0;
  mMETDifference_GenMETTrue_MET150to200=0;
  mMETDifference_GenMETTrue_MET200to300=0;
  mMETDifference_GenMETTrue_MET300to400=0;
  mMETDifference_GenMETTrue_MET400to500=0;
 

  mMETDifference_GenMETTrue_MET0to20 = iget.get(metdir+"/METResolution_GenMETTrue_MET0to20");
  mMETDifference_GenMETTrue_MET20to40 = iget.get(metdir+"/METResolution_GenMETTrue_MET20to40");
  mMETDifference_GenMETTrue_MET40to60 = iget.get(metdir+"/METResolution_GenMETTrue_MET40to60");
  mMETDifference_GenMETTrue_MET60to80 = iget.get(metdir+"/METResolution_GenMETTrue_MET60to80");
  mMETDifference_GenMETTrue_MET80to100 = iget.get(metdir+"/METResolution_GenMETTrue_MET80to100");
  mMETDifference_GenMETTrue_MET100to150 = iget.get(metdir+"/METResolution_GenMETTrue_MET100to150");
  mMETDifference_GenMETTrue_MET150to200 = iget.get(metdir+"/METResolution_GenMETTrue_MET150to200");
  mMETDifference_GenMETTrue_MET200to300 = iget.get(metdir+"/METResolution_GenMETTrue_MET200to300");
  mMETDifference_GenMETTrue_MET300to400 = iget.get(metdir+"/METResolution_GenMETTrue_MET300to400");
  mMETDifference_GenMETTrue_MET400to500 = iget.get(metdir+"/METResolution_GenMETTrue_MET400to500"); 
  if(mMETDifference_GenMETTrue_MET0to20 && mMETDifference_GenMETTrue_MET0to20->getRootObject()){//check one object, if existing, then the remaining ME's exist too
    //for genmet none of these ME's are filled
    mMETDifference_GenMETTrue_METResolution->setBinContent(1, mMETDifference_GenMETTrue_MET0to20->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(2, mMETDifference_GenMETTrue_MET20to40->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(3, mMETDifference_GenMETTrue_MET40to60->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(4, mMETDifference_GenMETTrue_MET60to80->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(5, mMETDifference_GenMETTrue_MET80to100->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(6, mMETDifference_GenMETTrue_MET100to150->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(7, mMETDifference_GenMETTrue_MET150to200->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(8, mMETDifference_GenMETTrue_MET200to300->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(9, mMETDifference_GenMETTrue_MET300to400->getMean());
    mMETDifference_GenMETTrue_METResolution->setBinContent(10, mMETDifference_GenMETTrue_MET400to500->getMean());
    
    //the error computation should be done in a postProcessor in the harvesting step otherwise the histograms will be just summed
    mMETDifference_GenMETTrue_METResolution->setBinError(1, mMETDifference_GenMETTrue_MET0to20->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(2, mMETDifference_GenMETTrue_MET20to40->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(3, mMETDifference_GenMETTrue_MET40to60->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(4, mMETDifference_GenMETTrue_MET60to80->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(5, mMETDifference_GenMETTrue_MET80to100->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(6, mMETDifference_GenMETTrue_MET100to150->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(7, mMETDifference_GenMETTrue_MET150to200->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(8, mMETDifference_GenMETTrue_MET200to300->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(9, mMETDifference_GenMETTrue_MET300to400->getRMS());
    mMETDifference_GenMETTrue_METResolution->setBinError(10, mMETDifference_GenMETTrue_MET400to500->getRMS());
  }
}
