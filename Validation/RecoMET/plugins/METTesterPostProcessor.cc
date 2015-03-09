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

    std::vector<MonitorElement*>ME_Reco;
    ME_Reco.push_back(mMET_Reco);
    ME_Reco.push_back(mMETPhi_Reco);
    ME_Reco.push_back(mSumET_Reco);
    ME_Reco.push_back(mMETDifference_GenMETTrue_Reco);
    ME_Reco.push_back(mMETDeltaPhi_GenMETTrue_Reco);
    ME_Reco.push_back(mPFPhotonEtFraction_Reco);
    ME_Reco.push_back(mPFNeutralHadronEtFraction_Reco);
    ME_Reco.push_back(mPFChargedHadronEtFraction_Reco);
    ME_Reco.push_back(mPFHFHadronEtFraction_Reco);
    ME_Reco.push_back(mPFHFEMEtFraction_Reco);
    ME_Reco.push_back(mMETDifference_GenMETTrue_MET20to40_Reco);
    ME_Reco.push_back(mMETDifference_GenMETTrue_MET100to150_Reco);
    ME_Reco.push_back(mMETDifference_GenMETTrue_MET300to400_Reco);


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

    std::vector<MonitorElement*>ME_MiniAOD;
    ME_MiniAOD.push_back(mMET_MiniAOD);
    ME_MiniAOD.push_back(mMETPhi_MiniAOD);
    ME_MiniAOD.push_back(mSumET_MiniAOD);
    ME_MiniAOD.push_back(mMETDifference_GenMETTrue_MiniAOD);
    ME_MiniAOD.push_back(mMETDeltaPhi_GenMETTrue_MiniAOD);
    ME_MiniAOD.push_back(mPFPhotonEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFNeutralHadronEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFChargedHadronEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFHFHadronEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mPFHFEMEtFraction_MiniAOD);
    ME_MiniAOD.push_back(mMETDifference_GenMETTrue_MET20to40_MiniAOD);
    ME_MiniAOD.push_back(mMETDifference_GenMETTrue_MET100to150_MiniAOD);
    ME_MiniAOD.push_back(mMETDifference_GenMETTrue_MET300to400_MiniAOD);

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

    std::vector<MonitorElement*>ME_MiniAOD_over_Reco;
    ME_MiniAOD_over_Reco.push_back(mMET_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETPhi_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mSumET_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDifference_GenMETTrue_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDeltaPhi_GenMETTrue_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFPhotonEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFNeutralHadronEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFChargedHadronEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFHFHadronEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mPFHFEMEtFraction_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDifference_GenMETTrue_MET20to40_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDifference_GenMETTrue_MET100to150_MiniAOD_over_Reco);
    ME_MiniAOD_over_Reco.push_back(mMETDifference_GenMETTrue_MET300to400_MiniAOD_over_Reco);

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
