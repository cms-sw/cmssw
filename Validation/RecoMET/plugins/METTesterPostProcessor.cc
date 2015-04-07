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
  //bin definition for resolution plot -> last bin contains overflow too, but for plotting purposes show up to 1 TeV only
  int nBins = 11;
  float bins[] = {0.,20.,40.,60.,80.,100.,150.,200.,300.,400.,500.,1000};
  //loop over met subdirectories
  for (int i=0; i<int(met_dirs.size()); i++) {
    ibook_.setCurrentFolder(met_dirs[i]);  
    mMETDifference_GenMETTrue_METResolution = ibook_.book1D("METResolution_GenMETTrue_InMETBins","METResolution_GenMETTrue_InMETBins",nBins, bins); 
    FillMETRes(met_dirs[i],iget_);
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
  mMETDifference_GenMETTrue_MET500=0;

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
  mMETDifference_GenMETTrue_MET500 = iget.get(metdir+"/METResolution_GenMETTrue_MET500"); 
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
    mMETDifference_GenMETTrue_METResolution->setBinContent(11, mMETDifference_GenMETTrue_MET500->getMean()); 

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
    mMETDifference_GenMETTrue_METResolution->setBinError(11, mMETDifference_GenMETTrue_MET500->getRMS());
  }
}
