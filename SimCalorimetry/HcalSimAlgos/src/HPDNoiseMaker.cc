// --------------------------------------------------------
// Engine to store HPD noise events in the library
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseMaker.cc,v 1.5 2008/08/04 22:07:08 fedor Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseMaker.h"

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseDataCatalog.h"


#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include <iostream>

HPDNoiseMaker::HPDNoiseMaker (const std::string& fFileName) {
  mFile = new TFile (fFileName.c_str(), "RECREATE");
  mCatalog = new HPDNoiseDataCatalog ();
}

HPDNoiseMaker::~HPDNoiseMaker () {
  for (size_t i = 0; i < mTrees.size(); ++i) {
    mTrees[i]->Write();
    delete mTrees[i];
  }
  mFile->WriteObject (mCatalog, HPDNoiseDataCatalog::objectName ());
  delete mCatalog;
  delete mFile;
}

int HPDNoiseMaker::addHpd (const std::string& fName) {
  TDirectory* currentDirectory = gDirectory;
  mFile->cd();
  mCatalog->addHpd (fName, 0., 0.,0.,0.);
  mNames.push_back (fName);
  mTrees.push_back (new TTree (fName.c_str(), fName.c_str()));
  HPDNoiseData* addr = 0;
  TBranch* newBranch = mTrees.back()->Branch (HPDNoiseData::branchName(), HPDNoiseData::className(), &addr, 32000, 1);
  if (!newBranch) {
    std::cerr << "HPDNoiseMaker::addHpd-> Can not make branch HPDNoiseData to the tree " << fName << std::endl;
  }
  currentDirectory->cd();
  return mNames.size();
}

void HPDNoiseMaker::setRate (const std::string& fName, float fDischargeRate, 
                             float fIonFeedbackFirstPeakRate, float fIonFeedbackSecondPeakRate, float fElectronEmissionRate) {
  mCatalog->setRate (fName, fDischargeRate, fIonFeedbackFirstPeakRate, fIonFeedbackSecondPeakRate, fElectronEmissionRate);
}

void HPDNoiseMaker::newHpdEvent (const std::string& fName, const HPDNoiseData& fData) {
  for (size_t i = 0; i < mNames.size(); ++i) {
    if (mNames[i] == fName) {
      newHpdEvent (i, fData);
    }
  }
}

void HPDNoiseMaker::newHpdEvent (size_t i, const HPDNoiseData& fData) {
  if (i < mTrees.size()) {
    HPDNoiseData* data = (HPDNoiseData*) &fData;
    TBranch* branch = mTrees[i]->GetBranch (HPDNoiseData::branchName());
    if (branch) {
      branch->SetAddress(&data);
      mTrees[i]->Fill();
    }
    else {
      std::cerr << "HPDNoiseMaker::newHpdEvent-> Can not find branch " << HPDNoiseData::branchName() 
		<< " in the tree" << std::endl;
    }
  }
}

unsigned long HPDNoiseMaker::totalEntries (const std::string& fName) const {
  for (size_t i = 0; i < mNames.size(); ++i) {
    if (mNames[i] == fName) return mTrees[i]->GetEntries ();
  }
  return 0;
}
