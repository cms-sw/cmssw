// --------------------------------------------------------
// Engine to read HPD noise events from the library
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseReader.cc,v 1.4 2008/08/04 22:07:08 fedor Exp $
// --------------------------------------------------------

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseReader.h"

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"

#include <iostream>

#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseDataCatalog.h"

HPDNoiseReader::HPDNoiseReader (const std::string& fFileName) {
  mFile = new TFile (fFileName.c_str(), "READ");
  mBuffer = new HPDNoiseData ();
  HPDNoiseDataCatalog* catalog;
  mFile->GetObject (HPDNoiseDataCatalog::objectName(), catalog);
  if (catalog) {
    // initiate trees
    const std::vector<std::string> names = catalog->allNames();
    for (size_t i = 0; i < names.size(); ++i) {
      TTree* tree = (TTree*) mFile->Get (names[i].c_str());
      if (tree) {
	mTrees.push_back (tree);
	mDischargeRate.push_back (catalog->getDischargeRate (i));
	mIonFeedbackFirstPeakRate.push_back(catalog->getIonFeedbackFirstPeakRate(i));
	mIonFeedbackSecondPeakRate.push_back(catalog->getIonFeedbackSecondPeakRate(i));
        mElectronEmissionRate.push_back(catalog->getElectronEmissionRate(i));
	mCurrentEntry.push_back (0);
      }
      else {
	std::cerr << "HPDNoiseReader::HPDNoiseReader-> Can not open tree " << names[i] << " in file " << fFileName << std::endl;
      }
    }
  }
  else {
    std::cerr << "HPDNoiseReader::HPDNoiseReader-> Can not open catalog infile " << fFileName << std::endl;
  }
}

HPDNoiseReader::~HPDNoiseReader () {
  for (size_t i = 0; i < mTrees.size(); ++i) {
    delete mTrees[i];
  }
  delete mFile;
  delete mBuffer;
}


std::vector<std::string> HPDNoiseReader::allNames () const {
  std::vector<std::string> result;
  for (size_t i = 0; i < mTrees.size(); ++i) result.push_back (mTrees[i]->GetName ());
  return result;
}

HPDNoiseReader::Handle HPDNoiseReader::getHandle (const std::string& fName) {
  for (size_t i = 0; i < mTrees.size(); ++i) {
    if (std::string (mTrees[i]->GetName ()) == fName) return i;
  }
  return -1;
}

float HPDNoiseReader::dischargeRate (Handle fHandle) const {
  if (!valid (fHandle)) return 0;
  return mDischargeRate[fHandle];
}
float HPDNoiseReader::ionFeedbackFirstPeakRate (Handle fHandle) const {
  if (!valid (fHandle)) return 0;
  return mIonFeedbackFirstPeakRate[fHandle];
}
float HPDNoiseReader::ionFeedbackSecondPeakRate (Handle fHandle) const {
  if (!valid (fHandle)) return 0;
  return mIonFeedbackSecondPeakRate[fHandle];
}
float HPDNoiseReader::emissionRate (Handle fHandle) const {
  if (!valid (fHandle)) return 0;
  return mElectronEmissionRate[fHandle];
}

unsigned long HPDNoiseReader::totalEntries (Handle fHandle) const {
  if (!valid (fHandle)) return 0;
  return mTrees[fHandle]->GetEntries ();
}

void HPDNoiseReader::grabEntry (Handle fHandle, unsigned long fEntry) {
  if (!valid (fHandle)) return;
  TBranch* branch = mTrees[fHandle]->GetBranch (HPDNoiseData::branchName());
  branch->SetAddress (&mBuffer);
  branch->GetEntry (fEntry);
  mCurrentEntry [fHandle] = fEntry;
}

void HPDNoiseReader::getEntry (Handle fHandle, HPDNoiseData** fData) {
  if (!valid (fHandle)) return;
  unsigned int entry = mCurrentEntry [fHandle];
  if (++entry >= totalEntries (fHandle)) entry = 0; // roll over  
  grabEntry (fHandle, entry);
  *fData = mBuffer;
}

void HPDNoiseReader::getEntry (Handle fHandle, unsigned long fEntry, HPDNoiseData** fData) {
  grabEntry (fHandle, fEntry);
  *fData = mBuffer;
}

