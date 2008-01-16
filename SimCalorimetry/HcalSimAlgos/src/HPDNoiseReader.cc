// --------------------------------------------------------
// Engine to read HPD noise events from the library
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: BasicJet.h,v 1.11 2007/09/20 21:04:43 fedor Exp $
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
  mFile->GetObject ("HPDNoiseDataCatalog;1", catalog);
  if (catalog) {
    std::cout << "HPDNoiseReader::HPDNoiseReader-> catalog: " << *catalog << std::endl;
    // initiate trees
    const std::vector<std::string> names = catalog->allNames();
    for (size_t i = 0; i < names.size(); ++i) {
      TTree* tree = (TTree*) mFile->Get (names[i].c_str());
      if (tree) {
	mTrees.push_back (tree);
	mRates.push_back (catalog->getRate (i));
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

float HPDNoiseReader::rate (Handle fHandle) const {
  if (!valid (fHandle)) return 0;
  return mRates[fHandle];
}

unsigned long HPDNoiseReader::totalEntries (Handle fHandle) const {
  if (!valid (fHandle)) return 0;
  return mTrees[fHandle]->GetEntries ();
}

void HPDNoiseReader::grabEntry (Handle fHandle, unsigned long fEntry) {
  if (!valid (fHandle)) return;
  TBranch* branch = mTrees[fHandle]->GetBranch ("HPDNoiseData");
  branch->SetAddress (&mBuffer);
  branch->GetEntry (fEntry);
}

void HPDNoiseReader::getEntry (Handle fHandle, unsigned long fEntry, HPDNoiseData** fData) {
  grabEntry (fHandle, fEntry);
  *fData = mBuffer;
}

