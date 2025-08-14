// -*- C++ -*-
//
// Package:    Validation/RecoMET
// Class:      METTesterPostProcessor
//
// Original Author:  "Matthias Weber"
//         Created:  Sun Feb 22 14:35:25 CET 2015
//
#include "Validation/RecoMET/plugins/METTesterPostProcessor.h"

// Some switches
//
// constructors and destructor
//
METTesterPostProcessor::METTesterPostProcessor(const edm::ParameterSet &iConfig) {}

METTesterPostProcessor::~METTesterPostProcessor() {}

// ------------ method called right after a run ends ------------
void METTesterPostProcessor::dqmEndJob(DQMStore::IBooker &ibook_, DQMStore::IGetter &iget_) {
  std::vector<std::string> subDirVec;
  std::string RunDir = "JetMET/METValidation/";
  iget_.setCurrentFolder(RunDir);
  met_dirs = iget_.getSubdirs();

  // loop over met subdirectories
  for (size_t i=0; i<met_dirs.size(); i++) {
    ibook_.setCurrentFolder(met_dirs[i]);
    mMETResolution_GenMETTrue_METBins =
	  ibook_.book1D("METResolution_GenMETTrue_METBins", "METResolution_GenMETTrue_METBins",
					mNMETBins, mMETBins.data());
    mMETResolution_GenMETTrue_EtaBins =
	  ibook_.book1D("METResolution_GenMETTrue_EtaBins", "METResolution_GenMETTrue_EtaBins",
					mNEtaBins, mEtaBins.data());
    mMETResolution_GenMETTrue_EtaBins =
	  ibook_.book1D("METResolution_GenMETTrue_PhiBins", "METResolution_GenMETTrue_PhiBins",
					mNPhiBins, mPhiBins.data());
    FillMETRes(met_dirs[i], iget_);
  }
}

void METTesterPostProcessor::FillMETRes(std::string metdir, DQMStore::IGetter &iget) {
  for (unsigned metIdx=0; metIdx<mNMETBins-1; ++metIdx) {
	std::string met_folder = "/METResolution_GenMETTrue_MET" + std::to_string((int)mMETBins[metIdx]) + "to" + std::to_string((int)mMETBins[metIdx+1]);
	mMETDifference_GenMETTrue_METBins[metIdx] = iget.get(metdir + met_folder);
  }
  for (unsigned metIdx=0; metIdx<mNEtaBins-1; ++metIdx) {
	std::string met_folder = "/METResolution_GenMETTrue_Eta" + std::to_string((int)mEtaBins[metIdx]) + "to" + std::to_string((int)mEtaBins[metIdx+1]);
	mMETDifference_GenMETTrue_EtaBins[metIdx] = iget.get(metdir + met_folder);
  }
  for (unsigned metIdx=0; metIdx<mNPhiBins-1; ++metIdx) {
	std::string met_folder = "/METResolution_GenMETTrue_Phi" + std::to_string((int)mPhiBins[metIdx]) + "to" + std::to_string((int)mPhiBins[metIdx+1]);
	mMETDifference_GenMETTrue_PhiBins[metIdx] = iget.get(metdir + met_folder);
  }

  // check one object, if existing, then the remaining ME's exist too
  if (mMETDifference_GenMETTrue_METBins[0] && mMETDifference_GenMETTrue_METBins[0]->getRootObject()) {
    // for genmet none of these ME's are filled
	for (unsigned metIdx=0; metIdx<mNMETBins-1; ++metIdx) {
	  mMETResolution_GenMETTrue_METBins->setBinError(metIdx+1, mMETDifference_GenMETTrue_METBins[metIdx]->getMean());
	  mMETResolution_GenMETTrue_METBins->setBinError(metIdx+1, mMETDifference_GenMETTrue_METBins[metIdx]->getRMS());
	}
  }

  if (mMETDifference_GenMETTrue_EtaBins[0] && mMETDifference_GenMETTrue_EtaBins[0]->getRootObject()) {
    // for genmet none of these ME's are filled
	for (unsigned metIdx=0; metIdx<mNEtaBins-1; ++metIdx) {
	  mMETResolution_GenMETTrue_EtaBins->setBinError(metIdx+1, mMETDifference_GenMETTrue_EtaBins[metIdx]->getMean());
	  mMETResolution_GenMETTrue_EtaBins->setBinError(metIdx+1, mMETDifference_GenMETTrue_EtaBins[metIdx]->getRMS());
	}
  }

  if (mMETDifference_GenMETTrue_PhiBins[0] && mMETDifference_GenMETTrue_PhiBins[0]->getRootObject()) {
    // for genmet none of these ME's are filled
	for (unsigned metIdx=0; metIdx<mNPhiBins-1; ++metIdx) {
	  mMETResolution_GenMETTrue_PhiBins->setBinError(metIdx+1, mMETDifference_GenMETTrue_PhiBins[metIdx]->getMean());
	  mMETResolution_GenMETTrue_PhiBins->setBinError(metIdx+1, mMETDifference_GenMETTrue_PhiBins[metIdx]->getRMS());
	}
  }

}
