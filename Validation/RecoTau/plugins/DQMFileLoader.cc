#include "Validation/RecoTau/plugins/DQMFileLoader.h"

#include "Validation/RecoTau/plugins/dqmAuxFunctions.h"

// framework & common header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <TFile.h>
#include <TList.h>
#include <TKey.h>
#include <TH1.h>

#include <iostream>

const std::string dqmRootDirectory_inTFile = "DQMData";

const double defaultScaleFactor = 1.;

const int verbosity = 0;

void mapSubDirectoryStructure(TDirectory* directory, std::string directoryName, std::set<std::string>& subDirectories) {
  //std::cout << "<mapSubDirectoryStructure>:" << std::endl;
  //std::cout << " directoryName = " << directoryName << std::endl;

  TList* subDirectoryNames = directory->GetListOfKeys();
  if (!subDirectoryNames)
    return;

  TIter next(subDirectoryNames);
  while (TKey* key = dynamic_cast<TKey*>(next())) {
    //std::cout << " key->GetName = " << key->GetName() << std::endl;
    TObject* obj = directory->Get(key->GetName());
    //std::cout << " obj = " << obj << std::endl;
    if (TDirectory* subDirectory = dynamic_cast<TDirectory*>(obj)) {
      std::string subDirectoryName = dqmDirectoryName(directoryName).append(key->GetName());
      //std::cout << " subDirectoryName = " << subDirectoryName << std::endl;

      subDirectories.insert(subDirectoryName);

      mapSubDirectoryStructure(subDirectory, subDirectoryName, subDirectories);
    }
  }
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMFileLoader::cfgEntryFileSet::cfgEntryFileSet(const std::string& name, const edm::ParameterSet& cfg) {
  //std::cout << "<TauDQMFileLoader::cfgEntryFileSet>" << std::endl;

  name_ = name;

  vstring inputFileList = cfg.getParameter<vstring>("inputFileNames");
  for (vstring::const_iterator inputFile = inputFileList.begin(); inputFile != inputFileList.end(); ++inputFile) {
    if (inputFile->find(rangeKeyword) != std::string::npos) {
      size_t posRangeStart = inputFile->find(rangeKeyword) + rangeKeyword.length();
      size_t posRangeEnd = inputFile->find('#', posRangeStart);

      size_t posRangeSeparator = inputFile->find('-', posRangeStart);

      if ((posRangeEnd == std::string::npos) || (posRangeSeparator >= posRangeEnd)) {
        edm::LogError("TauDQMFileLoader::cfgEntryFileSet")
            << " Invalid range specification in inputFile = " << (*inputFile) << " !!";
        continue;
      }

      std::string firstFile = std::string(*inputFile, posRangeStart, posRangeSeparator - posRangeStart);
      //std::cout << "firstFile = " << firstFile << std::endl;
      std::string lastFile = std::string(*inputFile, posRangeSeparator + 1, posRangeEnd - (posRangeSeparator + 1));
      //std::cout << "lastFile = " << lastFile << std::endl;

      if (firstFile.length() != lastFile.length()) {
        edm::LogError("TauDQMFileLoader::cfgEntryFileSet")
            << " Invalid range specification in inputFile = " << (*inputFile) << " !!";
        continue;
      }

      int numFirstFile = atoi(firstFile.data());
      int numLastFile = atoi(lastFile.data());
      for (int iFile = numFirstFile; iFile <= numLastFile; ++iFile) {
        std::ostringstream fileName;
        fileName << std::string(*inputFile, 0, inputFile->find(rangeKeyword));
        fileName << std::setfill('0') << std::setw(firstFile.length()) << iFile;
        fileName << std::string(*inputFile, posRangeEnd + 1);
        //std::cout << "iFile = " << iFile << ", fileName = " << fileName.str() << std::endl;
        inputFileNames_.push_back(fileName.str());
      }
    } else {
      inputFileNames_.push_back(*inputFile);
    }
  }

  scaleFactor_ = (cfg.exists("scaleFactor")) ? cfg.getParameter<double>("scaleFactor") : defaultScaleFactor;

  //dqmDirectory_store_ = ( cfg.exists("dqmDirectory_store") ) ? cfg.getParameter<std::string>("dqmDirectory_store") : name_;
  dqmDirectory_store_ = (cfg.exists("dqmDirectory_store")) ? cfg.getParameter<std::string>("dqmDirectory_store") : "";

  if (verbosity)
    print();
}

void TauDQMFileLoader::cfgEntryFileSet::print() const {
  std::cout << "<cfgEntryFileSet::print>:" << std::endl;
  std::cout << " name = " << name_ << std::endl;
  std::cout << " inputFileNames = " << format_vstring(inputFileNames_) << std::endl;
  std::cout << " scaleFactor = " << scaleFactor_ << std::endl;
  std::cout << " dqmDirectory_store = " << dqmDirectory_store_ << std::endl;
}

//
//-----------------------------------------------------------------------------------------------------------------------
//

TauDQMFileLoader::TauDQMFileLoader(const edm::ParameterSet& cfg) {
  std::cout << "<TauDQMFileLoader::TauDQMFileLoader>:" << std::endl;

  cfgError_ = 0;

  //--- configure fileSets
  //std::cout << "--> configuring fileSets..." << std::endl;
  readCfgParameter<cfgEntryFileSet>(cfg, fileSets_);

  //--- check that dqmDirectory_store configuration parameters are specified for all fileSets,
  //    unless there is only one fileSet to be loaded
  //    (otherwise histograms of different fileSets get overwritten,
  //     once histograms of the next fileSet are loaded)
  for (std::map<std::string, cfgEntryFileSet>::const_iterator fileSet = fileSets_.begin(); fileSet != fileSets_.end();
       ++fileSet) {
    if (fileSet->second.dqmDirectory_store_.empty() && fileSets_.size() > 1) {
      edm::LogError("TauDQMFileLoader") << " dqmDirectory_store undefined for fileSet = " << fileSet->second.name_
                                        << " !!";
      cfgError_ = 1;
      break;
    }
  }

  std::cout << "done." << std::endl;
}

TauDQMFileLoader::~TauDQMFileLoader() {
  // nothing to be done yet...
}

void TauDQMFileLoader::analyze(const edm::Event&, const edm::EventSetup&) {
  // nothing to be done yet...
}

void TauDQMFileLoader::endRun(const edm::Run& r, const edm::EventSetup& c) {
  std::cout << "<TauDQMFileLoader::endJob>:" << std::endl;

  //--- check that configuration parameters contain no errors
  if (cfgError_) {
    edm::LogError("endJob") << " Error in Configuration ParameterSet"
                            << " --> histograms will NOT be loaded !!";
    return;
  }

  //--- check that DQMStore service is available
  if (!edm::Service<DQMStore>().isAvailable()) {
    edm::LogError("endJob") << " Failed to access dqmStore"
                            << " --> histograms will NOT be loaded !!";
    return;
  }

  //--- stop ROOT from keeping references to all histograms
  //TH1::AddDirectory(false);

  //--- check that inputFiles exist;
  //    store list of directories existing in inputFile,
  //    in order to separate histogram directories existing in the inputFile from directories existing in DQMStore
  //    when calling recursive function dqmCopyRecursively
  for (std::map<std::string, cfgEntryFileSet>::const_iterator fileSet = fileSets_.begin(); fileSet != fileSets_.end();
       ++fileSet) {
    for (vstring::const_iterator inputFileName = fileSet->second.inputFileNames_.begin();
         inputFileName != fileSet->second.inputFileNames_.end();
         ++inputFileName) {
      //std::cout << " checking inputFile = " << (*inputFileName) << std::endl;
      TFile inputFile(inputFileName->data());
      if (inputFile.IsZombie()) {
        edm::LogError("endJob") << " Failed to open inputFile = " << (*inputFileName)
                                << "--> histograms will NOT be loaded !!";
        return;
      }

      TObject* obj = inputFile.Get(dqmRootDirectory_inTFile.data());
      //std::cout << " obj = " << obj << std::endl;
      if (TDirectory* directory = dynamic_cast<TDirectory*>(obj)) {
        mapSubDirectoryStructure(directory, dqmRootDirectory, subDirectoryMap_[*inputFileName]);
      } else {
        edm::LogError("endJob") << " Failed to access " << dqmRootDirectory_inTFile
                                << " in inputFile = " << (*inputFileName) << "--> histograms will NOT be loaded !!";
        return;
      }

      inputFile.Close();
    }
  }

  //for ( std::map<std::string, sstring>::const_iterator inputFile = subDirectoryMap_.begin();
  //  	  inputFile != subDirectoryMap_.end(); ++inputFile ) {
  //  std::cout << "inputFile = " << inputFile->first << ":" << std::endl;
  //  for ( sstring::const_iterator directory = inputFile->second.begin();
  //	    directory != inputFile->second.end(); ++directory ) {
  //    std::cout << " " << (*directory) << std::endl;
  //  }
  //}

  //--- load histograms from file
  //std::cout << "--> loading histograms from file..." << std::endl;
  DQMStore& dqmStore = (*edm::Service<DQMStore>());
  for (std::map<std::string, cfgEntryFileSet>::const_iterator fileSet = fileSets_.begin(); fileSet != fileSets_.end();
       ++fileSet) {
    for (vstring::const_iterator inputFileName = fileSet->second.inputFileNames_.begin();
         inputFileName != fileSet->second.inputFileNames_.end();
         ++inputFileName) {
      if (verbosity)
        std::cout << " opening inputFile = " << (*inputFileName) << std::endl;
      dqmStore.open(*inputFileName, true);

      //--- if dqmDirectory_store specified in configuration parameters,
      //    move histograms from dqmRootDirectory to dqmDirectory_store
      //    (if the histograms are not moved, the histograms get overwritten,
      //     the next time DQMStore::open is called)
      if (!fileSet->second.dqmDirectory_store_.empty()) {
        std::string inputDirectory = dqmRootDirectory;
        //std::cout << "inputDirectory = " << inputDirectory << std::endl;
        std::string outputDirectory =
            dqmDirectoryName(std::string(inputDirectory)).append(fileSet->second.dqmDirectory_store_);
        //std::cout << "outputDirectory = " << outputDirectory << std::endl;

        dqmStore.setCurrentFolder(inputDirectory);
        std::vector<std::string> dirNames = dqmStore.getSubdirs();
        for (std::vector<std::string>::const_iterator dirName = dirNames.begin(); dirName != dirNames.end();
             ++dirName) {
          std::string subDirName = dqmSubDirectoryName_merged(inputDirectory, *dirName);
          //std::cout << " subDirName = " << subDirName << std::endl;

          const sstring& subDirectories = subDirectoryMap_[*inputFileName];
          if (subDirectories.find(subDirName) != subDirectories.end()) {
            std::string inputDirName_full = dqmDirectoryName(inputDirectory).append(subDirName);
            //std::cout << " inputDirName_full = " << inputDirName_full << std::endl;

            std::string outputDirName_full = dqmDirectoryName(outputDirectory).append(subDirName);
            //std::cout << " outputDirName_full = " << outputDirName_full << std::endl;

            //--- load histograms contained in inputFile into inputDirectory;
            //    when processing first inputFile, check that histograms in outputDirectory do not yet exist;
            //    add histograms in inputFile to those in outputDirectory afterwards;
            //    clear inputDirectory once finished processing all inputFiles.
            int mode = (inputFileName == fileSet->second.inputFileNames_.begin()) ? 1 : 3;
            dqmCopyRecursively(
                dqmStore, inputDirName_full, outputDirName_full, fileSet->second.scaleFactor_, mode, true);
          }
        }
      }
    }
  }

  std::cout << "done." << std::endl;
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TauDQMFileLoader);
