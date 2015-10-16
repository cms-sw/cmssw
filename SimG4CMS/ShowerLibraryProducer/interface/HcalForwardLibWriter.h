#ifndef SimG4CMS_ShowerLibraryProducer_HcalForwardLibWriter_h
#define SimG4CMS_ShowerLibraryProducer_HcalForwardLibWriter_h

#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TFile.h"
#include "TTree.h"


class HcalForwardLibWriter : public edm::EDAnalyzer {
public:
  struct FileHandle{
    std::string name;
    std::string id;
    int momentum;
  };
  explicit HcalForwardLibWriter(const edm::ParameterSet&);
  ~HcalForwardLibWriter();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  int readUserData();
  int nbins;
  int nshowers;

  TFile* theFile;
  TTree* theTree;
  TFile* LibFile;
  TTree* LibTree;

  edm::Service<TFileService> fs;
  std::string theDataFile;
  std::vector<FileHandle> theFileHandle;

  HFShowerLibraryEventInfo evtInfo;
  HFShowerPhotonCollection emColl;
  HFShowerPhotonCollection hadColl;

};
#endif
