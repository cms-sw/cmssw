#ifndef SimG4CMS_ShowerLibraryProducer_HcalForwardLibWriter_h
#define SimG4CMS_ShowerLibraryProducer_HcalForwardLibWriter_h

#include <memory>
#include <string>
#include <fstream>
#include <utility>
#include <vector>


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"

class HcalForwardLibWriter : public edm::EDProducer {

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
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  //void fillEvent(HFShowerPhotonCollection& em, HFShowerPhotonCollection& had);
  int readUserData();

  std::string theDataFile;
  std::vector<FileHandle> theFileHandle;

};
#endif
