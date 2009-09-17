#ifndef DIGI2RAW2DIGI_H
#define DIGI2RAW2DIGI_H

// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <map>

class Digi2Raw2Digi : public edm::EDAnalyzer {
public:
  explicit Digi2Raw2Digi(const edm::ParameterSet&);
  ~Digi2Raw2Digi();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  template<class Digi>  void compare(const edm::Event&, const edm::EventSetup&);  virtual void beginJob() ;
  virtual void endJob() ;

 private:

  edm::InputTag inputTag1_;
  edm::InputTag inputTag2_;

  std::string outputFile_;
  DQMStore* dbe_;

  MonitorElement* meStatus;

  int unsuppressed; // flag for ZSC unsuppressedDigis picking up

};

#endif
