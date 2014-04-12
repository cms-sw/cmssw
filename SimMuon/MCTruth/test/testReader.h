#ifndef testReader_h
#define testReader_h

#include <memory>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"

class testReader : public edm::EDAnalyzer {
  
 public:
  testReader(const edm::ParameterSet&);
  virtual ~testReader();
  virtual void beginJob() {}  
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:
  edm::InputTag tracksTag;
  edm::InputTag tpTag;
  edm::InputTag assoMapsTag;
};

#endif
