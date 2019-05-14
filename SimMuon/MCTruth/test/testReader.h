#ifndef testReader_h
#define testReader_h

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include <memory>

class testReader : public edm::EDAnalyzer {
public:
  testReader(const edm::ParameterSet &);
  ~testReader() override;
  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;

private:
  edm::InputTag tracksTag;
  edm::InputTag tpTag;
  edm::InputTag assoMapsTag;
};

#endif
