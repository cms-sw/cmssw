#ifndef SimTrackSimVertexDumper_H
#define SimTrackSimVertexDumper_H
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include <vector>

class SimTrackSimVertexDumper : public edm::EDAnalyzer {
public:
  explicit SimTrackSimVertexDumper(const edm::ParameterSet&);
  ~SimTrackSimVertexDumper() override{};

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void beginJob() override{};
  void endJob() override{};

private:
  edm::InputTag HepMCLabel;
  edm::InputTag SimTkLabel;
  edm::InputTag SimVtxLabel;
  bool dumpHepMC;
};

#endif
