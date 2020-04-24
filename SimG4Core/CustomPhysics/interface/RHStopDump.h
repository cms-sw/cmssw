#ifndef SimG4Core_RHStopDump_H
#define SimG4Core_RHStopDump_H 
//
// Dump stopping points from the Event into ASCII file
// F.Ratnikov, Apr. 8, 2010
//

#include <fstream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

class RHStopDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
 public:
  explicit RHStopDump(const edm::ParameterSet&);
  ~RHStopDump() override {};
  void analyze(const edm::Event&, const edm::EventSetup&) override;
 private:
  std::ofstream mStream;
  std::string mProducer;
};

#endif
