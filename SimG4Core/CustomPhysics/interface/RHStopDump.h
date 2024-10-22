#ifndef SimG4Core_CustomPhysics_RHStopDump_H
#define SimG4Core_CustomPhysics_RHStopDump_H
//
// Dump stopping points from the Event into ASCII file
// F.Ratnikov, Apr. 8, 2010
//

#include <fstream>

#include "FWCore/Framework/interface/one/EDAnalyzer.h"

class RHStopDump : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit RHStopDump(const edm::ParameterSet&);
  ~RHStopDump() override = default;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  std::ofstream mStream;
  const std::string mProducer;
  const edm::EDGetTokenT<std::vector<std::string> > tokNames_;
  const edm::EDGetTokenT<std::vector<float> > tokenXs_;
  const edm::EDGetTokenT<std::vector<float> > tokenYs_;
  const edm::EDGetTokenT<std::vector<float> > tokenZs_;
  const edm::EDGetTokenT<std::vector<float> > tokenTs_;
  const edm::EDGetTokenT<std::vector<int> > tokenIds_;
  const edm::EDGetTokenT<std::vector<float> > tokenMasses_;
  const edm::EDGetTokenT<std::vector<float> > tokenCharges_;
};

#endif
