#ifndef RPCDigitizer_RPCFakeEvent_h
#define RPCDigitizer_RPCFakeEvent_h

/** \class RPCFakeEvent
 *   Class for the Digitizationn from Event Dump
 *
 *  \author Marcello Maggi -- INFN Bari
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

class RPCFakeEvent : public edm::one::EDProducer<> {
public:
  RPCFakeEvent(const edm::ParameterSet& config);
  ~RPCFakeEvent() override {}
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  const std::vector<std::string> filesed;
  const bool rpcdigiprint;
  const edm::ESGetToken<RPCGeometry, MuonGeometryRecord> tokGeom_;
};
#endif
