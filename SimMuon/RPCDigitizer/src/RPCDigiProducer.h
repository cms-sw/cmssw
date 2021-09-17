#ifndef RPCDigiProducer_h
#define RPCDigiProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CondFormats/DataRecord/interface/RPCClusterSizeRcd.h"

class RPCGeometry;
class RPCSimSetUp;
class RPCSynchronizer;

class RPCDigiProducer : public edm::stream::EDProducer<> {
public:
  typedef RPCDigitizer::RPCDigiSimLinks RPCDigitizerSimLinks;

  explicit RPCDigiProducer(const edm::ParameterSet& ps);
  ~RPCDigiProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /**Produces the EDM products,*/
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  void setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>&, const std::vector<double>&);

private:
  RPCDigitizer* theDigitizer;
  RPCSimSetUp* theRPCSimSetUp;

  //Name of Collection used for create the XF
  std::string mix_;
  std::string collection_for_XF;

  //Token for accessing data
  edm::EDGetTokenT<CrossingFrame<PSimHit>> crossingFrameToken;

  //EventSetup Tokens
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> geomToken;
  edm::ESGetToken<RPCStripNoises, RPCStripNoisesRcd> noiseToken;
  edm::ESGetToken<RPCClusterSize, RPCClusterSizeRcd> clsToken;
};

#endif
