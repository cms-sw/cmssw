#ifndef RPCandIRPCDigiProducer_h
#define RPCandIRPCDigiProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizer.h"
#include "SimMuon/RPCDigitizer/src/IRPCDigitizer.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CondFormats/DataRecord/interface/RPCClusterSizeRcd.h"

class RPCGeometry;
class RPCSimSetUp;
class RPCSynchronizer;

class RPCandIRPCDigiProducer : public edm::stream::EDProducer<> {
public:
  //  typedef edm::DetSetVector<RPCDigiSimLink> RPCDigiSimLinks;
  typedef RPCDigitizer::RPCDigiSimLinks RPCDigitizerSimLinks;

  explicit RPCandIRPCDigiProducer(const edm::ParameterSet& ps);
  ~RPCandIRPCDigiProducer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /**Produces the EDM products,*/
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  void setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>&, const std::vector<double>&);

private:
  RPCDigitizer* theRPCDigitizer;
  IRPCDigitizer* theIRPCDigitizer;
  RPCSimSetUp* theRPCSimSetUpRPC;
  RPCSimSetUp* theRPCSimSetUpIRPC;
  //  RPCSimSetUp* theRPCSimSetUp;

  //Name of Collection used for create the XF
  std::string mix_;
  std::string collection_for_XF;

  //Token for accessing data
  edm::EDGetTokenT<CrossingFrame<PSimHit>> crossingFrameToken;
  const RPCGeometry* _pGeom;
};

#endif
