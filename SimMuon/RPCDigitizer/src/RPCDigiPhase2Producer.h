#ifndef RPCDigiPhase2Producer_h
#define RPCDigiPhase2Producer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimMuon/RPCDigitizer/src/RPCDigitizerPhase2.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CondFormats/DataRecord/interface/RPCClusterSizeRcd.h"

class RPCGeometry;
class RPCSimSetUp;
class RPCSynchronizer;

class RPCDigiPhase2Producer : public edm::stream::EDProducer<> {
public:
  //  typedef edm::DetSetVector<RPCDigiSimLink> RPCDigiSimLinks;
  typedef RPCDigitizerPhase2::RPCDigiSimLinks RPCDigitizerPhase2SimLinks;

  explicit RPCDigiPhase2Producer(const edm::ParameterSet& ps);
  ~RPCDigiPhase2Producer() override;

  void beginRun(const edm::Run&, const edm::EventSetup&) override;

  /**Produces the EDM products,*/
  void produce(edm::Event& e, const edm::EventSetup& c) override;

  void setRPCSetUp(const std::vector<RPCStripNoises::NoiseItem>&, const std::vector<double>&);

private:
  RPCDigitizerPhase2* theRPCDigitizerPhase2;
  RPCSimSetUp* theRPCSimSetUpRPC;

  //Name of Collection used for create the XF
  std::string mix_;
  std::string collection_for_XF;

  //Token for accessing data
  edm::EDGetTokenT<CrossingFrame<PSimHit>> crossingFrameToken;
  const RPCGeometry* _pGeom;

  //EventSetup Tokens
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> geomToken;
  edm::ESGetToken<RPCStripNoises, RPCStripNoisesRcd> noiseToken;
  edm::ESGetToken<RPCClusterSize, RPCClusterSizeRcd> clsToken;
};

#endif
