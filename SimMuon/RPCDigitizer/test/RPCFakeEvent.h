#ifndef RPCDigitizer_RPCFakeEvent_h
#define RPCDigitizer_RPCFakeEvent_h

/** \class RPCFakeEvent
 *   Class for the Digitizationn from Event Dump
 *
 *  \author Marcello Maggi -- INFN Bari
 */
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>


class RPCFakeEvent : public edm::EDProducer {
 public:
  RPCFakeEvent(const edm::ParameterSet& config);
  ~RPCFakeEvent(){}
  void produce(edm::Event& e, const edm::EventSetup& c) override;

 private:
  std::vector<std::string> filesed;
  bool rpcdigiprint;

};
#endif
