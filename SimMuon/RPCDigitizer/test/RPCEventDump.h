#ifndef RPCDigitizer_RPCEventDump_h
#define RPCDigitizer_RPCEventDump_h

/** \class RPCEventDump
 *   Class for the Digitizationn from Event Dump
 *
 *  \author Marcello Maggi -- INFN Bari
 */
#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>


class RPCEventDump : public edm::EDProducer {
 public:
  RPCEventDump(const edm::ParameterSet& config);
  ~RPCEventDump(){}
  void produce(edm::Event& e, const edm::EventSetup& c) override;

 private:
  std::vector<std::string> filesed;
  bool rpcdigiprint;

};
#endif
