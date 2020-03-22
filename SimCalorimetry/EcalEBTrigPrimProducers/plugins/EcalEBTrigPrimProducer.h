#ifndef EcalEBTrigPrimProducer_h
#define EcalEBTrigPrimProducer_h

/** \class EcalEBTrigPrimProducer
 *  For Phase II 
 *
 ************************************************************/

#include <memory>

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

class EcalEBTrigPrimTestAlgo;

class EcalEBTrigPrimProducer : public edm::stream::EDProducer<> {
public:
  explicit EcalEBTrigPrimProducer(const edm::ParameterSet& conf);

  ~EcalEBTrigPrimProducer() override;

  void beginRun(const edm::Run& run, const edm::EventSetup& es) override;
  void endRun(const edm::Run&, const edm::EventSetup&) override;
  void produce(edm::Event& e, const edm::EventSetup& c) override;

private:
  std::unique_ptr<EcalEBTrigPrimTestAlgo> algo_;
  bool barrelOnly_;
  bool tcpFormat_;
  bool debug_;
  bool famos_;
  int nSamples_;
  int nEvent_;

  edm::EDGetTokenT<EBDigiCollection> tokenEBdigi_;

  int binOfMaximum_;
  bool fillBinOfMaximumFromHistory_;

  unsigned long long getRecords(edm::EventSetup const& setup);
  unsigned long long cacheID_;
};

#endif
