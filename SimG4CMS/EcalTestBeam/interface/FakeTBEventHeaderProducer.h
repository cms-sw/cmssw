#ifndef FakeTBEventHeaderProducer_H
#define FakeTBEventHeaderProducer_H
/*
 * \file FakeTBEventHeaderProducer.h
 *
 *
 * Mimic the event header information
 * for the test beam simulation
 *
 */

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/EcalTestBeam/interface/EcalTBHodoscopeGeometry.h"
#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBEventHeader.h"

class FakeTBEventHeaderProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit FakeTBEventHeaderProducer(const edm::ParameterSet &ps);

  /// Destructor
  ~FakeTBEventHeaderProducer() override;

  /// Produce digis out of raw data
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  edm::EDGetTokenT<PEcalTBInfo> ecalTBInfo_;
};

#endif
