#ifndef FakeTBHodoscopeRawInfoProducer_H
#define FakeTBHodoscopeRawInfoProducer_H
/*
 * \file FakeTBHodoscopeRawInfoProducer.h
 *
 *
 * Mimic the hodoscope raw information using
 * the generated vertex of the test beam simulation
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
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopePlaneRawHits.h"
#include "TBDataFormats/EcalTBObjects/interface/EcalTBHodoscopeRawInfo.h"

class FakeTBHodoscopeRawInfoProducer : public edm::stream::EDProducer<> {
public:
  /// Constructor
  explicit FakeTBHodoscopeRawInfoProducer(const edm::ParameterSet &ps);

  /// Destructor
  ~FakeTBHodoscopeRawInfoProducer() override;

  /// Produce digis out of raw data
  void produce(edm::Event &event, const edm::EventSetup &eventSetup) override;

private:
  EcalTBHodoscopeGeometry *theTBHodoGeom_;

  edm::EDGetTokenT<PEcalTBInfo> ecalTBInfo_;
};

#endif
