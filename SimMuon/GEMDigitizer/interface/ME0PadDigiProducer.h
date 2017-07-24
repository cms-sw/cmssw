#ifndef SimMuon_GEMDigitizer_ME0PadDigiProducer_h
#define SimMuon_GEMDigitizer_ME0PadDigiProducer_h

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"

class ME0Geometry;

/// \class ME0PadDigiProducer

class ME0PadDigiProducer : public edm::stream::EDProducer<>
{
public:

  explicit ME0PadDigiProducer(const edm::ParameterSet& ps);

  virtual ~ME0PadDigiProducer();

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:

  void buildPads(const ME0DigiCollection &digis, ME0PadDigiCollection &out_pads) const;

  /// Name of input digi Collection
  edm::EDGetTokenT<ME0DigiCollection> digi_token_;
  edm::InputTag digis_;

  const ME0Geometry * geometry_;
};

#endif

