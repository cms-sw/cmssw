#ifndef SimMuon_GEMDigitizer_GEMPadDigiClusterProducer_h
#define SimMuon_GEMDigitizer_GEMPadDigiClusterProducer_h

/**
 *  \class ME0PadDigiClusterProducer
 *
 *  Produces GEM pad clusters from at most 8 adjacent GEM pads.
 *  Clusters are used downstream in the CSC local trigger to build
 *  GEM-CSC triggers and in the muon trigger to build EMTF tracks
 *
 *  \author Sven Dildick (TAMU)
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiClusterCollection.h"

class GEMGeometry;

class GEMPadDigiClusterProducer : public edm::stream::EDProducer<> {
public:
  explicit GEMPadDigiClusterProducer(const edm::ParameterSet &ps);

  ~GEMPadDigiClusterProducer() override;

  void beginRun(const edm::Run &, const edm::EventSetup &) override;

  void produce(edm::Event &, const edm::EventSetup &) override;

private:
  void buildClusters(const GEMPadDigiCollection &pads, GEMPadDigiClusterCollection &out_clusters);

  /// Name of input digi Collection
  edm::EDGetTokenT<GEMPadDigiCollection> pad_token_;
  edm::InputTag pads_;

  unsigned int maxClusters_;
  unsigned int maxClusterSize_;

  const GEMGeometry *geometry_;
};

#endif
