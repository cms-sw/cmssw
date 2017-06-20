#ifndef SimMuon_GEMDigitizer_ME0PadDigiClusterProducer_h
#define SimMuon_GEMDigitizer_ME0PadDigiClusterProducer_h

/** 
 *  \class ME0PadDigiClusterProducer
 *
 *  Produces ME0 pad clusters from at most 8 adjacent ME0 pads.
 *  Clusters are used downstream to build triggers. 
 *  
 *  \author Sven Dildick (TAMU)
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/GEMDigi/interface/ME0PadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiClusterCollection.h"

class ME0Geometry;

class ME0PadDigiClusterProducer : public edm::stream::EDProducer<>
{
public:

  explicit ME0PadDigiClusterProducer(const edm::ParameterSet& ps);

  virtual ~ME0PadDigiClusterProducer();

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;

  virtual void produce(edm::Event&, const edm::EventSetup&) override;

private:
  
  void buildClusters(const ME0PadDigiCollection &pads, ME0PadDigiClusterCollection &out_clusters);

  /// Name of input digi Collection
  edm::EDGetTokenT<ME0PadDigiCollection> pad_token_;
  edm::InputTag pads_;

  unsigned int maxClusters_;
  unsigned int maxClusterSize_;

  const ME0Geometry * geometry_;
};

#endif

