/*! \class   TTStubAssociator
 *  \brief   Plugin to create the MC truth for TTStubs.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_STUB_ASSOCIATOR_H
#define L1_TRACK_TRIGGER_STUB_ASSOCIATOR_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTStub.h"

#include "L1Trigger/TrackTrigger/interface/classNameFinder.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/plugins/TTClusterAssociator.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <memory>
#include <map>
#include <vector>

template <typename T>
class TTStubAssociator : public edm::stream::EDProducer<> {
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
public:
  /// Constructors
  explicit TTStubAssociator(const edm::ParameterSet& iConfig);

  /// Destructor
  ~TTStubAssociator() override;

private:
  /// Data members
  std::vector<edm::InputTag> TTStubsInputTags;
  std::vector<edm::InputTag> TTClusterTruthInputTags;

  std::vector<edm::EDGetTokenT<edmNew::DetSetVector<TTStub<T> > > > TTStubsTokens;
  std::vector<edm::EDGetTokenT<TTClusterAssociationMap<T> > > TTClusterTruthTokens;

  edm::ESHandle<TrackerGeometry> theTrackerGeometry;
  edm::ESHandle<TrackerTopology> theTrackerTopology;

  /// Mandatory methods
  void beginRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void endRun(const edm::Run& run, const edm::EventSetup& iSetup) override;
  void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

};  /// Close class

/*! \brief   Implementation of methods
 *  \details Here, in the header file, the methods which do not depend
 *           on the specific type <T> that can fit the template.
 *           Other methods, with type-specific features, are implemented
 *           in the source file.
 */

/// Constructors
template <typename T>
TTStubAssociator<T>::TTStubAssociator(const edm::ParameterSet& iConfig) {
  TTStubsInputTags = iConfig.getParameter<std::vector<edm::InputTag> >("TTStubs");
  TTClusterTruthInputTags = iConfig.getParameter<std::vector<edm::InputTag> >("TTClusterTruth");

  for (auto iTag = TTClusterTruthInputTags.begin(); iTag != TTClusterTruthInputTags.end(); iTag++) {
    TTClusterTruthTokens.push_back(consumes<TTClusterAssociationMap<T> >(*iTag));
  }

  for (auto iTag = TTStubsInputTags.begin(); iTag != TTStubsInputTags.end(); iTag++) {
    TTStubsTokens.push_back(consumes<edmNew::DetSetVector<TTStub<T> > >(*iTag));

    produces<TTStubAssociationMap<T> >((*iTag).instance());
  }
}

/// Destructor
template <typename T>
TTStubAssociator<T>::~TTStubAssociator() {}

/// Begin run
template <typename T>
void TTStubAssociator<T>::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  /// Print some information when loaded
  edm::LogInfo("TTStubAssociator< ") << templateNameFinder<T>() << " > loaded.";

  iSetup.get<TrackerTopologyRcd>().get(theTrackerTopology);
  iSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
}

/// End run
template <typename T>
void TTStubAssociator<T>::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

/// Implement the producer
template <>
void TTStubAssociator<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

#endif
