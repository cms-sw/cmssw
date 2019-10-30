/*! \class   TTClusterAssociator
 *  \brief   Plugin to create the MC truth for TTClusters.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *
 */

#ifndef L1_TRACK_TRIGGER_CLUSTER_ASSOCIATOR_H
#define L1_TRACK_TRIGGER_CLUSTER_ASSOCIATOR_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "L1Trigger/TrackTrigger/interface/classNameFinder.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTClusterAssociationMap.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithmRecord.h"

#include "Geometry/CommonTopologies/interface/Topology.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

#include <memory>
#include <map>
#include <vector>

template <typename T>
class TTClusterAssociator : public edm::stream::EDProducer<> {
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
public:
  /// Constructors
  explicit TTClusterAssociator(const edm::ParameterSet& iConfig);

  /// Destructor
  ~TTClusterAssociator() override;

private:
  /// Data members
  edm::Handle<edm::DetSetVector<PixelDigiSimLink> > thePixelDigiSimLinkHandle;
  edm::Handle<std::vector<TrackingParticle> > TrackingParticleHandle;

  std::vector<edm::InputTag> TTClustersInputTags;

  edm::EDGetTokenT<edm::DetSetVector<PixelDigiSimLink> > digisimLinkToken;
  edm::EDGetTokenT<std::vector<TrackingParticle> > tpToken;
  //std::vector< edm::EDGetTokenT< edm::DetSetVector< TTCluster< T > > > > TTClustersTokens;
  std::vector<edm::EDGetTokenT<edmNew::DetSetVector<TTCluster<T> > > > TTClustersTokens;

  //    const StackedTrackerGeometry                           *theStackedTrackers;
  //unsigned int                                           ADCThreshold;

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
TTClusterAssociator<T>::TTClusterAssociator(const edm::ParameterSet& iConfig) {
  digisimLinkToken =
      consumes<edm::DetSetVector<PixelDigiSimLink> >(iConfig.getParameter<edm::InputTag>("digiSimLinks"));
  tpToken = consumes<std::vector<TrackingParticle> >(iConfig.getParameter<edm::InputTag>("trackingParts"));

  TTClustersInputTags = iConfig.getParameter<std::vector<edm::InputTag> >("TTClusters");

  for (auto iTag = TTClustersInputTags.begin(); iTag != TTClustersInputTags.end(); iTag++) {
    TTClustersTokens.push_back(consumes<edmNew::DetSetVector<TTCluster<T> > >(*iTag));

    produces<TTClusterAssociationMap<T> >((*iTag).instance());
  }
}

/// Destructor
template <typename T>
TTClusterAssociator<T>::~TTClusterAssociator() {}

/// Begin run
template <typename T>
void TTClusterAssociator<T>::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  /// Get the geometry
  iSetup.get<TrackerDigiGeometryRecord>().get(theTrackerGeometry);
  iSetup.get<TrackerTopologyRcd>().get(theTrackerTopology);

  /// Print some information when loaded
  edm::LogInfo("TTClusterAssociator< ") << templateNameFinder<T>() << " > loaded.";
}

/// End run
template <typename T>
void TTClusterAssociator<T>::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

/// Implement the producer
template <>
void TTClusterAssociator<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

#endif
