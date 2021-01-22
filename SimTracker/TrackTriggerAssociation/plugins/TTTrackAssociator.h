/*! \class   TTTrackAssociator
 *  \brief   Plugin to create the MC truth for TTTracks.
 *  \details After moving from SimDataFormats to DataFormats,
 *           the template structure of the class was maintained
 *           in order to accomodate any types other than PixelDigis
 *           in case there is such a need in the future.
 *
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 19
 *  (tidy up: Ian Tomalin, 2020)
 *
 */

#ifndef L1_TRACK_TRIGGER_TRACK_ASSOCIATOR_H
#define L1_TRACK_TRIGGER_TRACK_ASSOCIATOR_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"

#include "L1Trigger/TrackTrigger/interface/classNameFinder.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTStubAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/interface/TTTrackAssociationMap.h"
#include "SimTracker/TrackTriggerAssociation/plugins/TTStubAssociator.h"
#include "SimTracker/TrackTriggerAssociation/plugins/TTClusterAssociator.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include <memory>
#include <map>
#include <vector>

template <typename T>
class TTTrackAssociator : public edm::stream::EDProducer<> {
  /// NOTE since pattern hit correlation must be performed within a stacked module, one must store
  /// Clusters in a proper way, providing easy access to them in a detector/member-wise way
public:
  /// Constructors
  explicit TTTrackAssociator(const edm::ParameterSet& iConfig);

  /// Destructor
  ~TTTrackAssociator() override;

private:
  /// Data members
  std::vector<edm::InputTag> ttTracksInputTags_;

  std::vector<edm::EDGetTokenT<std::vector<TTTrack<T> > > > ttTracksTokens_;
  edm::EDGetTokenT<TTStubAssociationMap<T> > ttStubTruthToken_;
  edm::EDGetTokenT<TTClusterAssociationMap<T> > ttClusterTruthToken_;

  bool TTTrackAllowOneFalse2SStub;

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
TTTrackAssociator<T>::TTTrackAssociator(const edm::ParameterSet& iConfig) {
  ttTracksInputTags_ = iConfig.getParameter<std::vector<edm::InputTag> >("TTTracks");
  ttClusterTruthToken_ = consumes<TTClusterAssociationMap<T> >(iConfig.getParameter<edm::InputTag>("TTClusterTruth"));
  ttStubTruthToken_ = consumes<TTStubAssociationMap<T> >(iConfig.getParameter<edm::InputTag>("TTStubTruth"));
  TTTrackAllowOneFalse2SStub = iConfig.getParameter<bool>("TTTrackAllowOneFalse2SStub");
  if (TTTrackAllowOneFalse2SStub) {
    edm::LogInfo("TTTrackAssociator< ") << "Allow track if no more than one 2S stub doesn't match truth.";
  } else {
    edm::LogInfo("TTTrackAssociator< ") << "All 2S stubs must match truth.";
  }

  for (const auto& iTag : ttTracksInputTags_) {
    ttTracksTokens_.push_back(consumes<std::vector<TTTrack<T> > >(iTag));

    produces<TTTrackAssociationMap<T> >(iTag.instance());
  }
}

/// Destructor
template <typename T>
TTTrackAssociator<T>::~TTTrackAssociator() {}

/// Begin run
template <typename T>
void TTTrackAssociator<T>::beginRun(const edm::Run& run, const edm::EventSetup& iSetup) {
  /// Print some information when loaded
  edm::LogInfo("TTStubAssociator< ") << "TTTrackAssociator< " << templateNameFinder<T>() << " > loaded.";
}

/// End run
template <typename T>
void TTTrackAssociator<T>::endRun(const edm::Run& run, const edm::EventSetup& iSetup) {}

/// Implement the producer
template <>
void TTTrackAssociator<Ref_Phase2TrackerDigi_>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

#endif
