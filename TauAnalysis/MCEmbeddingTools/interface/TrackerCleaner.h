/** \class TrackerCleaner
 *
 *
 * \author Stefan Wayand;
 *         Christian Veelken, LLR
 *
 *
 *
 *
 *
 */

#ifndef TauAnalysis_MCEmbeddingTools_TrackerCleaner_H
#define TauAnalysis_MCEmbeddingTools_TrackerCleaner_H

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"
#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/BaseTrackerRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"

#include <iostream>
#include <map>
#include <string>

template <typename T>
class TrackerCleaner : public edm::stream::EDProducer<> {
public:
  explicit TrackerCleaner(const edm::ParameterSet &);
  ~TrackerCleaner() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;

  const edm::EDGetTokenT<edm::View<pat::Muon>> mu_input_;
  typedef edmNew::DetSetVector<T> TrackClusterCollection;

  std::map<std::string, edm::EDGetTokenT<TrackClusterCollection>> inputs_;

  bool match_rechit_type(const TrackingRecHit &murechit);
};

#endif
