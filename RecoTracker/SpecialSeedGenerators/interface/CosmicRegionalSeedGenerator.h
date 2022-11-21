#ifndef CosmicRegionalSeedGenerator_h
#define CosmicRegionalSeedGenerator_h

//
// Class:           CosmicRegionalSeedGenerator

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/SpecialSeedGenerators/interface/CosmicTrackingRegion.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

//Geometry
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/PixelGeomDetUnit.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"

class CosmicRegionalSeedGenerator : public TrackingRegionProducer {
public:
  explicit CosmicRegionalSeedGenerator(const edm::ParameterSet& conf, edm::ConsumesCollector&& iC);

  ~CosmicRegionalSeedGenerator() override {}

  std::vector<std::unique_ptr<TrackingRegion> > regions(const edm::Event& event,
                                                        const edm::EventSetup& es) const override;

private:
  float ptMin_;
  float rVertex_;
  float zVertex_;
  float deltaEta_;
  float deltaPhi_;

  std::string regionBase_;

  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;

  edm::InputTag recoMuonsCollection_;
  edm::InputTag recoTrackMuonsCollection_;
  edm::InputTag recoL2MuonsCollection_;
  edm::EDGetTokenT<reco::CaloJetCollection> recoCaloJetsToken_;
  edm::EDGetTokenT<reco::MuonCollection> recoMuonsToken_;
  edm::EDGetTokenT<reco::TrackCollection> recoTrackMuonsToken_;
  edm::EDGetTokenT<reco::RecoChargedCandidateCollection> recoL2MuonsToken_;
  edm::EDGetTokenT<MeasurementTrackerEvent> measurementTrackerEventToken_;

  bool doJetsExclusionCheck_;
  double deltaRExclusionSize_;
  double jetsPtMin_;
  edm::InputTag recoCaloJetsCollection_;
};

#endif
