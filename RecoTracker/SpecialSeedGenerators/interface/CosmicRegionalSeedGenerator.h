#ifndef CosmicRegionalSeedGenerator_h
#define CosmicRegionalSeedGenerator_h

//
// Class:           CosmicRegionalSeedGenerator

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDProducer.h"
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
#include "../interface/CosmicTrackingRegion.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/GeomPropagators/interface/StateOnTrackerBound.h"

// Math
#include "Math/GenVector/VectorUtil.h"
#include "Math/GenVector/PxPyPzE4D.h"

//Geometry
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"


class CosmicRegionalSeedGenerator : public TrackingRegionProducer { 
 

 public:
  explicit CosmicRegionalSeedGenerator(const edm::ParameterSet& conf);

  virtual ~CosmicRegionalSeedGenerator() {};
  
  virtual std::vector<TrackingRegion* > regions(const edm::Event& event, const edm::EventSetup& es) const;

 private:
  edm::ParameterSet conf_;
  edm::ParameterSet regionPSet;

  float ptMin_;
  float rVertex_;
  float zVertex_;
  float deltaEta_;
  float deltaPhi_;

  std::string thePropagatorName_;
  std::string regionBase_;

  edm::InputTag recoMuonsCollection_;
  edm::InputTag recoTrackMuonsCollection_;
  edm::InputTag recoL2MuonsCollection_;
  
  bool   doJetsExclusionCheck_;
  double deltaRExclusionSize_;
  double jetsPtMin_;
  edm::InputTag recoCaloJetsCollection_;

};

#endif
