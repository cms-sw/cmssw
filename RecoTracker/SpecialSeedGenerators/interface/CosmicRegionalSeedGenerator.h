#ifndef CosmicRegionalSeedGenerator_h
#define CosmicRegionalSeedGenerator_h

//
// Class:           CosmicRegionalSeedGenerator

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionProducer.h"
#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "../interface/CosmicTrackingRegion.h"

#include "DQM/HLTEvF/interface/FourVectorHLT.h"
 
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "FWCore/Framework/interface/TriggerNames.h"
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
#include "DataFormats/SiStripDetId/interface/TOBDetId.h"
#include "DataFormats/SiStripDetId/interface/TECDetId.h"


class CosmicRegionalSeedGenerator : public TrackingRegionProducer { 
 

 public:
  explicit CosmicRegionalSeedGenerator(const edm::ParameterSet& conf);

  virtual ~CosmicRegionalSeedGenerator() {};
  
  virtual std::vector<TrackingRegion* > regions(const edm::Event& event, const edm::EventSetup& es) const;

 private:
  edm::ParameterSet conf_;
  float m_ptMin;
  float m_rVertex;
  float m_zVertex;
  float m_deltaEta;
  float m_deltaPhi;
  edm::InputTag m_tp_label;
  edm::InputTag hltTag_;
  std::string triggerSummaryLabel_;
  std::string thePropagatorName_;
  std::string seeding_;

};

#endif
