#ifndef MCTruth_GEMHitAssociator_h
#define MCTruth_GEMHitAssociator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <vector>
#include <map>
#include <string>
#include <set>

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHitCollection.h"
#include "DataFormats/GEMRecHit/interface/GEMSegmentCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class GEMHitAssociator {

 public:

   typedef edm::DetSetVector<StripDigiSimLink> DigiSimLinks;
   typedef edm::DetSet<StripDigiSimLink> LayerLinks;
   typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;

   // Constructor with configurable parameters
   GEMHitAssociator(const edm::ParameterSet&, edm::ConsumesCollector && ic);
   GEMHitAssociator(const edm::Event& e, const edm::EventSetup& eventSetup, const edm::ParameterSet& conf);
    
   void initEvent(const edm::Event&, const edm::EventSetup&);

   // Destructor
   ~GEMHitAssociator(){}

   std::vector<SimHitIdpr> associateRecHit(const GEMRecHit * gemrechit) const;

 private:
    
   const DigiSimLinks * theDigiSimLinks;
   edm::InputTag GEMdigisimlinkTag;
 
   bool crossingframe;
   bool useGEMs_;
   edm::InputTag GEMsimhitsTag;
   edm::InputTag GEMsimhitsXFTag;
    
   edm::EDGetTokenT<CrossingFrame<PSimHit> > GEMsimhitsXFToken_;
   edm::EDGetTokenT<edm::PSimHitContainer> GEMsimhitsToken_;
   edm::EDGetTokenT<edm::DetSetVector<StripDigiSimLink> > GEMdigisimlinkToken_;

   std::map<unsigned int, edm::PSimHitContainer> _SimHitMap;

 };

#endif

