#ifndef MCTruth_RPCHitAssociator_h
#define MCTruth_RPCHitAssociator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"

#include <vector>
#include <map>
#include <string>
#include <set>

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/MuonDetId/interface/MuonSubdetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "DataFormats/Common/interface/DetSetVector.h"

class RPCHitAssociator {

 public:
   typedef edm::DetSetVector<RPCDigiSimLink> RPCDigiSimLinks;
   typedef std::pair<uint32_t, EncodedEventId> SimHitIdpr;

  // Constructor with configurable parameters
  RPCHitAssociator(const edm::Event&, const edm::EventSetup&, const edm::ParameterSet&); 

  // Destructor
  ~RPCHitAssociator(){}

   std::vector<SimHitIdpr> associateRecHit(const TrackingRecHit & hit);
   std::set<RPCDigiSimLink> findRPCDigiSimLink(uint32_t rpcDetId, int strip, int bx);
   //   const PSimHit* linkToSimHit(RPCDigiSimLink link);


 private:
   edm::Handle< edm::DetSetVector<RPCDigiSimLink> > _thelinkDigis;
   edm::InputTag RPCdigisimlinkTag;
 
   bool crossingframe;
   edm::InputTag RPCsimhitsTag;
   edm::InputTag RPCsimhitsXFTag;

   std::map<unsigned int, edm::PSimHitContainer> _SimHitMap;

 };

#endif



