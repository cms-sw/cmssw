#ifndef MCTruth_DTHitAssociator_h
#define MCTruth_DTHitAssociator_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/Handle.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/DTRecHit/interface/DTRecHitCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1D.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "SimDataFormats/DigiSimLinks/interface/DTDigiSimLinkCollection.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"

#include <vector>
#include <map>
#include <string>

typedef std::pair <uint32_t, EncodedEventId> SimHitIdpr;

class DTHitAssociator {

 public:
  DTHitAssociator(const edm::Event&, const edm::EventSetup&, const edm::ParameterSet&); 
  virtual ~DTHitAssociator(){}

  std::vector<SimHitIdpr> associateHitId(const TrackingRecHit & hit);
  std::vector<SimHitIdpr> associateRecHit(const DTRecHit1D * dtrechit);

  typedef std::pair<PSimHit,bool> PSimHit_withFlag;
  typedef std::map<DTWireId, std::vector<PSimHit_withFlag> > SimHitMap;
  SimHitMap mapOfSimHit;

  typedef std::map<DTWireId, std::vector<DTRecHit1DPair> > RecHitMap;
  RecHitMap mapOfRecHit;

  typedef std::map<DTWireId, std::vector<DTDigi> > DigiMap;
  DigiMap mapOfDigi;

  typedef std::map<DTWireId, std::vector<DTDigiSimLink> > LinksMap;
  LinksMap mapOfLinks;

 private:
  bool dumpDT;
  bool crossingframe;
  bool links_exist;
  bool associatorByWire;

  typedef std::vector<std::string> vstring;
  vstring trackerSimHitContainers,muonSimHitContainers;
  
  edm::Handle<DTDigiSimLinkCollection> digisimlinks;
  std::vector<SimHitIdpr> simtrackids; 

  bool SimHitOK(const edm::ESHandle<DTGeometry> &, const PSimHit &);
 
};

#endif



