#include "TauAnalysis/MCEmbeddingTools/plugins/MuonCaloDistanceProducer.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "TrackingTools/TrackAssociator/interface/TrackDetectorAssociator.h"
#include "TrackingTools/TrackAssociator/interface/TrackAssociatorParameters.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"

#include <DataFormats/MuonReco/interface/Muon.h>
#include <DataFormats/MuonReco/interface/MuonFwd.h>
#include <DataFormats/Candidate/interface/ShallowCloneCandidate.h>

#include "TauAnalysis/MCEmbeddingTools/interface/embeddingAuxFunctions.h"

#include <string>
#include <vector>

MuonCaloDistanceProducer::MuonCaloDistanceProducer(const edm::ParameterSet& cfg)
  : srcSelectedMuons_(cfg.getParameter<edm::InputTag>("selectedMuons"))
{
  // maps of detId to distance traversed by muon through detector volume
  produces<reco::CandidateCollection>("muons");
  produces<detIdToFloatMap>("distancesMuPlus");
  produces<detIdToFloatMap>("distancesMuMinus");
  
  edm::ParameterSet cfgTrackAssociator = cfg.getParameter<edm::ParameterSet>("trackAssociator");
  trackAssociatorParameters_.loadParameters(cfgTrackAssociator);
  trackAssociator_.useDefaultPropagator();
}

MuonCaloDistanceProducer::~MuonCaloDistanceProducer()
{
// nothing to be done yet...
}

void MuonCaloDistanceProducer::produce(edm::Event& evt, const edm::EventSetup& es)
{
  std::auto_ptr<detIdToFloatMap> distanceMuPlus(new detIdToFloatMap());
  std::auto_ptr<detIdToFloatMap> distanceMuMinus(new detIdToFloatMap());

  std::vector<reco::CandidateBaseRef > selMuons = getSelMuons(evt, srcSelectedMuons_);
  const reco::CandidateBaseRef muPlus  = getTheMuPlus(selMuons);
  const reco::CandidateBaseRef muMinus = getTheMuMinus(selMuons);
  
  fillDistanceMap(evt, es, dynamic_cast<const reco::Muon*>(&*muPlus), *distanceMuPlus);
  fillDistanceMap(evt, es, dynamic_cast<const reco::Muon*>(&*muMinus), *distanceMuMinus);

  std::auto_ptr<reco::CandidateCollection> muons(new reco::CandidateCollection);
  muons->push_back(new reco::ShallowCloneCandidate(muPlus));
  muons->push_back(new reco::ShallowCloneCandidate(muMinus));

  // References to the muons themselves
  evt.put(muons, "muons");

  // maps of detId to distance traversed by muon through calorimeter cell
  evt.put(distanceMuPlus, "distancesMuPlus");
  evt.put(distanceMuMinus, "distancesMuMinus");
}

void MuonCaloDistanceProducer::fillDistanceMap(edm::Event& evt, const edm::EventSetup& es, const reco::Muon* muon, detIdToFloatMap& distanceMap)
{
  if ( muon->globalTrack().isNull() ) 
    throw cms::Exception("InvalidData") 
      << "Muon is not a global muon: Pt = " << muon->pt() << ", eta = " << muon->eta() << ", phi = " << muon->phi() << " !!\n";
  
  TrackDetMatchInfo trackDetMatchInfo = trackAssociator_.associate(evt, es, *muon->globalTrack(), trackAssociatorParameters_);

  typedef std::map<std::string, const std::vector<DetId>*> CaloToDetIdMap;
  CaloToDetIdMap caloToDetIdMap;
  caloToDetIdMap["ecal"] = &(trackDetMatchInfo.crossedEcalIds);
  caloToDetIdMap["hcal"] = &(trackDetMatchInfo.crossedHcalIds);
  caloToDetIdMap["ho"]   = &(trackDetMatchInfo.crossedHOIds);
  caloToDetIdMap["es"]   = &(trackDetMatchInfo.crossedPreshowerIds);

  edm::ESHandle<CaloGeometry> caloGeo;
  es.get<CaloGeometryRecord>().get(caloGeo); 

  for ( CaloToDetIdMap::const_iterator caloToDetIdEntry = caloToDetIdMap.begin();
	caloToDetIdEntry != caloToDetIdMap.end(); ++caloToDetIdEntry ) {
    std::vector<SteppingHelixStateInfo>::const_iterator itHelixState_first, itHelixState_last;
    if ( caloToDetIdEntry->first == "ecal" ) {
      itHelixState_first = trackAssociator_.getCachedTrajectory().getEcalTrajectory().begin();
      itHelixState_last  = trackAssociator_.getCachedTrajectory().getEcalTrajectory().end();
    } else if ( caloToDetIdEntry->first == "hcal" ) {
      itHelixState_first = trackAssociator_.getCachedTrajectory().getHcalTrajectory().begin();
      itHelixState_last  = trackAssociator_.getCachedTrajectory().getHcalTrajectory().end();
    } else if ( caloToDetIdEntry->first == "ho" ) {
      itHelixState_first = trackAssociator_.getCachedTrajectory().getHOTrajectory().begin();
      itHelixState_last  = trackAssociator_.getCachedTrajectory().getHOTrajectory().end();
    } else if ( caloToDetIdEntry->first == "es" ) {
      itHelixState_first = trackAssociator_.getCachedTrajectory().getPreshowerTrajectory().begin();
      itHelixState_last  = trackAssociator_.getCachedTrajectory().getPreshowerTrajectory().end();
    } else assert(0);
    
    // copy trajectory points
    std::vector<GlobalPoint> trajectory;
    for ( std::vector<SteppingHelixStateInfo>::const_iterator helixState = itHelixState_first;
	  helixState != itHelixState_last; ++helixState ) {
      trajectory.push_back(helixState->position());
    }
    
    // iterate over crossed detIds
    for ( std::vector<DetId>::const_iterator detId = caloToDetIdEntry->second->begin();
	  detId != caloToDetIdEntry->second->end(); ++detId ) {
      if ( detId->rawId() == 0 ) continue;
      
      const CaloSubdetectorGeometry* subDetGeo = caloGeo->getSubdetectorGeometry(*detId);
      const CaloCellGeometry* caloCellGeo = subDetGeo->getGeometry(*detId);
      GlobalPoint previousPoint;
      bool previousPoint_initialized;
      float distanceWithinDetId = 0;
      for ( std::vector<GlobalPoint>::const_iterator point = trajectory.begin();
	    point != trajectory.end(); ++point ) {
	if ( previousPoint_initialized ) {
	  float dx = point->x() - previousPoint.x();
	  float dy = point->y() - previousPoint.y();
	  float dz = point->z() - previousPoint.z();
	  float distanceBetweenPoints = sqrt(dx*dx + dy*dy + dz*dz);
	  int numSteps = 100;
	  int numStepsWithinDetId = 0;
	  for ( int iStep = 0; iStep <= numSteps; ++iStep ){
	    float stepX = previousPoint.x() + iStep*dx/numSteps;
	    float stepY = previousPoint.y() + iStep*dy/numSteps;
	    float stepZ = previousPoint.z() + iStep*dz/numSteps;
	    GlobalPoint stepPoint(stepX, stepY, stepZ);	    
	    bool isWithinDetId = caloCellGeo->inside(stepPoint);
	    if ( isWithinDetId ) ++numStepsWithinDetId;
	  }
	  distanceWithinDetId += (numStepsWithinDetId/float(numSteps + 1))*distanceBetweenPoints;
	}
	previousPoint = (*point);
	previousPoint_initialized = true;
      } 
      distanceMap[detId->rawId()] = distanceWithinDetId;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(MuonCaloDistanceProducer);
