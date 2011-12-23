#include "RecoTracker/TkNavigation/interface/StartingLayerFinder.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


#include <utility>

using namespace std;

vector<const DetLayer*> 
StartingLayerFinder::startingLayers(const FTS& aFts, float dr, float dz) const {



  
  vector<const DetLayer*> mylayers; 
  mylayers.reserve(3);

  FTS fastFts(aFts.parameters());
  

  //barrel pixel
  TSOS pTsos = 
    propagator()->propagate(fastFts, firstPixelBarrelLayer()->surface());
  
  if(pTsos.isValid()) {

    Range barrZRange(firstPixelBarrelLayer()->position().z() - 
		    0.5*(firstPixelBarrelLayer()->surface().bounds().length()),
		     firstPixelBarrelLayer()->position().z() + 
		   0.5*(firstPixelBarrelLayer()->surface().bounds().length()));
    Range trajZRange(pTsos.globalPosition().z() - dz,
		     pTsos.globalPosition().z() + dz);

    if(rangesIntersect(trajZRange, barrZRange)) {
      mylayers.push_back(firstPixelBarrelLayer());

    }
  }


  //negative fwd pixel

  const vector<ForwardDetLayer*> nfwd = firstPosPixelFwdLayer();
  for(vector<ForwardDetLayer*>::const_iterator infwd = nfwd.begin();
      infwd != nfwd.end(); infwd++) {
    pTsos = propagator()->propagate(fastFts, (*infwd)->surface());  
    if(pTsos.isValid()) {
      Range nfwdRRange((*infwd)->specificSurface().innerRadius(),
		       (*infwd)->specificSurface().outerRadius());
      Range trajRRange(pTsos.globalPosition().perp() - dr,
		       pTsos.globalPosition().perp() + dr);
      if(rangesIntersect(trajRRange, nfwdRRange)) {
	mylayers.push_back(*infwd);

      }
    }
  }

  //positive fwd pixel
  const vector<ForwardDetLayer*> pfwd = firstPosPixelFwdLayer();
  for(vector<ForwardDetLayer*>::const_iterator ipfwd = pfwd.begin();
      ipfwd != pfwd.end(); ipfwd++) {
    pTsos = propagator()->propagate(fastFts, (*ipfwd)->surface());
    if(pTsos.isValid()) {
      Range pfwdRRange((*ipfwd)->specificSurface().innerRadius(),
		       (*ipfwd)->specificSurface().outerRadius());
      Range trajRRange(pTsos.globalPosition().perp() - dr,
		       pTsos.globalPosition().perp() + dr);
      if(rangesIntersect(trajRRange, pfwdRRange)) {
	mylayers.push_back(*ipfwd);

      }
    }
  }



  return mylayers;





}

vector<const DetLayer*> 
StartingLayerFinder::startingLayers(const TrajectorySeed& aSeed) const {



  float dr = 0., dz = 0.;


  if(propagator()->propagationDirection() != aSeed.direction())
    return vector<const DetLayer*>();

  if(aSeed.nHits() != 2) return vector<const DetLayer*>();
 

  TrackingRecHitCollection::const_iterator firstHit= aSeed.recHits().first;
  const TrackingRecHit* recHit1=&(*firstHit);
  const DetLayer* hit1Layer = theMeasurementTracker->geometricSearchTracker()->detLayer(recHit1->geographicalId());

  TrackingRecHitCollection::const_iterator secondHit= aSeed.recHits().second;
  const TrackingRecHit* recHit2=&(*secondHit);
  const DetLayer* hit2Layer = theMeasurementTracker->geometricSearchTracker()->detLayer(recHit2->geographicalId());

  
  GeomDetEnumerators::Location p1 =  hit1Layer->location();
  GeomDetEnumerators::Location p2 =  hit2Layer->location();

  if(p1 == GeomDetEnumerators::barrel && p2 == GeomDetEnumerators::barrel) {
    dr = 0.1; dz = 5.;
  } else if(p1 == GeomDetEnumerators::endcap && p2 == GeomDetEnumerators::endcap) {
    dr = 5.; dz = 0.1;
  } else {
    dr = 0.1; dz = 0.1;
  }


  
  const GeomDet* gdet = theMeasurementTracker->geomTracker()->idToDet( DetId( aSeed.startingState().detId()));
  
  
  TrajectoryStateOnSurface tsos = trajectoryStateTransform::transientState( aSeed.startingState(), &(gdet->surface()), 
							      thePropagator->magneticField());


  FreeTrajectoryState* fts=tsos.freeTrajectoryState();
  
  return startingLayers(*fts, dr, dz);
}
  
const BarrelDetLayer* StartingLayerFinder::firstPixelBarrelLayer() const {
  checkPixelLayers();
  return theFirstPixelBarrelLayer;  
}

const vector<ForwardDetLayer*> StartingLayerFinder::firstNegPixelFwdLayer() const {
  checkPixelLayers();
  return theFirstNegPixelFwdLayer;
}

const vector<ForwardDetLayer*>  StartingLayerFinder::firstPosPixelFwdLayer() const {
  checkPixelLayers();
  return theFirstPosPixelFwdLayer;
}

void StartingLayerFinder::checkPixelLayers() const {


  if(!thePixelLayersValid) {
   
    const GeometricSearchTracker* theGeometricSearchTracker=theMeasurementTracker->geometricSearchTracker();

   
    theFirstPixelBarrelLayer = theGeometricSearchTracker->pixelBarrelLayers().front();
    theFirstNegPixelFwdLayer = theGeometricSearchTracker->negPixelForwardLayers();
    theFirstPosPixelFwdLayer = theGeometricSearchTracker->posPixelForwardLayers();
    thePixelLayersValid = true;


 
  }


}



