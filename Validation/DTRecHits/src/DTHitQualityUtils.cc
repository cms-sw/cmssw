
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/12/03 12:17:41 $
 *  $Revision: 1.7 $
 *  \author S. Bolognesi and G. Cerminara - INFN Torino
 */


#include "Validation/DTRecHits/interface/DTHitQualityUtils.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include <iostream>

using namespace std;
using namespace edm;

bool DTHitQualityUtils::debug;

// Constructor
DTHitQualityUtils::DTHitQualityUtils(){
  //DTHitQualityUtils::setDebug(debug);
}

// Destructor
DTHitQualityUtils::~DTHitQualityUtils(){
}


// Return a map between simhits of a layer,superlayer or chamber and the wireId of their cell
map<DTWireId, PSimHitContainer >
DTHitQualityUtils::mapSimHitsPerWire(const PSimHitContainer& simhits) {
  map<DTWireId, PSimHitContainer > hitWireMapResult;
   
  for(PSimHitContainer::const_iterator simhit = simhits.begin();
      simhit != simhits.end();
      simhit++) {
    hitWireMapResult[DTWireId((*simhit).detUnitId())].push_back(*simhit);
  }
   
  return hitWireMapResult;
}

// Extract mu simhits from a map of simhits by wire and map them by wire
map<DTWireId, const PSimHit*>
DTHitQualityUtils::mapMuSimHitsPerWire(const map<DTWireId, PSimHitContainer>& simHitWireMap) {

  map<DTWireId, const PSimHit*> ret;
  
  for(map<DTWireId, PSimHitContainer>::const_iterator wireAndSimHit = simHitWireMap.begin();
      wireAndSimHit != simHitWireMap.end();
      wireAndSimHit++) {

    const PSimHit* muHit = findMuSimHit((*wireAndSimHit).second);
    if(muHit != 0) {
      ret[(*wireAndSimHit).first]=(muHit);
    }
  }
  return ret;
}


// Find the sim hit from muon track in a vector of simhits
// If no mu sim hit is found then it returns a null pointer
const PSimHit* DTHitQualityUtils::findMuSimHit(const PSimHitContainer& hits) {
  //PSimHitContainer muHits;
  vector<const PSimHit*> muHits;
 
   // Loop over simhits
  for (PSimHitContainer::const_iterator hit=hits.begin();
       hit != hits.end(); hit++) {
    if (abs((*hit).particleType())==13) muHits.push_back(&(*hit));
  }

  if (muHits.size()==0)
    return 0; //FIXME: Throw of exception???
  else if (muHits.size()>1)
    if(debug)
      cout << "[DTHitQualityUtils]***WARNING: # muSimHits in a wire = " << muHits.size() << endl;

  return (muHits.front());
}


// Find Innermost and outermost SimHit from Mu in a SL (they identify a simulated segment)
pair<const PSimHit*, const PSimHit*>
DTHitQualityUtils::findMuSimSegment(const map<DTWireId, const PSimHit*>& mapWireAndMuSimHit) {

  int outSL = 0;
  int inSL = 4;
  int outLayer = 0;
  int inLayer = 5;
  const PSimHit *inSimHit = 0;
  const PSimHit *outSimHit = 0;

  for(map<DTWireId, const PSimHit*>::const_iterator wireAndMuSimHit = mapWireAndMuSimHit.begin();
      wireAndMuSimHit != mapWireAndMuSimHit.end();
      wireAndMuSimHit++) {
    
    const DTWireId wireId = (*wireAndMuSimHit).first;
    const PSimHit *theMuHit = (*wireAndMuSimHit).second;

    int sl = ((wireId.layerId()).superlayerId()).superLayer();
    int layer = (wireId.layerId()).layer();

    if(sl == outSL) {
      if(layer > outLayer) {
	outLayer = layer;
	outSimHit = theMuHit;
      }
    }
    if(sl > outSL) {
      outSL = sl;
      outLayer = layer;
      outSimHit = theMuHit;
    }
    if(sl == inSL) {
      if(layer < inLayer) {
	inLayer = layer;
	inSimHit = theMuHit;
      }
    }
    if(sl < inSL) {
      inSL = sl;
      inLayer = layer;
      inSimHit = theMuHit;
    }
  }

  if(inSimHit != 0) {
    if(debug)
      cout << "Innermost SimHit on SL: " << inSL << " layer: " << inLayer << endl;
  } else {
    cout << "[DTHitQualityUtils]***Error: No Innermost SimHit found!!!" << endl;
    abort();
  }

  if(outSimHit != 0) {
    if(debug)
      cout << "Outermost SimHit on SL: " << outSL << " layer: " << outLayer << endl;
  } else {
    cout << "[DTHitQualityUtils]***Error: No Outermost SimHit found!!!" << endl;
    abort();
  }

  // //Check that outermost and innermost SimHit are not the same
  // if(outSimHit == inSimHit) {
  //   cout << "[DTHitQualityUtils]***Warning: outermost and innermost SimHit are the same!" << endl;
  //   abort();
  //     }
  return make_pair(inSimHit, outSimHit);
}



// Find direction and position of a segment (in local RF) from outer and inner mu SimHit in the RF of object Det
// (Concrete implementation of Det are MuBarSL and MuBarChamber)
pair<LocalVector, LocalPoint>
DTHitQualityUtils::findMuSimSegmentDirAndPos(const pair<const PSimHit*, const PSimHit*>& inAndOutSimHit,
					     const DetId detId, const DTGeometry *muonGeom) {

  //FIXME: What should happen if outSimHit = inSimHit???? Now, this case is not considered
  const PSimHit* innermostMuSimHit = inAndOutSimHit.first;
  const PSimHit* outermostMuSimHit = inAndOutSimHit.second;

  //Find simulated segment direction from SimHits position
  const DTLayer* layerIn = muonGeom->layer((DTWireId(innermostMuSimHit->detUnitId())).layerId()); 
  const DTLayer* layerOut = muonGeom->layer((DTWireId(outermostMuSimHit->detUnitId())).layerId()); 
  GlobalPoint inGlobalPos = layerIn->toGlobal(innermostMuSimHit->localPosition());
  GlobalPoint outGlobalPos = layerOut->toGlobal(outermostMuSimHit->localPosition());
  LocalVector simHitDirection = (muonGeom->idToDet(detId))->toLocal(inGlobalPos - outGlobalPos);
  simHitDirection = -simHitDirection.unit();

  //SimHit position extrapolated at z=0 in the Det RF
  LocalPoint outLocalPos = (muonGeom->idToDet(detId))->toLocal(outGlobalPos);
  LocalPoint simSegLocalPosition =
    outLocalPos + simHitDirection*(-outLocalPos.z()/(simHitDirection.mag()*cos(simHitDirection.theta())));

  return make_pair(simHitDirection, simSegLocalPosition);
}

// Find the angles from a segment direction:
// NB: For 4D RecHits: 
//                        Alpha = angle measured by SL RPhi
//                        Beta  = angle measured by SL RZ
//     For 2D RecHits: only Alpha makes sense
pair<double, double> DTHitQualityUtils::findSegmentAlphaAndBeta(const LocalVector& direction) {
  //return make_pair(atan(direction.x()/direction.z()), atan(direction.y()/direction.z()));
  return make_pair((direction.x()/direction.z()), (direction.y()/direction.z()));
}










