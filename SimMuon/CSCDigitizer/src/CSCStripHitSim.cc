#include "SimMuon/CSCDigitizer/src/CSCStripHitSim.h"
#include "Geometry/CSCGeometry/interface/CSCLayer.h"
#include "Geometry/CSCGeometry/interface/CSCChamberSpecs.h"
#include "Geometry/CSCGeometry/interface/CSCLayerGeometry.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"

// This is CSCStripHitSim.cc
// Author: Rick Wilkinson, Tim Cox


std::vector<CSCDetectorHit> &
CSCStripHitSim::simulate(const CSCLayer * layer, 
                           const std::vector<CSCDetectorHit> & wireHits)
{
  // make sure the gatti function is initialized
  const CSCChamberSpecs * chamberSpecs = layer->chamber()->specs();
  const CSCLayerGeometry* geom = layer->geometry();
  theGattiFunction.initChamberSpecs(*chamberSpecs);
  const int nNodes = chamberSpecs->nNodes();
  
  // for every wire hit, induce a Gatti-distributed charge on the
  // cathode strips
  newStripHits.clear();
  newStripHits.reserve((2*nNodes+1)*wireHits.size());
  std::vector<CSCDetectorHit>::const_iterator wireHitI;
  for(wireHitI = wireHits.begin(); wireHitI != wireHits.end(); ++wireHitI){
    int   wire        = (*wireHitI).getElement();
    float wireCharge  = (*wireHitI).getCharge();
    float wireHitTime = (*wireHitI).getTime();
    // The wire hit position is _along the wire_, measured from where
    // the wire intersects local y axis, so convert to local x...
    float hitX   = (*wireHitI).getPosition() * cos(geom->wireAngle());
    float hitY   = geom->yOfWire(wire, hitX);
    const LocalPoint wireHitPos(hitX, hitY);

    int centerStrip = geom->nearestStrip(wireHitPos);
    int firstStrip = std::max(centerStrip - nNodes, 1);
    int lastStrip  = std::min(centerStrip + nNodes, geom->numberOfStrips());
    for(int istrip = firstStrip; istrip <= lastStrip; istrip++) {
      float offset = hitX - geom->xOfStrip(istrip, hitY);
      float stripWidth = geom->stripPitch(wireHitPos);
      float binValue = theGattiFunction.binValue(offset, stripWidth);
      // we divide by 2 because charge goes on either cathode.
      // if you're following the TDR, we already multiplied the
      // charge by 0.82 in the WireHitSim (well, DriftSim), so that explains 
      // their f_ind=0.41.
   
      // this seems to be folded in the Amp response, which peaks
      // around 0.14.  The difference may be because the amp response
      // convolutes in different drift times.
      //float collectionFraction = 0.19;
      const float igain = 1./0.9; // mv/fC
      float stripCharge = wireCharge * binValue * igain * 0.5;
      float stripTime = wireHitTime;
      float position = hitY / sin(geom->stripAngle(istrip));
      CSCDetectorHit newStripHit(istrip, stripCharge, position, stripTime, 
                                   (*wireHitI).getSimHit());
      newStripHits.push_back(newStripHit);
    }
  }  // loop over wire hits
  return newStripHits;
}
