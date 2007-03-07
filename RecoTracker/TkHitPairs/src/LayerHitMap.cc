
#include "RecoTracker/TkHitPairs/interface/LayerHitMap.h"
#include "RecoTracker/TkHitPairs/interface/LayerHitMapLoop.h"

#include <cmath>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCellManager.h"
#include "RecoTracker/TkHitPairs/interface/TkHitPairsCacheCell.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"

using namespace std;

LayerHitMap::LayerHitMap(const LayerWithHits* layerhits,const edm::EventSetup&iSetup) : theCells(0)
{
  static int nRZ=5 ;
  static int nPhi=10;
  theNbinsRZ = nRZ;
  theNbinsPhi = nPhi;
  
  theLayerPhimin = -M_PI;
  theCellDeltaPhi = 2*M_PI/theNbinsPhi;
  if (layerhits->layer()->location() == GeomDetEnumerators::barrel) {
      float z0=layerhits->layer()->surface().position().z();
    float length =layerhits->layer()->surface().bounds().length();
    theLayerRZmin = z0 - length/2.;
    theCellDeltaRZ = length/theNbinsRZ;
  }
  else {
    const ForwardDetLayer* lf = dynamic_cast<const ForwardDetLayer*>(layerhits->layer());
    theLayerRZmin = lf->specificSurface().innerRadius();
    float  theLayerRZmax = lf->specificSurface().outerRadius();
    theCellDeltaRZ = (theLayerRZmax-theLayerRZmin)/theNbinsRZ;
  }

  for (vector<const TrackingRecHit*>::const_iterator ih=layerhits->recHits().begin();
       ih != layerhits->recHits().end(); ih++){
    theHits.push_back( TkHitPairsCachedHit(*ih,iSetup));
  }
}
  

      

LayerHitMap::LayerHitMap(const LayerHitMap & lhm,const edm::EventSetup&iSetup)
    : theCells(0),
      theHits(lhm.theHits),
      theLayerRZmin(lhm.theLayerRZmin),
      theCellDeltaRZ(lhm.theCellDeltaRZ),
      theLayerPhimin(lhm.theLayerPhimin),
      theCellDeltaPhi(lhm.theCellDeltaPhi),
      theNbinsRZ(lhm.theNbinsRZ), theNbinsPhi(lhm.theNbinsPhi) 
{
  if(lhm.theCells) initCells(); }

LayerHitMapLoop LayerHitMap::loop() const
  { return LayerHitMapLoop(*this); }

LayerHitMapLoop LayerHitMap::loop(
    const Range & phiRange, const Range & rzRange) const
{
    if(!theCells) initCells();
 
    return LayerHitMapLoop(*this,phiRange,rzRange); }

void LayerHitMap::initCells() const
{
//  static TimingReport::Item * theTimer =
//    PixelRecoUtilities::initTiming("-- cache (9) sorting",1);
//  TimeMe tm( *theTimer, false);
  vector<TkHitPairsCachedHit> hits(theHits);
  int size = hits.size();

  typedef vector<TkHitPairsCachedHit*> Cell;
  Cell aCell; aCell.reserve(2*size/theNbinsRZ);
  vector<Cell> cells(theNbinsRZ, aCell);

  vector<TkHitPairsCachedHit>::iterator ih;
  for (ih = hits.begin(); ih != hits.end(); ih++) {
    int irz = idxRz(ih->rOrZ());
    // --- FIX MANDATORY  to make caching work also with SiStrip RecHit
    // It is connected to the fact that sometimes the mathed hit are located
    // outside their gluedDet surface.
    if(irz>=theNbinsRZ) irz = theNbinsRZ-1;
    if(irz<0)   irz = 0;
    // ---
    cells[irz].push_back(&(*ih));
  }

  theCells = new TkHitPairsCellManager(theNbinsRZ, theNbinsPhi);
  Cell::const_iterator ph;
  
  int idx_theHits = 0;
  vector<TkHitPairsCachedHit>::iterator iBeg, iEnd, ie;
  for (int irz = 0; irz < theNbinsRZ; irz++) {
    Cell & vec = cells[irz];
    sort(vec.begin(), vec.end(), TkHitPairsCacheCell::lessPhiHitHit);
    iBeg = theHits.begin()+idx_theHits;
    for (ph = vec.begin(); ph < vec.end(); ph++) theHits[idx_theHits++] = **ph;
    iEnd = theHits.begin()+idx_theHits;
    for (int iphi = 0; iphi < theNbinsPhi; iphi++) {
      float upval= -M_PI + (iphi+1)*theCellDeltaPhi;
      ie = upper_bound( iBeg, iEnd, upval, TkHitPairsCacheCell::lessPhiValHit);
      cell(irz, iphi) = TkHitPairsCacheCell(iBeg,ie);
      iBeg = ie;
    }
  }
}

