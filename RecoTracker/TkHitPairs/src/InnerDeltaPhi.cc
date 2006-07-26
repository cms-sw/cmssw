#include "RecoTracker/TkHitPairs/interface/InnerDeltaPhi.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"

InnerDeltaPhi::InnerDeltaPhi( const DetLayer& layer, 
			      float ptMin,  float rOrigin,
			      float zMinOrigin, float zMaxOrigin,
			      const edm::EventSetup& iSetup,
                        bool precise) :
  theROrigin( rOrigin), theRLayer(0),theA(0), theB(0), 
  thePtMin(ptMin), sigma(0), thePrecise(precise)
{

  theRCurvature = PixelRecoUtilities::bendingRadius(ptMin,iSetup);

  sigma = new MultipleScatteringParametrisation(&layer,iSetup);

  theVtxZ = (zMinOrigin + zMaxOrigin)/2.;

  if (layer.location() == GeomDetEnumerators::barrel){


    initBarrelLayer( layer);

  }
 
  else initForwardLayer( layer, zMinOrigin, zMaxOrigin);
}

InnerDeltaPhi::~InnerDeltaPhi() { delete sigma; }

void InnerDeltaPhi::initBarrelLayer( const DetLayer& layer) 
{
  const BarrelDetLayer& bl = dynamic_cast<const BarrelDetLayer&>(layer); 
  float rLayer = bl.specificSurface().radius(); 
//    dynamic_cast<const BarrelDetLayer&>(layer).specificSurface().radius();

  // the maximal delta phi will be for the innermost hits
  theRLayer = rLayer - layer.surface().bounds().thickness()/2;
  theHitError = TrackingRegionBase::hitErrRPhi( &bl);
  theRDefined = true;
}

void InnerDeltaPhi::initForwardLayer( const DetLayer& layer, 
				 float zMinOrigin, float zMaxOrigin)
{
  const ForwardDetLayer &fl = dynamic_cast<const ForwardDetLayer&>(layer);
  theRLayer = fl.specificSurface().innerRadius();
  float layerZ = layer.position().z();
  float halfthickness = layer.surface().bounds().thickness()/2.;
  float layerZmin = layerZ > 0 ? layerZ-halfthickness : layerZ+halfthickness;
  theB = layerZ > 0 ? zMaxOrigin : zMinOrigin;
  theA = layerZmin - theB;
  theRDefined = false;
  theHitError = TrackingRegionBase::hitErrRPhi(&fl);
}

float InnerDeltaPhi::operator()( float rHit, float zHit, float errRPhi) const
{

  // complementary angle to phi, asin is more accurate than acos for small angles
  float alphaHit = asin( rHit/(2*theRCurvature));

  float rMin = minRadius( rHit, zHit);
  float deltaPhi = fabs( alphaHit - asin( rMin/(2*theRCurvature)));

  // compute additional delta phi due to origin radius
  float deltaPhiOrig = asin( theROrigin * (rHit-rMin) / (rHit*rMin));

  // hit error taken as constant
  float deltaPhiHit = theHitError / rMin;

  if (!thePrecise) {
    return deltaPhi+deltaPhiOrig+deltaPhiHit;
  } else {
    // add multiple scattering correction
    PixelRecoPointRZ zero(0., theVtxZ);
    PixelRecoPointRZ point(rHit, zHit);
//    float scatt = 0;
    float scatt = 3*(*sigma)(thePtMin,zero, point) / rMin; 
    float deltaPhiHitOuter = errRPhi/rMin; 
    return deltaPhi+deltaPhiOrig+deltaPhiHit + scatt + deltaPhiHitOuter;
  }
}

// void InnerDeltaPhi::initTiming() 
// {
//   if (theTimingDone) return;

//   TimingReport& tr(*TimingReport::current());

//   theConstructTimer   =   &tr["InnerDeltaPhi construct"];
//   theDeltaPhiTimer    =   &tr["InnerDeltaPhi delta phi"];

//   static bool detailedTiming =
//     SimpleConfigurable<bool>(false,"TkHitPairs:detailedTiming");
//   if (!detailedTiming) {
//     theConstructTimer->switchOn(false);
//     theDeltaPhiTimer->switchOn(false);
//   }
//   theTimingDone = true;
// }

// TimingReport::Item* InnerDeltaPhi::theConstructTimer = 0;
// TimingReport::Item* InnerDeltaPhi::theDeltaPhiTimer = 0;
// bool InnerDeltaPhi::theTimingDone = false;
