#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include <cmath>
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkTrackingRegions/interface/HitEtaCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRCheck.h"
#include "RecoTracker/TkTrackingRegions/interface/HitZCheck.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
template <class T> T sqr( T t) {return t*t;}

using namespace GeomDetEnumerators;

std::vector<ctfseeding::SeedingHit> GlobalTrackingRegion::hits(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const ctfseeding::SeedingLayer* layer) const
{
 return layer->hits(ev,es);
}

HitRZCompatibility* GlobalTrackingRegion::
checkRZ(const DetLayer* layer, 
	const TrackingRecHit* outerHit,
	const edm::EventSetup& iSetup) const
{


  edm::ESHandle<TrackerGeometry> tracker;
  iSetup.get<TrackerDigiGeometryRecord>().get(tracker);
  //MP
  bool isBarrel = (layer->location() == barrel);
  bool isPixel = (layer->subDetector() == PixelBarrel || layer->subDetector() == PixelEndcap);

  

    GlobalPoint ohit =  tracker->idToDet(outerHit->geographicalId())->surface().toGlobal(outerHit->localPosition());
 
  PixelRecoPointRZ outer(ohit.perp(), ohit.z());
  PixelRecoPointRZ vtxR = (outer.z() > origin().z()+originZBound()) ?
      PixelRecoPointRZ(-originRBound(), origin().z()+originZBound())
    : PixelRecoPointRZ( originRBound(), origin().z()+originZBound());
  PixelRecoPointRZ vtxL = (outer.z() < origin().z()-originZBound()) ?
      PixelRecoPointRZ(-originRBound(), origin().z()-originZBound())
    : PixelRecoPointRZ( originRBound(), origin().z()-originZBound()); 

  if ((!thePrecise) &&(isPixel )) {
    double VcotMin = PixelRecoLineRZ( vtxR, outer).cotLine();
    double VcotMax = PixelRecoLineRZ( vtxL, outer).cotLine();
    return new HitEtaCheck(isBarrel, outer, VcotMax, VcotMin);
  }
  
  float errZ = hitErrZ(layer);
  float errR = hitErrR(layer);

  PixelRecoPointRZ  outerL, outerR;

  if (layer->location() == barrel) {

    outerL = PixelRecoPointRZ(outer.r(), outer.z()-errZ);
    outerR = PixelRecoPointRZ(outer.r(), outer.z()+errZ);
  } else if (outer.z() > 0) {

    outerL = PixelRecoPointRZ(outer.r()+errR, outer.z());
    outerR = PixelRecoPointRZ(outer.r()-errR, outer.z());
  } else {
    outerL = PixelRecoPointRZ(outer.r()-errR, outer.z());
    outerR = PixelRecoPointRZ(outer.r()+errR, outer.z());
  }
  

  MultipleScatteringParametrisation iSigma(layer,iSetup);
  PixelRecoPointRZ vtxMean(0.,origin().z());
  float innerScatt = 3 * iSigma( ptMin(), vtxMean, outer);

  PixelRecoLineRZ leftLine( vtxL, outerL);
  PixelRecoLineRZ rightLine( vtxR, outerR);

  HitRZConstraint rzConstraint(leftLine, rightLine);
  float cotTheta = PixelRecoLineRZ(vtxMean,outer).cotLine();

  if (isBarrel) {
    float sinTheta = 1/sqrt(1+sqr(cotTheta));
    float corrZ = innerScatt/sinTheta + errZ;
    return new HitZCheck(rzConstraint, HitZCheck::Margin(corrZ,corrZ));
  } else {
    float cosTheta = 1/sqrt(1+sqr(1/cotTheta));
    float corrR = innerScatt/cosTheta + errR;
    return new HitRCheck( rzConstraint, HitRCheck::Margin(corrR,corrR));
  }
}

