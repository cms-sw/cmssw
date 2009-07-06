#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include <cmath>
#include <sstream>
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

std::string GlobalTrackingRegion::print() const {
  std::ostringstream str;
  str << TrackingRegionBase::print() << "precise: "<<thePrecise;
  return str.str(); 
}

TrackingRegion::Hits GlobalTrackingRegion::hits(
      const edm::Event& ev,
      const edm::EventSetup& es,
      const ctfseeding::SeedingLayer* layer) const
{
 return layer->hits(ev,es);
}

HitRZCompatibility* GlobalTrackingRegion::checkRZ(const DetLayer* layer, 
	const Hit& outerHit, const edm::EventSetup& iSetup) const
{

  bool isBarrel = (layer->location() == barrel);
  bool isPixel = (layer->subDetector() == PixelBarrel || layer->subDetector() == PixelEndcap);
  

  GlobalPoint ohit =  outerHit->globalPosition();
  float outerred_r = sqrt( sqr(ohit.x()-origin().x())+sqr(ohit.y()-origin().y()) );
  PixelRecoPointRZ outerred(outerred_r, ohit.z());


  PixelRecoPointRZ vtxR = (outerred.z() > origin().z()+originZBound()) ?
      PixelRecoPointRZ(-originRBound(), origin().z()+originZBound())
    : PixelRecoPointRZ( originRBound(), origin().z()+originZBound());
  PixelRecoPointRZ vtxL = (outerred.z() < origin().z()-originZBound()) ?
      PixelRecoPointRZ(-originRBound(), origin().z()-originZBound())
    : PixelRecoPointRZ( originRBound(), origin().z()-originZBound()); 

  if ((!thePrecise) &&(isPixel )) {
    double VcotMin = PixelRecoLineRZ( vtxR, outerred).cotLine();
    double VcotMax = PixelRecoLineRZ( vtxL, outerred).cotLine();
    return new HitEtaCheck(isBarrel, outerred, VcotMax, VcotMin);
  }
  
  float nSigmaPhi = 3.;
  float errZ =  nSigmaPhi*outerHit->errorGlobalZ(); 
  float errR =  nSigmaPhi*outerHit->errorGlobalR();

  PixelRecoPointRZ  outerL, outerR;

  if (layer->location() == barrel) {
    outerL = PixelRecoPointRZ(outerred.r(), outerred.z()-errZ);
    outerR = PixelRecoPointRZ(outerred.r(), outerred.z()+errZ);
  } 
  else if (outerred.z() > 0) {
    outerL = PixelRecoPointRZ(outerred.r()+errR, outerred.z());
    outerR = PixelRecoPointRZ(outerred.r()-errR, outerred.z());
  } 
  else {
    outerL = PixelRecoPointRZ(outerred.r()-errR, outerred.z());
    outerR = PixelRecoPointRZ(outerred.r()+errR, outerred.z());
  }
  
  MultipleScatteringParametrisation iSigma(layer,iSetup);
  PixelRecoPointRZ vtxMean(0.,origin().z());
  float innerScatt = 3 * iSigma( ptMin(), vtxMean, outerred);

  //
  //
  //
  PixelRecoLineRZ leftLine( vtxL, outerL);
  PixelRecoLineRZ rightLine( vtxR, outerR);
  HitRZConstraint rzConstraint(leftLine, rightLine);
  float cotTheta = PixelRecoLineRZ(vtxMean,outerred).cotLine();

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
