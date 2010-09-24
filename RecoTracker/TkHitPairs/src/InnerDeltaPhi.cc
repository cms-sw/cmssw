#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"


using namespace std;

#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"

template <class T> inline T sqr( T t) {return t*t;}

inline double cropped_asin(double x) {
    return abs(x) <= 1 ? asin(x) : (x > 0 ? M_PI/2 : -M_PI/2);
}

namespace {
  double checked_asin(double x, const char *expr, const char *file, int line) {
    if (fabs(x) >= 1.0) throw cms::Exception("CorruptData") <<  "asin(x) called with x = " << expr << " = " << x << "\n\tat " << file << ":" << line << "\n";
    return asin(x);
  }
}

//#define asin(X) checked_asin(X, #X, __FILE__, __LINE__)

InnerDeltaPhi:: InnerDeltaPhi( const DetLayer& layer,
                 const TrackingRegion & region,
                 const edm::EventSetup& iSetup,
                 bool precise, float extraTolerance)
  : thePrecise(precise),
    theROrigin(region.originRBound()),
    theRLayer(0),
    theThickness(0),
    theExtraTolerance(extraTolerance),
    theA(0),
    theB(0),
    theVtxZ(region.origin().z()),
    thePtMin(region.ptMin()),
    theVtx(region.origin().x(),region.origin().y()),
    sigma(&layer,iSetup)
{
  float zMinOrigin = theVtxZ-region.originZBound();
  float zMaxOrigin = theVtxZ+region.originZBound();
  theRCurvature = PixelRecoUtilities::bendingRadius(thePtMin,iSetup);
 

  if (layer.location() == GeomDetEnumerators::barrel) initBarrelLayer( layer);
  else initForwardLayer( layer, zMinOrigin, zMaxOrigin);

}


InnerDeltaPhi::~InnerDeltaPhi() {}

void InnerDeltaPhi::initBarrelLayer( const DetLayer& layer) 
{
  const BarrelDetLayer& bl = static_cast<const BarrelDetLayer&>(layer); 
  float rLayer = bl.specificSurface().radius(); 

  // the maximal delta phi will be for the innermost hits
  theThickness = layer.surface().bounds().thickness();
  theRLayer = rLayer - 0.5f*theThickness;
  theRDefined = true;
}

void InnerDeltaPhi::initForwardLayer( const DetLayer& layer, 
				 float zMinOrigin, float zMaxOrigin)
{
  const ForwardDetLayer &fl = static_cast<const ForwardDetLayer&>(layer);
  theRLayer = fl.specificSurface().innerRadius();
  float layerZ = layer.position().z();
  theThickness = layer.surface().bounds().thickness();
  float layerZmin = layerZ > 0 ? layerZ-0.5f*theThickness: layerZ+0.5f*theThickness;
  theB = layerZ > 0 ? zMaxOrigin : zMinOrigin;
  theA = layerZmin - theB;
  theRDefined = false;
}



PixelRecoRange<float> InnerDeltaPhi::phiRange(const Point2D& hitXY,float hitZ,float errRPhi) const
{
  double rLayer = theRLayer;
  bool checkCrossing = true;
  Point2D crossing;

  Point2D dHit = hitXY-theVtx;
  double  dHitmag = dHit.mag();
  double  dLayer = 0.;
  double dL = 0.;
  //
  // compute crossing of stright track with inner layer
  //
  if (!theRDefined) {
    double t = theA/(hitZ-theB); double dt = std::abs(theThickness/(hitZ-theB));
    crossing = theVtx + t*dHit;
    rLayer =  crossing.mag();
    dLayer = t*dHitmag;           dL = dt * dHitmag; 
    checkCrossing = false;
    if (rLayer < theRLayer) {
      checkCrossing = true;
      rLayer = theRLayer;
      dL = 0.;
    } 
  }

  //
  // compute crossing of track with layer
  // dHit - from VTX to outer hit
  // rLayer - layer radius
  // dLayer - distance from VTX to inner layer in direction of dHit
  // vect(rLayer) = vect(rVTX) + vect(dHit).unit * dLayer
  //     rLayer^2 = (ect(rVTX) + vect(dHit).unit * dLayer)^2 and we have square eqation for dLayer 
  //
  // barrel case
  //

  if (checkCrossing) {
    double vtxmag2 = theVtx.mag2();
    if (vtxmag2 < 1.e-10) {
      dLayer = rLayer;
    }
    else { 
      double var_c = theVtx.mag2()-sqr(rLayer);
      double var_b = 2*theVtx.dot(dHit.unit());
      double var_delta = sqr(var_b)-4*var_c;
      if (var_delta <=0.) var_delta = 0;
      dLayer = 0.5*(-var_b + std::sqrt(var_delta)); //only the value along vector is OK. 
    }
    crossing = theVtx+ dHit.unit() * dLayer;
    double cosCross = std::abs( dHit.unit().dot(crossing.unit()));
    dL = theThickness/cosCross; 
  }


  // track is crossing layer with angle such as:
  // this factor should be taken in computation of eror projection
  double cosCross = std::abs( dHit.unit().dot(crossing.unit()));

  double alphaHit = cropped_asin( dHitmag/(2*theRCurvature)); 
  double deltaPhi = fabs( alphaHit - cropped_asin( dLayer/(2*theRCurvature)));
  deltaPhi *= dLayer/(rLayer*cosCross);  

  // additinal angle due to not perpendicular stright line crossing  (for displaced beam)
  //  double dPhiCrossing = (cosCross > 0.9999) ? 0 : dL *  sqrt(1-sqr(cosCross))/ rLayer;
  Point2D crossing2 = theVtx + dHit.unit()* (dLayer+dL);
  double phicross2 = crossing2.phi();  
  double phicross1 = crossing.phi();
  double dphicross = phicross2-phicross1;
  if (dphicross < -M_PI) dphicross += 2*M_PI;
  if (dphicross >  M_PI) dphicross -= 2*M_PI;
  if (dphicross > M_PI/2) dphicross = 0.;  // something wrong?
  phicross2 = phicross1 + dphicross;
        

  // compute additional delta phi due to origin radius
  double deltaPhiOrig = cropped_asin( theROrigin * (dHitmag-dLayer) / (dHitmag*dLayer));
  deltaPhiOrig *= dLayer/(rLayer*cosCross);

  // inner hit error taken as constant
  double deltaPhiHit = theExtraTolerance / rLayer;

  // outer hit error
//   double deltaPhiHitOuter = errRPhi/rLayer; 
    double deltaPhiHitOuter = errRPhi/hitXY.mag();

  double margin = deltaPhi+deltaPhiOrig+deltaPhiHit+deltaPhiHitOuter ;

  if (thePrecise) {
    // add multiple scattering correction
    PixelRecoPointRZ zero(0., theVtxZ);
    PixelRecoPointRZ point(hitXY.mag(), hitZ);
    double scatt = 3.f*sigma(thePtMin,zero, point) / rLayer; 
   
    margin += scatt ;
  }
  
  return PixelRecoRange<float>( std::min(phicross1,phicross2)-margin, 
                                std::max(phicross1,phicross2)+margin);
}

float InnerDeltaPhi::operator()( float rHit, float zHit, float errRPhi) const
{
  // alpha - angle between particle direction at vertex and position of hit.
  // (pi/2 - alpha) - angle hit-vertex-cernter_of_curvature
  // cos (pi/2 - alpha) = (hRhi/2) / theRCurvature
  // so:

  float alphaHit = std::asin( rHit/(2*theRCurvature));

  
  float rMin = minRadius( rHit, zHit);
  float deltaPhi = std::abs( alphaHit - std::asin( rMin/(2*theRCurvature)));

  // compute additional delta phi due to origin radius
  float deltaPhiOrig = asin( theROrigin * (rHit-rMin) / (rHit*rMin));

  // hit error taken as constant
  float deltaPhiHit = theExtraTolerance / rMin;

  if (!thePrecise) {
    return deltaPhi+deltaPhiOrig+deltaPhiHit;
  } else {
    // add multiple scattering correction
    PixelRecoPointRZ zero(0., theVtxZ);
    PixelRecoPointRZ point(rHit, zHit);
    float scatt = 3.f*sigma(thePtMin,zero, point) / rMin; 
    float deltaPhiHitOuter = errRPhi/rMin; 
   
    return deltaPhi+deltaPhiOrig+deltaPhiHit + scatt + deltaPhiHitOuter;
  }

}

PixelRecoRange<float> InnerDeltaPhi::operator()( 
    float rHit, float phiHit, float zHit, float errRPhi) const
{
//     float phiM =  operator()( rHit,zHit,errRPhi);
//     return PixelRecoRange<float>(phiHit-phiM,phiHit+phiM);

  Point2D hitXY( rHit*std::cos(phiHit), rHit*std::sin(phiHit));
  return phiRange(hitXY,zHit,errRPhi);
}

float InnerDeltaPhi::minRadius( float hitR, float hitZ) const 
{
  if (theRDefined) return theRLayer;
  else {
    float rmin = (theA*hitR)/(hitZ-theB);
    return ( rmin> 0) ? std::max( rmin, theRLayer) : theRLayer;
  }
}


