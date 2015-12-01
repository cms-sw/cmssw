#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"


using namespace std;

namespace {
  struct Stat {

   float xmin=1.1;
   float xmax=-1.1;
   int nt=0;
   int nn=0;
   int nl=0;

   ~Stat() { std::cout << "ASIN " << xmin <<',' << xmax <<',' << nt <<','<< nn <<','<< nl << std::endl;}

  };

  Stat stat;

}

namespace {
  template <class T> inline T sqr( T t) {return t*t;}

  template <class T> 
  inline T cropped_asin(T x) {
    stat.nt++;
    if (x<0.f) stat.nn++;
    if (x>0.5f) stat.nl++;
    stat.xmin = std::min(x,stat.xmin);
    stat.xmax =	std::max(x,stat.xmax);
    
    return std::abs(x) <= 1 ? std::asin(x) : (x > 0 ? T(M_PI/2) : -T(M_PI/2));
  }
}

namespace {
  // valid only for |x|<0.5 then degrades...
  template<typename T>
  inline T f_asin05f(T xx)  {
    auto x = xx>0 ? xx : -xx;
    // x = x> 0.5f ? 0.5f : x;

    auto z = x * x;
    
    z =  (((( 4.2163199048E-2f * z
            + 2.4181311049E-2f) * z
            + 4.5470025998E-2f) * z
            + 7.4953002686E-2f) * z
            + 1.6666752422E-1f) * z * x
            + x;

    return xx>0 ? z : -z;
  }

  typedef float __attribute__( ( vector_size( 16 ) ) ) float32x4_t;


}


#include "DataFormats/Math/interface/approx_atan2.h"

namespace {
  inline
  float f_atan2f(float y, float x) { return unsafe_atan2f<9>(y,x); }
}

namespace {
  inline double checked_asin(double x, const char *expr, const char *file, int line) {
    if (fabs(x) >= 1.0) throw cms::Exception("CorruptData") <<  "asin(x) called with x = " << expr << " = " << x << "\n\tat " << file << ":" << line << "\n";
    return asin(x);
  }
}

//#define asin(X) checked_asin(X, #X, __FILE__, __LINE__)

InnerDeltaPhi:: InnerDeltaPhi( const DetLayer& outlayer, const DetLayer& layer,
                 const TrackingRegion & region,
                 const edm::EventSetup& iSetup,
                 bool precise, float extraTolerance)
  : thePrecise(precise),
    ol( outlayer.seqNum()), 
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
 

  if (layer.isBarrel()) initBarrelLayer( layer);
  else initForwardLayer( layer, zMinOrigin, zMaxOrigin);

}



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
  float rLayer = theRLayer;
  bool checkCrossing = true;
  Point2D crossing;

  Point2D dHit = hitXY-theVtx;
  auto  dHitmag = dHit.mag();
  float  dLayer = 0.;
  float dL = 0.;
  //
  // compute crossing of stright track with inner layer
  //
  if (!theRDefined) {
    auto t = theA/(hitZ-theB); auto dt = std::abs(theThickness/(hitZ-theB));
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
  //     rLayer^2 = (vect(rVTX) + vect(dHit).unit * dLayer)^2 and we have square eqation for dLayer 
  //
  // barrel case
  //

  if (checkCrossing) {
    auto vtxmag2 = theVtx.mag2();
    if (vtxmag2 < 1.e-10f) {
      dLayer = rLayer;
    }
    else { 
      // there are cancellation here....
      double var_c = vtxmag2-sqr(rLayer);
      double var_b = theVtx.dot(dHit.unit());
      double var_delta = sqr(var_b)-var_c;
      if (var_delta <=0.) var_delta = 0;
      dLayer = -var_b + std::sqrt(var_delta); //only the value along vector is OK. 
    }
    crossing = theVtx+ dHit.unit() * dLayer;
    float cosCross = std::abs( dHit.unit().dot(crossing.unit()));
    dL = theThickness/cosCross; 
  }


  // track is crossing layer with angle such as:
  // this factor should be taken in computation of eror projection
  auto cosCross = std::abs( dHit.unit().dot(crossing.unit()));

  
  float32x4_t num{dHitmag,dLayer,theROrigin * (dHitmag-dLayer),1.f};
  float32x4_t den{2*theRCurvature,2*theRCurvature,dHitmag*dLayer,1.f};
  auto phis = f_asin05f(num/den);
  phis = phis*dLayer/(rLayer*cosCross);
  auto deltaPhi = std::abs(phis[0]-phis[1]);
  auto deltaPhiOrig = phis[2];
  
  /*
  auto alphaHit = cropped_asin( dHitmag/(2*theRCurvature)); 
  auto OdeltaPhi = std::abs( alphaHit - cropped_asin( dLayer/(2*theRCurvature)));
  OdeltaPhi *= dLayer/(rLayer*cosCross);  
  // compute additional delta phi due to origin radius
  auto OdeltaPhiOrig = cropped_asin( theROrigin * (dHitmag-dLayer) / (dHitmag*dLayer));
  OdeltaPhiOrig *= dLayer/(rLayer*cosCross);
  std::cout << "dphi " << OdeltaPhi<<'/'<<OdeltaPhiOrig << ' ' << deltaPhi<<'/'<<deltaPhiOrig << std::endl;
  */

  // additinal angle due to not perpendicular stright line crossing  (for displaced beam)
  //  double dPhiCrossing = (cosCross > 0.9999) ? 0 : dL *  sqrt(1-sqr(cosCross))/ rLayer;
  Point2D crossing2 = theVtx + dHit.unit()* (dLayer+dL);
  auto phicross2 = f_atan2f(crossing2.y(),crossing2.x());  
  auto phicross1 = f_atan2f(crossing.y(),crossing.x());
  auto dphicross = phicross2-phicross1;
  if (dphicross < -float(M_PI)) dphicross += float(2*M_PI);
  if (dphicross >  float(M_PI)) dphicross -= float(2*M_PI);
  if (dphicross > float(M_PI/2)) dphicross = 0.;  // something wrong?
  phicross2 = phicross1 + dphicross;
        


  // inner hit error taken as constant
  auto deltaPhiHit = theExtraTolerance / rLayer;

  // outer hit error
//   double deltaPhiHitOuter = errRPhi/rLayer; 
  auto deltaPhiHitOuter = errRPhi/hitXY.mag();

  auto margin = deltaPhi+deltaPhiOrig+deltaPhiHit+deltaPhiHitOuter ;

  if (thePrecise) {
    // add multiple scattering correction
    PixelRecoPointRZ zero(0., theVtxZ);
    PixelRecoPointRZ point(hitXY.mag(), hitZ);
    auto scatt = 3.f*sigma(thePtMin,zero, point, ol) / rLayer; 
   
    margin += scatt ;
  }
  
  return PixelRecoRange<float>( std::min(phicross1,phicross2)-margin, 
                                std::max(phicross1,phicross2)+margin);
}




float InnerDeltaPhi::minRadius( float hitR, float hitZ) const 
{
  if (theRDefined) return theRLayer;
  else {
    float rmin = (theA*hitR)/(hitZ-theB);
    return ( rmin> 0) ? std::max( rmin, theRLayer) : theRLayer;
  }
}


