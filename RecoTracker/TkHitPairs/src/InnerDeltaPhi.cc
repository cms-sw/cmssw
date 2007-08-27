#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisation.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include "DataFormats/GeometryVector/interface/Basic2DVector.h"

#include "TVector3.h"
#include "DataFormats/GeometryVector/interface/Phi.h"
#include <iostream>

using namespace std;

#include "RecoTracker/TkSeedGenerator/interface/FastCircle.h"

template <class T> T sqr( T t) {return t*t;}

InnerDeltaPhi:: InnerDeltaPhi( const DetLayer& layer,
                 const TrackingRegion & region,
                 const edm::EventSetup& iSetup,
                 bool precise)
  : theROrigin(region.originRBound()),
    theRLayer(0),
    theA(0),
    theB(0),
    theVtxX(region.origin().x()),theVtxY(region.origin().y()),theVtxZ(region.origin().z()),
    thePtMin(region.ptMin()),
    sigma(0),
    thePrecise(precise)
{
  float zMinOrigin = theVtxZ-region.originZBound();
  float zMaxOrigin = theVtxZ+region.originZBound();
  theRCurvature = PixelRecoUtilities::bendingRadius(thePtMin,iSetup);
 
  sigma = new MultipleScatteringParametrisation(&layer,iSetup);

  if (layer.location() == GeomDetEnumerators::barrel) initBarrelLayer( layer);
  else initForwardLayer( layer, zMinOrigin, zMaxOrigin);

}


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

  ///////////The vertex position
  theVtxX = 0;   theVtxY = 0;
  //phiIP = Geom::Phi<float>(atan2(theVtxY,theVtxX));
  ///////////////////////////////

  if (layer.location() == GeomDetEnumerators::barrel) initBarrelLayer( layer);
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
  theVtxR = 0.0; //Do not change. 
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
  theVtxR = sqrt(pow(theVtxX,2) +  pow(theVtxY,2));
}

float InnerDeltaPhi::operator()( float rHit, float zHit, float errRPhi) const
{
  // alpha - angle between particle direction at vertex and position of hit.
  // (pi/2 - alpha) - angle hit-vertex-cernter_of_curvature
  // cos (pi/2 - alpha) = (hRhi/2) / theRCurvature
  // so:

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
    float scatt = 3*(*sigma)(thePtMin,zero, point) / rMin; 
    float deltaPhiHitOuter = errRPhi/rMin; 
   
    return deltaPhi+deltaPhiOrig+deltaPhiHit + scatt + deltaPhiHitOuter;
  }

}

/////////////////////////////////// AK //////////////////////////////
/////////////////////////////////////////////////////////////////////
float InnerDeltaPhi::innerRadius( float  hitX, float hitY, float hitZ) const
{
// cout <<"--------------------------------"<<endl;
// cout <<"HIT (x,y,z) :"<<" ("<<hitX<<", "<<hitY<<", "<<hitZ<<") "<<endl;
  if (theRDefined) {
//     cout <<" return defined R layer!"<<theRLayer<<endl;
     return theRLayer;
  }
  else {
   float t = theA/(hitZ-theB);
   float layer_X = theVtxX + (hitX-theVtxX)*t;
   float layer_Y = theVtxY + (hitY-theVtxY)*t;

//   cout <<"  t from line:"<<t<<endl;

   float layer_R = sqrt( sqr(layer_X)+ sqr(layer_Y) );
   
   
//   cout << " crossing in TR: "<<" (x,y):"<<layer_X<<","<<layer_Y <<" radius: "<<layer_R<< endl;
   return std::max(layer_R, theRLayer);

//   float hitR = sqrt( sqr(hitX)+sqr(hitY)); 
//   float invRmin = (hitZ-theB)/theA/(hitR-theVtxR);
//        << " result2: " <<1./invRmin+theVtxR << endl;
//   return ( invRmin> 0) ? std::max( 1./invRmin, (double)theRLayer) : theRLayer;
  }
}


PixelRecoRange<float> InnerDeltaPhi::operator()( float rHit, float phiHit, float zHit, float errRPhi)
{


  float xHit = rHit*cos(phiHit);
  float yHit = rHit*sin(phiHit);

  typedef Basic2DVector<double> Point2D;
  //
  // compute crossing of track with layer
  // rVTX - from 0,0 to vertex
  // dHit - from VTX to outer hit
  // rLayer - layer radius
  // dLayer - distance from VTX to inner layer in direction of dHit
  // vect(rLayer) = vect(rVTX) + vect(dHit).unit * dLayer
  //     rLayer^2 = (ect(rVTX) + vect(dHit).unit * dLayer)^2 and we have square eqation for dLayer 
  //
  // barrel case
  //
  float rLayer = innerRadius( xHit, yHit, zHit);
//    cout <<" rLayer = " << rLayer << endl;
    Point2D vtx(theVtxX,theVtxY);
    Point2D hit(xHit,yHit);
    Point2D dHit = hit-vtx;
    double var_c = vtx.mag2()-sqr(rLayer);
    double var_b = 2*vtx.dot(dHit.unit());
    double var_delta = sqr(var_b)-4*var_c;
    if (var_delta <=0.) var_delta = 0;
    double dLayer = (-var_b + sqrt(var_delta))/2.; //only the value along vector is OK. 
    Point2D crossing = vtx+ dHit.unit() * dLayer;
//    cout <<" crossing1 "<<crossing.x()<<", "<<crossing.y()
//         <<" dLayer = "<<dLayer<<" rLayer from crossing: "<<crossing.r()<<" t:"<<dLayer/dHit.mag()
//          <<endl;


  // track is crossing layer with angle such as:
  // this factor should be taken in computation of eror projection
     double cosCross = dHit.unit().dot(crossing.unit());
  
  float dHitv = dHit.mag(); 
  float alphaHit = asin( dHitv/(2*theRCurvature));
  float deltaPhi = fabs( alphaHit - asin( dLayer/(2*theRCurvature)));
        deltaPhi *= (dLayer/rLayer/cosCross);  
        

  // compute additional delta phi due to origin radius
  float deltaPhiOrig = asin( theROrigin * (dHitv-dLayer) / (dHitv*dLayer));
        deltaPhiOrig *= (dLayer/rLayer/cosCross);

  // hit error taken as constant
  float deltaPhiHit = theHitError / rLayer;

  float margin = deltaPhi+deltaPhiOrig+deltaPhiHit;

  if (thePrecise) {
    // add multiple scattering correction
    PixelRecoPointRZ zero(0., theVtxZ);
    PixelRecoPointRZ point(rHit, zHit);
    float scatt = 3*(*sigma)(thePtMin,zero, point) / rLayer; 
    float deltaPhiHitOuter = errRPhi/rLayer; 
   
    margin += deltaPhi+deltaPhiOrig+deltaPhiHit + scatt + deltaPhiHitOuter;
  }
  
  return PixelRecoRange<float>(crossing.phi()-margin, crossing.phi()+margin);
  
    
/*


  TVector3 tkCenter = findTrackCenter(xHit,yHit,-1);
  double phi1 = findPhi(0,0,rLayer, 
			tkCenter.x(),tkCenter.y(),theRCurvature,phiHit);
  
  tkCenter = findTrackCenter(xHit,yHit,1);
  double phi2 = findPhi(0,0,rLayer, 
			tkCenter.x(),tkCenter.y(),theRCurvature,phiHit);


 // hit error taken as constant
  float deltaPhiHit = theHitError / rLayer;
  // compute additional delta phi due to origin radius
  float deltaPhiOrig = asin( theROrigin * (rHit-rLayer) / (rHit*rLayer));

  float delta = deltaPhiOrig+deltaPhiHit;
  thePrecise = true;
  if (thePrecise){
    // add multiple scattering correction
    PixelRecoPointRZ zero(0., theVtxZ);
    PixelRecoPointRZ point(rHit, zHit);
    float scatt = 3*(*sigma)(thePtMin,zero, point) / rLayer; 
    float deltaPhiHitOuter = errRPhi/rLayer;    
    delta += scatt + deltaPhiHitOuter;
  }
  
 PixelRecoRange<float> phiRange(phi2,phi1);
 phiRange.sort();
 PixelRecoRange<float> phiRangeFinal( phiRange.min()-delta/2.0,
				      phiRange.max()+delta/2.0);

   return phiRangeFinal;
*/
}


TVector3 InnerDeltaPhi::findTrackCenter(float xHit, float yHit, int sign)const{

  TVector3 innerHit(xHit,yHit,0);
  TVector3 ip(theVtxX,theVtxY,0);

  sign/=abs(sign);

 TVector3 d = innerHit-ip;
 TVector3 dOrt = d.Orthogonal().Unit();
 double tmp = sqrt(theRCurvature*theRCurvature - d.Mag()*d.Mag()/4.0);
 TVector3 r = ip + d*0.5 + sign*dOrt*tmp;
 
 return r;
}


double InnerDeltaPhi::findPhi(double x0, double y0, double r0, 
			      double x1, double y1, double r1, double phiHit)const{

  double tmp = x0*x0 - x1*x1 + y0*y0 - y1*y1 - r0*r0 + r1*r1;

  double A = 2.0 + 2.0*pow((x0-x1)/(y0-y1),2);
  double B = -2.0*(x0+x1) + 2.0*(x0-x1)/(y0-y1)*(y0+y1)- 2.0*(x0-x1)*tmp/(y0-y1)/(y0-y1);
  double C = x0*x0 + x1*x1+ tmp*tmp/2.0/(y0-y1)/(y0-y1) - tmp/(y0-y1)*(y0+y1) 
    + y0*y0 + y1*y1 - r0*r0 -  r1*r1;

  double sqrtDelta = sqrt(B*B - 4.0*A*C);
 
  double xA = (-B-sqrtDelta)/2.0/A;
  double yA = (-2.0*xA*(x0-x1) + tmp)/2.0/(y0-y1);
  Geom::Phi<float> phiHit1(phiHit);
  Geom::Phi<float> phiA(atan2(yA,xA));
  //
  double delta2 = fabs(phiA - phiHit1);

  //std::cout<<"   A: "<<A<<" B: "<<B<<" C: "<<C<<" delta: "<<(B*B - 4.0*A*C)<<std::endl;
  //std::cout<<"A (x,y,phi): "<<xA<<", "<<yA<<", "<<phiA<<std::endl;
  //std::cout<<"delta2: "<<delta2<<std::endl;


  if(delta2<0.785) return phiA; //Delta phi always less than pi/4;
  
  double xB = (-B+sqrtDelta)/2.0/A;
  double yB = (-2.0*xB*(x0-x1) + tmp)/2.0/(y0-y1);
  Geom::Phi<float> phiB(atan2(yB,xB));
  delta2 = fabs(phiB - phiHit);
  
  //std::cout<<"B (x,y,phi): "<<xB<<", "<<yB<<", "<<phiB<<std::endl;

  if(delta2<0.785)return phiB; //In any case one solution should be good.
  return 0.0;  //If not return 0; Problem.

}
