#include "RecoTracker/TkMSParametrization/interface/MSLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/MSLayersKeeper.h"

using namespace GeomDetEnumerators;

template <class T> T sqr( T t) {return t*t;}

//----------------------------------------------------------------------
ostream& operator<<( ostream& s, const MSLayer & l) 
{
  s <<" face: "<<l.face() 
    <<" pos:"<<l.position()<<", "
    <<" range:"<<l.range()<<", "
    <<l.theX0Data;
  return s;
}
//----------------------------------------------------------------------
ostream& operator<<( ostream& s, const MSLayer::DataX0 & d)
{
  if (d.hasX0)  s << "x0="<<d.x0 <<" sumX0D="<<d.sumX0D;  
  else if (d.allLayers)  s << "x0 by MSLayersKeeper"; 
  else  s <<"empty DataX0"; 
  return s; 
}
//----------------------------------------------------------------------
//MP
MSLayer::MSLayer(const DetLayer* layer, DataX0 dataX0)
 : theFace(layer->location()), theX0Data(dataX0) 
{
  const BarrelDetLayer* bl; const ForwardDetLayer * fl;
  theHalfThickness = layer->surface().bounds().thickness()/2;  

  switch (theFace) {
  case barrel : 
    bl = dynamic_cast<const BarrelDetLayer* >(layer);
    thePosition = bl->specificSurface().radius();
    theRange = Range(-bl->surface().bounds().length()/2,
		     bl->surface().bounds().length()/2);
    break;
  case endcap : 
    fl = dynamic_cast<const ForwardDetLayer* >(layer);
    thePosition = fl->position().z(); 
    theRange = Range(fl->specificSurface().innerRadius(),
                     fl->specificSurface().outerRadius());
    break;
  default:
    cout << " ** MSLayer ** unknown part - will not work!" <<endl; 
    break;
  } 
}
//----------------------------------------------------------------------
MSLayer::MSLayer(Location part, float position, Range range, float halfThickness,
    DataX0 dataX0)
  : theFace(part), 
    thePosition(position), 
    theRange(range), 
    theHalfThickness(halfThickness), 
    theX0Data(dataX0)
  { }


//----------------------------------------------------------------------
bool MSLayer::operator== (const MSLayer &o) const
{

  if ( theFace != o.theFace || fabs(thePosition-o.thePosition) > 1.e-3 ) 
    return false;
  else 
    return true;
}

//----------------------------------------------------------------------
pair<PixelRecoPointRZ,bool> MSLayer::crossing( 
    const PixelRecoLineRZ & line) const
{ 
 const float eps = 1.e-5;
  bool  inLayer = true;
  if (theFace==barrel) { 
    float value = line.zAtR(thePosition);
    if (value > theRange.max()) { 
      value = theRange.max()-eps;
      inLayer = false;
    } 
    else if (value < theRange.min() ) { 
      value = theRange.min()+eps;
      inLayer = false;
    }
    return make_pair( PixelRecoPointRZ(thePosition, value), inLayer) ;
  }
  else {
    float value = line.rAtZ(thePosition);
    if (value > theRange.max()) { 
      value = theRange.max()-eps;
      inLayer = false;
    }
    else if (value < theRange.min() ) { 
      value = theRange.min()+eps;
      inLayer = false;
    }
    return make_pair( PixelRecoPointRZ( value, thePosition), inLayer);
  }
}
//----------------------------------------------------------------------
float MSLayer::distance(const PixelRecoPointRZ & point) const
{
  float dr = 0;
  float dz = 0;
  switch(theFace) { 
  case barrel: 
    dr = fabs(point.r()-thePosition);
    if (theRange.inside(point.z())) {
      return (dr < theHalfThickness) ? 0. : dr;
    }
    else {
      dz = point.z() > theRange.max() ?
          point.z()-theRange.max() : theRange.min() - point.z();
    }
    break;
  case endcap:
    dz = fabs(point.z()-thePosition);
    if (theRange.inside(point.r())) {
      return (dz < theHalfThickness) ? 0. : dz;
    }
    else {
      dr = point.r() > theRange.max() ?
          point.r()-theRange.max() : theRange.min()-point.r();
    }
    break;
  }
  return sqrt(sqr(dr)+sqr(dz));
}

//----------------------------------------------------------------------
bool MSLayer::operator< (const MSLayer & o) const
{

  if (theFace==barrel && o.theFace==barrel) 
    return thePosition < o.thePosition;
  else if (theFace==barrel && o.theFace==endcap)
    return thePosition < o.range().max(); 
  else if (theFace==endcap && o.theFace==endcap ) 
    return fabs(thePosition) < fabs(o.thePosition);
  else 
    return range().max() < o.thePosition; 
}
//----------------------------------------------------------------------
float MSLayer::x0(float cotTheta) const
{
  if (theX0Data.hasX0) {
    float sinTheta = 1/sqrt(1+cotTheta*cotTheta);
    switch(theFace) {
    case barrel:  return theX0Data.x0/sinTheta;
    case endcap: return theX0Data.x0/fabs(cotTheta*sinTheta);
    }
  } else if (theX0Data.allLayers) {
    const MSLayer * dataLayer =
       theX0Data.allLayers->layers(cotTheta).findLayer(*this);
    if (dataLayer) return  dataLayer->x0(cotTheta);
  } 
  return 0.;
}

//----------------------------------------------------------------------
float MSLayer::sumX0D(float cotTheta) const
{
if (theX0Data.hasX0) {
    switch(theFace) {
    case barrel:  
      return theX0Data.sumX0D
          *sqrt(sqrt(   (1+cotTheta*cotTheta)
                      / (1+theX0Data.cotTheta*theX0Data.cotTheta)));
        return theX0Data.sumX0D;
    case endcap: 
      return (theX0Data.hasFSlope) ?  
         theX0Data.sumX0D 
             + theX0Data.slopeSumX0D * (1/cotTheta-1/theX0Data.cotTheta)
       : theX0Data.sumX0D;
    }
  } else if (theX0Data.allLayers) {
    const MSLayer* dataLayer =
       theX0Data.allLayers->layers(cotTheta).findLayer(*this);
    if (dataLayer) return  dataLayer->sumX0D(cotTheta);
  }
  return 0.;

}
