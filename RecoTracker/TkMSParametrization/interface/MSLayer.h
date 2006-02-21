#ifndef MSLayer_H
#define MSLayer_H
#include "TrackingTools/DetLayers/interface/DetLayer.h"
//#include "CommonDet/BasicDet/interface/Enumerators.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include <iostream>

class DetLayer;
class MSLayersKeeper;

class MSLayer {
public:
  typedef PixelRecoRange<float> Range;

  struct DataX0 { 
    DataX0(const MSLayersKeeper *al = 0) 
      : hasX0(false), allLayers(al) { }
    DataX0(float ax0, float asX0D, float aCotTheta) 
      : hasX0(true), hasFSlope(false), x0(ax0), sumX0D(asX0D), 
        cotTheta(aCotTheta), allLayers(0) { }
    void setForwardSumX0DSlope(float aSlope) 
        { hasFSlope= true; slopeSumX0D = aSlope; }
    bool hasX0, hasFSlope;
    float x0, sumX0D, cotTheta, slopeSumX0D;
    const MSLayersKeeper *allLayers; 
  };

public:
  MSLayer(const DetLayer* layer, DataX0 dataX0 = DataX0(0) );
  MSLayer() { }
  //MP  
 /*  MSLayer(Part part, float position, Range range,  */
/* 	  float halfThickness = 0.,  */
/* 	  DataX0 dataX0 = DataX0(0) ); */
  MSLayer(float position, Range range,  
	  float halfThickness = 0., 
	  DataX0 dataX0 = DataX0(0) );
  const Range & range() const  { return theRange; }
  //MP 
  //    const Part & face() const  { return theFace; }
  float position() const { return thePosition; }
  float halfThickness() const { return theHalfThickness; }

  float x0(float cotTheta) const;
  float sumX0D(float cotTheta) const; 

  bool operator== (const MSLayer &o) const;
  bool operator<  (const MSLayer &o) const;
  pair<PixelRecoPointRZ,bool> crossing(
      const PixelRecoLineRZ &line) const;
  float distance(const PixelRecoPointRZ & point) const;

private:
  //MP 
  //Part theFace;
  float thePosition;
  Range theRange;
  float theHalfThickness;
  DataX0 theX0Data;

  friend struct MSLayersKeeper;
  friend ostream& operator<<( ostream& s, const MSLayer & l);

};

ostream& operator<<( ostream& s, const MSLayer & l);
ostream& operator<<( ostream& s, const MSLayer::DataX0 & d);
#endif
