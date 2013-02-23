#ifndef MSLayer_H
#define MSLayer_H
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoPointRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoLineRZ.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoRange.h"
#include <iosfwd>

#include "FWCore/Utilities/interface/GCC11Compatibility.h"

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
  MSLayer(const DetLayer* layer, DataX0 dataX0 = DataX0(0) ) dso_hidden;
  MSLayer() { }

  MSLayer(GeomDetEnumerators::Location part, float position, Range range, 
	   float halfThickness = 0., 
	   DataX0 dataX0 = DataX0(0) ) dso_hidden;


  // sequential number to be used in "maps"
  int seqNum() const { return theSeqNum;}
  // void setSeqNum(int sq) { theSeqNum=sq;}


  const Range & range() const  { return theRange; }
 
  const GeomDetEnumerators::Location & face() const  { return theFace; }
  float position() const { return thePosition; }
  float halfThickness() const { return theHalfThickness; }

  float x0(float cotTheta) const dso_hidden;
  float sumX0D(float cotTheta) const dso_hidden; 

  
  bool operator== (const MSLayer &o) const dso_hidden;
  bool operator<  (const MSLayer &o) const dso_hidden;
  
  std::pair<PixelRecoPointRZ,bool> crossing(const PixelRecoLineRZ &line) const  dso_hidden;
  std::pair<PixelRecoPointRZ,bool> crossing(const SimpleLineRZ &line) const  dso_hidden;

 
 float distance2(const PixelRecoPointRZ & point) const  dso_hidden;

private:

  GeomDetEnumerators::Location theFace;
  float thePosition;
  Range theRange;
  float theHalfThickness;
  int theSeqNum;

  DataX0 theX0Data;

  friend struct MSLayersKeeper;
  friend std::ostream& operator<<( std::ostream& s, const MSLayer & l);

};

std::ostream& operator<<( std::ostream& s, const MSLayer & l)  dso_hidden;
std::ostream& operator<<( std::ostream& s, const MSLayer::DataX0 & d)  dso_hidden;
#endif
