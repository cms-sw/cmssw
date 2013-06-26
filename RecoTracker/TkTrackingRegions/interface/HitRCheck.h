#ifndef HitRCheck_H
#define HitRCheck_H

/** provides allowed range of radius R from HitRZConstraint 
    at a given Z coordinate */

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/HitRZConstraint.h"

class HitRCheck GCC11_FINAL : public HitRZCompatibility {
public:
  static constexpr Algo me =rAlgo;

  typedef TkTrackingRegionsMargin<float> Margin;

  HitRCheck()  : HitRZCompatibility(me) { }
  HitRCheck(const HitRZConstraint & rz, Margin margin = Margin(0,0)) 
    :  HitRZCompatibility(me), theRZ(rz), theTolerance(margin) { } 

  virtual bool operator() (const float & r, const float & z) const
    { return range(z).inside(r); }

  inline Range range(const float & z) const;

  virtual HitRCheck * clone() const { return new HitRCheck(*this); }

  void setTolerance(const Margin & tolerance) { theTolerance = tolerance; }

private:
  HitRZConstraint theRZ;
  Margin theTolerance;
};



HitRCheck::Range HitRCheck::range(const float & z) const
{
  constexpr float rBig = 150.; //something above the detector ranges
  const auto & lineLeft =  theRZ.lineLeft();
  const auto & lineRight = theRZ.lineRight();
  
  float rR = lineRight.rAtZ(z);
  float rL = lineLeft.rAtZ(z);
  float rMin = (rR<rL) ? rR : rL;  
  float rMax = (rR<rL) ? rL : rR;
  // in reality all this never happens!
  float aMin = (rMin>0) ? rMin : rMax;
  float aMax = (rMin>0) ? rMax : rBig;
  aMin = (rMax>0) ? aMin : rBig;
  return Range(aMin-theTolerance.left(),aMax+theTolerance.right());

  /* check
  Range v(aMin-theTolerance.left(),aMax+theTolerance.right());
  Range ori;
  if (z > 0.) {
    if (lineRight.cotLine() <= 0.) ori = Range(rBig, 0); //empty
    else {
      float rMin = lineRight.rAtZ(z);
      if (lineLeft.cotLine() <= 0) ori= Range(rMin-theTolerance.left(),rBig);
      else {
	float rMax = lineLeft.rAtZ(z);
	ori = Range(rMin-theTolerance.left(),rMax+theTolerance.right());
      }
    } 
  } else {
    if (lineLeft.cotLine() >= 0.)  ori = Range(rBig, 0); //empty
    else {
      float rMin = lineLeft.rAtZ(z);
      if (lineRight.cotLine()>= 0) ori = Range(rMin-theTolerance.left(),rBig);
      else {
	float rMax = lineRight.rAtZ(z);
	ori= Range(rMin-theTolerance.left(),rMax+theTolerance.right());
      }
    }
  }

  if (ori!=v) 
    std::cout << "v,ori " << v.first << ',' << v.second <<" "  
	      << ori.first << ',' << ori.second << std::endl;
  return ori;
  */

  /*  original code
  if (z > 0.) {
    if (lineRight.cotLine() <= 0.) return Range(rBig, 0); //empty
    float rMin = lineRight.rAtZ(z);
    if (lineLeft.cotLine() <= 0) return Range(rMin-theTolerance.left(),rBig);
    float rMax = lineLeft.rAtZ(z);
    return Range(rMin-theTolerance.left(),rMax+theTolerance.right()); 
  } else {
    if (lineLeft.cotLine() >= 0.) return Range(rBig, 0); //empty 
    float rMin = lineLeft.rAtZ(z);
    if (lineRight.cotLine()>= 0) return Range(rMin-theTolerance.left(),rBig);
    float rMax = lineRight.rAtZ(z);
    return Range(rMin-theTolerance.left(),rMax+theTolerance.right());
  }
  */
}
#endif
