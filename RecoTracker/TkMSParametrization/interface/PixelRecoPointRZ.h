#ifndef PixelRecoPointRZ_H
#define PixelRecoPointRZ_H

/** Utility to group two floats into r-z coordinates */

class PixelRecoPointRZ {
public:
  PixelRecoPointRZ() { }
  PixelRecoPointRZ(float r,float z) : theR(r), theZ(z) { }
  float r() const { return theR; }
  float z() const { return theZ; }
private:
  float theR, theZ;
};
#endif
