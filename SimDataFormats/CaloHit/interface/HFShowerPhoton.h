#ifndef SimDataFormats_HFShowerPhoton_H
#define SimDataFormats_HFShowerPhoton_H
///////////////////////////////////////////////////////////////////////////////
// File: HFShowerPhoton.h
// Photons which will generate single photo electron as in HFShowerLibrary
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

class HFShowerPhoton {

public:

  HFShowerPhoton(float x=0, float y=0, float z=0, float lambda=0, float t=0);
  HFShowerPhoton(const HFShowerPhoton&);
  virtual ~HFShowerPhoton();

  float       x()      const {return x_;}
  float       y()      const {return y_;}
  float       z()      const {return z_;}
  float       lambda() const {return lambda_;}
  float       t()      const {return time_;}
 
private:

  float       x_, y_, z_;
  float       lambda_;
  float       time_;

};

std::ostream& operator<<(std::ostream&, const HFShowerPhoton&);
#endif
