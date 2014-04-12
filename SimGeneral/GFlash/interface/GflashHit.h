#ifndef GflashHit_H
#define GflashHit_H

#include "SimGeneral/GFlash/interface/Gflash3Vector.h"

class GflashHit
{
public:
  
  GflashHit();
  GflashHit(double time, double energy, Gflash3Vector& pos);
  ~GflashHit();
  
  inline double getTime() const { return theTime; }
  inline double getEnergy() const { return theEnergy; }
  inline const Gflash3Vector& getPosition() const { return thePosition; }

  inline void setTime(const double time) { theTime = time; }
  inline void setEnergy(const double energy) { theEnergy = energy; }
  inline void setPosition(const Gflash3Vector& pos) { thePosition = pos; }
    
private:
  double theTime;
  double theEnergy;
  Gflash3Vector thePosition;
};

#endif
