#ifndef SimDataFormats_PCaloHit_H
#define SimDataFormats_PCaloHit_H

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

// Persistent Calorimeter hit

class PCaloHit { 

public: 

  PCaloHit(float e = 0., float t = 0., int i = 0, float emFraction = 1.,
	   uint16_t d = 0) : myEnergy(e), myEMFraction(emFraction), myTime(t),
    myItra(i), myDepth(d) { }

  PCaloHit(unsigned int id, float e = 0., float t = 0., int i = 0, 
	   float emFraction = 1., uint16_t d = 0) : myEnergy (e), 
    myEMFraction(emFraction), myTime (t), myItra (i), detId(id), myDepth(d) { }
  PCaloHit(float eEM, float eHad, float t, int i = 0, uint16_t d = 0);
  PCaloHit(unsigned int id, float eEM, float eHad, float t, int i = 0, 
	   uint16_t d = 0);
  
  //Names
  static const char *name() { return "Hit"; }

  const char * getName() const { return name (); }

  //Energy deposit of the Hit
  double energy()    const { return myEnergy; }
  double energyEM()  const { return myEMFraction*myEnergy; }
  double energyHad() const { return (1.-myEMFraction)*myEnergy; }

  //Time of the deposit
  double time() const { return myTime; }

  //Geant track number
  int geantTrackId() const { return myItra; }

  //DetId where the Hit is recorded
  void setID(unsigned int id) { detId = id; }
  unsigned int  id() const { return detId; }

  //Encoded depth in the detector 
  //for ECAL: # radiation length, 30 == APD
  //for HCAL:
  void setDepth(uint16_t depth) { myDepth = depth; }
  uint16_t depth() const { return myDepth; } 

  //Event Id (for signal/pileup discrimination)

  void setEventId(EncodedEventId e) { theEventId = e; }
  EncodedEventId eventId() const {return theEventId;}

  // new method used by the new transient CF
  void setTime(float t) {myTime=t;}

  //Comparisons

  bool operator<(const PCaloHit &d) const { return myEnergy < d.myEnergy; }

  //Same Hit (by value)
  bool operator==(const PCaloHit &d) const 
  { return (myEnergy == d.myEnergy && detId == d.detId); }


protected: 
  float myEnergy;
  float myEMFraction; 
  float myTime; 
  int   myItra; 
  unsigned int detId; 
  uint16_t myDepth;
  EncodedEventId  theEventId;
}; 

#include<iosfwd>
std::ostream &operator<<(std::ostream &, const PCaloHit &); 

#endif // _SimDataFormats_SimCaloHit_PCaloHit_h_
