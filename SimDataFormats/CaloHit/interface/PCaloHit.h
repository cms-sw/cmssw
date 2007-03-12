#ifndef SimDataFormats_PCaloHit_H
#define SimDataFormats_PCaloHit_H

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

// Persistent Calorimeter hit

class PCaloHit { 

public: 

  PCaloHit(float eEM = 0., float eHad=0., float t = 0., int i = 0);

  PCaloHit(unsigned int id, float eEM = 0., float eHad =0., float t = 0., 
	   int i = 0);

  
  //Names
  static const char *name() { return "Hit"; }

  const char * getName() const { return name (); }

  //Energy deposit of the Hit
  double energy()    const { return myEnergyEM+myEnergyHad; }
  double energyEM()  const { return myEnergyEM; }
  double energyHad() const { return myEnergyHad; }

  //Time of the deposit
  double time() const { return myTime; }

  //Geant track number
  int geantTrackId() const { return myItra; }

  //DetId where the Hit is recorded
  unsigned int  id() const { return detId; }

  //Event Id (for signal/pileup discrimination)

  void setEventId(EncodedEventId e) { theEventId = e; }

  EncodedEventId eventId() const {return theEventId;}


  //Comparisons

  bool operator<(const PCaloHit &d) const 
  { return (myEnergyEM+myEnergyHad) < (d.myEnergyEM+d.myEnergyHad); }

  //Same Hit (by value)
  bool operator==(const PCaloHit &d) const 
  { return ((myEnergyEM+myEnergyHad) == (d.myEnergyEM+d.myEnergyHad) &&
	    detId == d.detId); }


protected: 
  float myEnergyEM;
  float myEnergyHad; 
  float myTime; 
  int   myItra; 
  unsigned int detId; 
  EncodedEventId  theEventId;
}; 

#include<iosfwd>
std::ostream &operator<<(std::ostream &, const PCaloHit &); 

#endif // _SimDataFormats_SimCaloHit_PCaloHit_h_
