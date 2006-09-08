#ifndef SimDataFormats_PCaloHit_H
#define SimDataFormats_PCaloHit_H

#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"

// Persistent Calorimeter hit

  class PCaloHit 
  { 
  public: 

    PCaloHit(float e = 0., float t = 0., int i = 0) : 
      myEnergy (e), myTime (t), myItra (i) { }

    PCaloHit(unsigned int id, float e = 0., float t = 0., int i = 0) : 
      myEnergy (e), myTime (t), myItra (i), detId(id) { }


    //Names
    static const char *name() { return "Hit"; }

    const char * getName() const { return name (); }

    //Energy deposit of the Hit
    double energy() const { return myEnergy; }

    //Time of the deposit
    double time() const { return myTime; }

    //G3 track number
    int geantTrackId() const { return myItra; }

    //DetId where the Hit is recorded
    unsigned int  id() const { return detId; }

    //Event Id (for signal/pileup discrimination)

    void setEventId(EncodedEventId e) { theEventId = e; }

    EncodedEventId eventId() const {return theEventId;}


    //Comparisons

    bool operator<(const PCaloHit &d) const 
    { return myEnergy < d.myEnergy; }

    //Same Hit (by value)

    bool operator==(const PCaloHit &d) const 
    { return myEnergy == d.myEnergy && detId == d.detId; }



  protected: 
    float myEnergy; 
    float myTime; 
    int myItra; 
    unsigned int detId; 
    EncodedEventId  theEventId;
  }; 

  #include<iosfwd>
  std::ostream &operator<<(std::ostream &, const PCaloHit &); 

#endif // _SimDataFormats_SimCaloHit_PCaloHit_h_
