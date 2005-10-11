#ifndef _CALO_HIT_H
#define _CALO_HIT_H

#include "DataFormats/DetId/interface/DetId.h"
#include<iosfwd>
/**

 \class CaloHit

 Simple class to encapsulate a Geant hit for ORCA.
*/
class CaloHit 
{ 
public: 

  /** 
    \fn CaloHit(float e, float t, int i)

    \brief Default constructor, which
    takes energy(e), time of the hit(t) and track number(i)
  */
  CaloHit(const DetId & id, float e = 0., float t = 0., int i = 0) : 
    id_(id), energy_ (e), time_ (t), theTimeOffset(0.), itra (i) { }

  DetId id() const {return id_;}

  //Energy deposit of the Hit
  float energy() const { return energy_; }

  //Time of the deposit
  float time() const { return time_ + theTimeOffset; }

  //G3 track number

  /** 
    \fn int geantTrackId() const

    \brief Geant 3 track number
  */
  int geantTrackId() const { return itra; }


  //Comaprisons

  /** 
    \fn bool operator<(const CaloHit &) const

    \brief Askes, whether hit energy is smaller than the one provided.

    Useful for sorting in energy.
  */
  bool operator<(const CaloHit &d) const 
  { return energy_ < d.energy_ ? true : false; }

  /// Offset of time of flight, e.g. when assigned to a new bunch crossing
  void offsetTime( float t0) { theTimeOffset = t0;}
  float timeOffset() const { return theTimeOffset;}


protected: 
  DetId id_;
  float energy_; 
  float time_; 
  float theTimeOffset;
  int itra; 
}; 

extern std::ostream &operator<<(std::ostream &, const CaloHit &);



#endif /* !defined(_CALO_HIT_H) */
