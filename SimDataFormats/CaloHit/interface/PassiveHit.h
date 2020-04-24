#ifndef SimDataFormats_PassiveHit_H
#define SimDataFormats_PassiveHit_H

#include<string>
#include <vector>

// Persistent Hit in passive material

class PassiveHit { 

public: 

 PassiveHit(std::string vname, unsigned int id, float e=0., float t=0., 
	    int it=0) : vname_(vname), id_(id), energy_(e), time_(t), it_(it) {}
 PassiveHit() : vname_(""), id_(0), energy_(0), time_(0), it_(0) {}
  
  //Names
  static const char *name() { return "PassiveHit"; }

  const char * getName() const { return name (); }

  //Energy deposit of the Hit
  double energy()    const { return energy_; }
  void setEnergy(double e) { energy_ = e; }


  //Time of the deposit
  double time() const   { return time_; }
  void setTime(float t) { time_ = t;}

  //Geant track number
  int trackId() const     { return it_; }
  void setTrackId(int it) { it_ = it; }

  //DetId where the Hit is recorded
  void setID(std::string vname, unsigned int id) { vname_ = vname; id_ = id; }
  std::string   vname() const { return vname_; }
  unsigned int  id()    const { return id_; }

  //Comparisons
  bool operator<(const PassiveHit &d) const { return energy_ < d.energy_; }

  //Same Hit (by value)
  bool operator==(const PassiveHit &d) const 
  { return (energy_ == d.energy_ && id_ == d.id_ && vname_ == d.vname_); }
  

protected: 
  std::string  vname_;
  unsigned int id_;
  float        energy_;
  float        time_;
  int          it_; 
}; 

namespace edm {
  typedef std::vector<PassiveHit> PassiveHitContainer;
} // edm

#include<iosfwd>
std::ostream &operator<<(std::ostream &, const PassiveHit &); 

#endif // _SimDataFormats_SimCaloHit_PassiveHit_h_
