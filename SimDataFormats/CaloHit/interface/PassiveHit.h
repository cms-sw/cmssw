#ifndef SimDataFormats_PassiveHit_H
#define SimDataFormats_PassiveHit_H

#include <string>
#include <vector>

// Persistent Hit in passive material

class PassiveHit {
public:
  PassiveHit(std::string vname,
             unsigned int id,
             float e = 0,
             float etot = 0,
             float t = 0,
             int it = 0,
             int ip = 0,
             float stepl = 0,
             float xp = 0,
             float yp = 0,
             float zp = 0)
      : vname_(vname),
        id_(id),
        energy_(e),
        etotal_(etot),
        time_(t),
        it_(it),
        ip_(ip),
        stepl_(stepl),
        xp_(xp),
        yp_(yp),
        zp_(zp) {}
  PassiveHit()
      : vname_(""), id_(0), energy_(0), etotal_(0), time_(0), it_(0), ip_(0), stepl_(0), xp_(0), yp_(0), zp_(0) {}

  //Names
  static const char *name() { return "PassiveHit"; }

  const char *getName() const { return name(); }

  //Energy deposit of the Hit
  double energy() const { return energy_; }
  void setEnergy(double e) { energy_ = e; }
  double energyTotal() const { return etotal_; }
  void setEnergyTotal(double e) { etotal_ = e; }

  //Time of the deposit
  double time() const { return time_; }
  void setTime(float t) { time_ = t; }

  //Geant track number
  int trackId() const { return it_; }
  void setTrackId(int it) { it_ = it; }

  //DetId where the Hit is recorded
  void setID(std::string vname, unsigned int id) {
    vname_ = vname;
    id_ = id;
  }
  std::string vname() const { return vname_; }
  unsigned int id() const { return id_; }

  //PDGId of the track causing the Hit
  int pdgId() const { return ip_; }
  void setPDGId(int ip) { ip_ = ip; }

  //Step length for the current Hit
  float stepLength() const { return stepl_; }
  void setStepLength(float stepl) { stepl_ = stepl; }

  //Position of the Hit
  float x() const { return xp_; }
  void setX(float xp) { xp_ = xp; }
  float y() const { return yp_; }
  void setY(float yp) { yp_ = yp; }
  float z() const { return zp_; }
  void setZ(float zp) { zp_ = zp; }

  //Comparisons
  bool operator<(const PassiveHit &d) const { return energy_ < d.energy_; }

  //Same Hit (by value)
  bool operator==(const PassiveHit &d) const { return (energy_ == d.energy_ && id_ == d.id_ && vname_ == d.vname_); }

protected:
  std::string vname_;
  unsigned int id_;
  float energy_;
  float etotal_;
  float time_;
  int it_;
  int ip_;
  float stepl_;
  float xp_;
  float yp_;
  float zp_;
};

namespace edm {
  typedef std::vector<PassiveHit> PassiveHitContainer;
}  // namespace edm

#include <iosfwd>
std::ostream &operator<<(std::ostream &, const PassiveHit &);

#endif  // _SimDataFormats_SimCaloHit_PassiveHit_h_
