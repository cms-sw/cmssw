#ifndef SimDataFormats_CaloHit_MaterialInformation_H
#define SimDataFormats_CaloHit_MaterialInformation_H

#include <string>
#include <vector>

// Persistent information about steps in material

class MaterialInformation {
public:
  MaterialInformation(
      std::string vname, int id = 0, float eta = 0, float phi = 0, float length = 0, float radlen = 0, float intlen = 0)
      : vname_(vname), id_(id), eta_(eta), phi_(phi), length_(length), radlen_(radlen), intlen_(intlen) {}
  MaterialInformation() : vname_(""), id_(0), eta_(0), phi_(0), length_(0), radlen_(0), intlen_(0) {}

  //Names
  static const char *name() { return "MaterialInformation"; }
  const char *getName() const { return name(); }
  std::string vname() const { return vname_; }
  int id() const { return id_; }
  void setID(int i) { id_ = i; }

  //Track eta, phi
  double trackEta() const { return eta_; }
  void setTrackEta(double e) { eta_ = e; }
  double trackPhi() const { return phi_; }
  void setTrackPhi(double f) { phi_ = f; }

  //Lengths
  double stepLength() const { return length_; }
  void setStepLength(double l) { length_ = l; }
  double radiationLength() const { return radlen_; }
  void setRadiationLength(double r) { radlen_ = r; }
  double interactionLength() const { return intlen_; }
  void setInteractionLength(double i) { intlen_ = i; }

protected:
  std::string vname_;
  int id_;
  float eta_;
  float phi_;
  float length_;
  float radlen_;
  float intlen_;
};

namespace edm {
  typedef std::vector<MaterialInformation> MaterialInformationContainer;
}  // namespace edm

#include <iosfwd>
std::ostream &operator<<(std::ostream &, const MaterialInformation &);

#endif  // _SimDataFormats_CaloHit_MaterialInformation_h_
