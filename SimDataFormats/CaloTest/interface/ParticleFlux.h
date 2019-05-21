#ifndef SimDataFormatsCaloTestParticleFlux_h
#define SimDataFormatsCaloTestParticleFlux_h

#include <string>
#include <vector>
#include <memory>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

class ParticleFlux {
public:
  ParticleFlux(std::string name = "", int id = 0) : detName_(name), detId_(id) {}
  virtual ~ParticleFlux() {}

  struct flux {
    int pdgId, vxType;
    float tof;
    math::GlobalPoint vertex, hitPoint;
    math::GlobalVector momentum;
    flux(int id = 0, int typ = 0, float t = 0) : pdgId(id), vxType(typ), tof(t) {}
  };

  std::string const& getName() const { return detName_; }
  int getId() const { return detId_; }
  unsigned int getComponents() const { return fluxVector_.size(); }
  std::vector<ParticleFlux::flux> getFlux() const& { return fluxVector_; }
  void setName(const std::string nm) { detName_ = nm; }
  void setId(const int id) { detId_ = id; }
  void addFlux(const ParticleFlux::flux f);
  void clear();

private:
  std::string detName_;
  int detId_;
  std::vector<flux> fluxVector_;
};

#endif
