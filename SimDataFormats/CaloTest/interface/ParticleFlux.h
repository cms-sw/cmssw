#ifndef SimDataFormatsCaloTestParticleFlux_h
#define SimDataFormatsCaloTestParticleFlux_h

#include <string>
#include <vector>
#include <memory>

#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"

class ParticleFlux {

public:

  ParticleFlux(std::string name="", int id=0) : detName(name), detId(id) {}
  virtual ~ParticleFlux() {}

  struct flux {
    int                pdgId, vxType;
    float              tof;
    math::GlobalPoint  vertex, hitPoint;
    math::GlobalVector momentum;
    flux(int id=0, int typ=0, float t=0) : pdgId(id), vxType(typ), tof(t) {}
  };

  std::string                     getName() const {return detName;}
  int                             getId() const {return detId;}
  unsigned int                    getComponents() const {return fluxVector.size();}
  std::vector<ParticleFlux::flux> getFlux() const {return fluxVector;}
  void                            setName(const std::string nm) {detName = nm;}
  void                            setId(const int id) {detId = id;}
  void                            addFlux(const ParticleFlux::flux f);
  void                            clear();

private:

  std::string       detName;
  int               detId;
  std::vector<flux> fluxVector;

};

#endif
