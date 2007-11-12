#ifndef SimTransport_HectorGenParticle
#define SimTransport_HectorGenParticle

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

class HectorGenParticle : public HepMC::GenParticle {
 public:
  //  HectorGenParticle();
  HectorGenParticle( const HepMC::GenParticle &eventParticle ) : HepMC::GenParticle( eventParticle ) {};
  virtual ~HectorGenParticle() {};

  double px() const;
  double py() const;
  double pz() const;
  double pt() const;
  double e() const;

  double x() const;
  double y() const;
  double z() const;
};
#endif
