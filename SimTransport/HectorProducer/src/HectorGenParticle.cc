#include "SimTransport/HectorProducer/interface/HectorGenParticle.h"

double HectorGenParticle::px() const {
  return  HepMC::GenParticle::momentum().px();
}

double HectorGenParticle::py() const {
  return  HepMC::GenParticle::momentum().py();
}

double HectorGenParticle::pz() const {
  return  HepMC::GenParticle::momentum().pz();
}

double HectorGenParticle::e() const {
  return  HepMC::GenParticle::momentum().e();
}

double HectorGenParticle::pt() const {
  return  sqrt( ( HectorGenParticle::px() * HectorGenParticle::px() ) + ( HectorGenParticle::py() * HectorGenParticle::py() ) );
}

double HectorGenParticle::x() const {
  if (  HepMC::GenParticle::production_vertex() ) return  HepMC::GenParticle::production_vertex()->position().x();
  else return 0;
}

double HectorGenParticle::y() const {
  if (  HepMC::GenParticle::production_vertex() ) return  HepMC::GenParticle::production_vertex()->position().y();
  else return 0;
}

double HectorGenParticle::z() const {
  if (  HepMC::GenParticle::production_vertex() ) return  HepMC::GenParticle::production_vertex()->position().z();
  else return 0;
}
