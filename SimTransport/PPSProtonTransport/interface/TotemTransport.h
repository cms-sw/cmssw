#ifndef TOTEM_TRANSPORT
#define TOTEM_TRANSPORT
#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"
#include "SimTransport/TotemRPProtonTransportParametrization/interface/LHCOpticsApproximator.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

namespace CLHEP {
  class HepRandomEngine;
}

class TotemTransport : public BaseProtonTransport {
public:
  TotemTransport(const edm::ParameterSet& ps);
  ~TotemTransport() override;
  // look for scattered protons, propagates them, add them to the event
  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) override;

private:
  bool transportProton(HepMC::GenParticle*);
  LHCOpticsApproximator* ReadParameterization(const std::string&, const std::string&);

  LHCOpticsApproximator* m_aprox_ip_150_r_;
  LHCOpticsApproximator* m_aprox_ip_150_l_;

  std::string m_model_ip_150_r_name_;
  std::string m_model_ip_150_l_name_;

  double m_beampipe_aperture_radius_;
};
#endif
