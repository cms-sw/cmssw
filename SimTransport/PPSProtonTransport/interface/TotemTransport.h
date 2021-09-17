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
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <unordered_map>
#include <array>

namespace CLHEP {
  class HepRandomEngine;
}

class TotemTransport : public BaseProtonTransport {
public:
  TotemTransport(const edm::ParameterSet& ps);
  ~TotemTransport() override{};
  // look for scattered protons, propagates them, add them to the event
  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) override;

private:
  bool transportProton(const HepMC::GenParticle*);
  LHCOpticsApproximator* ReadParameterization(const std::string&, const std::string&);

  LHCOpticsApproximator* m_aprox_ip_150_r = nullptr;
  LHCOpticsApproximator* m_aprox_ip_150_l = nullptr;

  std::string m_model_ip_150_r_name;
  std::string m_model_ip_150_l_name;

  double m_beampipe_aperture_radius;
  double m_fEtacut;
  double m_fMomentumMin;
};
#endif
