#ifndef TOTEM_TRANSPORT
#define TOTEM_TRANSPORT
#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"
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
class TotemTransport : public ProtonTransport {
public:
  TotemTransport(const edm::ParameterSet& ps, bool verbosity);
  TotemTransport();
  ~TotemTransport() override;
  // look for scattered protons, propagates them, add them to the event

  /*!Adds the stable protons from the event \a ev to a beamline*/
  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) override;

private:
  bool transportProton(const HepMC::GenParticle*);
  LHCOpticsApproximator* ReadParameterization(const std::string&, const std::string&);

  edm::ParameterSet m_parameters;
  bool m_verbosity;
  LHCOpticsApproximator* m_aprox_ip_150_r = nullptr;
  LHCOpticsApproximator* m_aprox_ip_150_l = nullptr;
  std::string m_model_root_file_r;
  std::string m_model_root_file_l;
  std::string m_model_ip_150_r_name;
  std::string m_model_ip_150_l_name;
  double m_model_ip_150_r_zmin;
  double m_model_ip_150_r_zmax;
  double m_model_ip_150_l_zmin;
  double m_model_ip_150_l_zmax;

  edm::EDGetTokenT<edm::HepMCProduct> m_protonsToken_;

  // Private data members
  edm::FileInPath m_opticsFileBeam1_, m_opticsFileBeam2_;

  edm::ESHandle<ParticleDataTable> m_pdt;

  double m_beampipe_aperture_radius;
  double m_Zin_;
  double m_Zout_;
  double m_fEtacut;
  double m_fMomentumMin;
};
#endif
