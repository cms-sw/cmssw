#ifndef HECTOR_TRANSPORT
#define HECTOR_TRANSPORT
#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include <cmath>
#include <iomanip>
#include <cstdlib>

// HepMC headers
#include "HepMC/GenEvent.h"
#include "HepMC/GenVertex.h"
#include "HepMC/GenParticle.h"
#include "HepMC/SimpleVector.h"

// user include files
#include <string>

namespace CLHEP {
  class HepRandomEngine;
}

class H_BeamParticle;
class H_BeamLine;

class HectorTransport : public BaseProtonTransport {
public:
  HectorTransport(const edm::ParameterSet& ps);
  ~HectorTransport() override;

  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) override;

private:
  bool m_verbosity;
  bool produceHitsRelativeToBeam_;

  static constexpr double fPPSBeamLineLength_ = 250.;  // default beam line length

  //!propagate the particles through a beamline to PPS
  bool transportProton(const HepMC::GenParticle*);

  // New function to calculate the LorentzBoost

  bool setBeamLine();
  // Defaults

  double m_fEtacut;
  double m_fMomentumMin;

  // PPSHector
  std::unique_ptr<H_BeamLine> m_beamline45;
  std::unique_ptr<H_BeamLine> m_beamline56;
};
#endif
