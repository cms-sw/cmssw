#ifndef HECTOR_TRANSPORT
#define HECTOR_TRANSPORT
#include "SimTransport/PPSProtonTransport/interface/BaseProtonTransport.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/CTPPSBeamParametersRcd.h"
#include "CondFormats/PPSObjects/interface/CTPPSBeamParameters.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
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
  HectorTransport(const edm::ParameterSet& ps, edm::ConsumesCollector iC);
  ~HectorTransport() override;

  void process(const HepMC::GenEvent* ev, const edm::EventSetup& es, CLHEP::HepRandomEngine* engine) override;

private:
  static constexpr double fPPSBeamLineLength_ = 250.;  // default beam line length

  //!propagate the particles through a beamline to PPS
  bool transportProton(const HepMC::GenParticle*);

  // function to calculate the LorentzBoost
  bool setBeamLine();

  // PPSHector
  std::unique_ptr<H_BeamLine> m_beamline45;
  std::unique_ptr<H_BeamLine> m_beamline56;

  edm::ESGetToken<CTPPSBeamParameters, CTPPSBeamParametersRcd> beamParametersToken_;
  edm::ESGetToken<SimBeamSpotObjects, SimBeamSpotObjectsRcd> beamspotToken_;

  const CTPPSBeamParameters* beamParameters_{nullptr};
  const SimBeamSpotObjects* beamspot_{nullptr};
};
#endif
