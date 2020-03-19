#ifndef HECTOR_TRANSPORT
#define HECTOR_TRANSPORT
#include "SimTransport/PPSProtonTransport/interface/ProtonTransport.h"

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

class HectorTransport : public ProtonTransport {
public:
  HectorTransport();
  HectorTransport(const edm::ParameterSet &ps, bool verbosity);
  ~HectorTransport() override;
  void process(const HepMC::GenEvent *, const edm::EventSetup &, CLHEP::HepRandomEngine *) override;

private:
  //!propagate the particles through a beamline to PPS
  bool transportProton(const HepMC::GenParticle *);
  /*!Adds the stable protons from the event \a ev to a beamline*/

  //!Clears BeamParticle, prepares PPSHector for a next Aperture check or/and a next event
  void genProtonsLoop(const HepMC::GenEvent *, const edm::EventSetup &);

  // New function to calculate the LorentzBoost
  void setBeamEnergy(double e) {
    fBeamEnergy = e;
    fBeamMomentum = sqrt(fBeamEnergy * fBeamEnergy - ProtonMassSQ);
  }

  double getBeamEnergy() { return fBeamEnergy; }

  double getBeamMomentum() { return fBeamMomentum; }

  bool setBeamLine();
  /*
 *
 *                        ATTENTION:  DATA MEMBERS AND FUNCTIONS COMMON TO BOTH METHODS SHOULD BE MOVED TO THE BASE CLASS
 *
 */
  // Defaults
  edm::ESHandle<ParticleDataTable> m_pdt;

  double m_fEtacut;
  double m_fMomentumMin;

  double m_lengthctpps;
  double m_f_ctpps_f;
  double m_b_ctpps_b;

  // PPSHector
  std::unique_ptr<H_BeamLine> m_beamline45;
  std::unique_ptr<H_BeamLine> m_beamline56;

  std::string m_beam1filename;
  std::string m_beam2filename;
};
#endif
