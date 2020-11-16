#include "SimTransport/PPSProtonTransport/interface/HectorTransport.h"
#include "Utilities/PPS/interface/PPSUtilities.h"
#include <CLHEP/Random/RandGauss.h>
#include "TLorentzVector.h"
//Hector headers
#include "H_BeamLine.h"
#include "H_BeamParticle.h"
#include <memory>

#include <string>

HectorTransport::~HectorTransport(){};

HectorTransport::HectorTransport(const edm::ParameterSet& iConfig)
    : BaseProtonTransport(iConfig), m_beamline45(nullptr), m_beamline56(nullptr) {
  // Create LHC beam line
  MODE = TransportMode::HECTOR;  // defines the MODE for the transport
  beam1Filename_ = iConfig.getParameter<std::string>("Beam1Filename");
  beam2Filename_ = iConfig.getParameter<std::string>("Beam2Filename");
  fCrossingAngleX_45 = iConfig.getParameter<double>("halfCrossingAngleXSector45");
  fCrossingAngleX_56 = iConfig.getParameter<double>("halfCrossingAngleXSector56");
  fCrossingAngleY_45 = iConfig.getParameter<double>("halfCrossingAngleYSector45");
  fCrossingAngleY_56 = iConfig.getParameter<double>("halfCrossingAngleYSector56");
  beamEnergy_ = iConfig.getParameter<double>("BeamEnergy");
  m_sigmaSTX = iConfig.getParameter<double>("BeamDivergenceX");
  m_sigmaSTY = iConfig.getParameter<double>("BeamDivergenceY");
  m_sigmaSX = iConfig.getParameter<double>("BeamSigmaX");
  m_sigmaSY = iConfig.getParameter<double>("BeamSigmaY");
  m_sig_E = iConfig.getParameter<double>("BeamEnergyDispersion");
  fBeamXatIP = iConfig.getUntrackedParameter<double>("BeamXatIP", 0.);
  fBeamYatIP = iConfig.getUntrackedParameter<double>("BeamYatIP", 0.);
  produceHitsRelativeToBeam_ = iConfig.getParameter<bool>("produceHitsRelativeToBeam");

  //PPS
  edm::LogInfo("ProtonTransport") << "=============================================================================\n"
                                  << "             Bulding LHC Proton transporter based on HECTOR model\n"
                                  << "=============================================================================\n";
  setBeamLine();
}
//
// this method is the same for all propagator, but since transportProton is different for each derived class
// it needes to be overriden
//
void HectorTransport::process(const HepMC::GenEvent* evt,
                              const edm::EventSetup& iSetup,
                              CLHEP::HepRandomEngine* _engine) {
  this->clear();

  engine_ = _engine;  // the engine needs to be updated for each event

  for (HepMC::GenEvent::particle_const_iterator eventParticle = evt->particles_begin();
       eventParticle != evt->particles_end();
       ++eventParticle) {
    if (!((*eventParticle)->status() == 1 && (*eventParticle)->pdg_id() == 2212))
      continue;

    if (!(fabs((*eventParticle)->momentum().eta()) > etaCut_ && fabs((*eventParticle)->momentum().pz()) > momentumCut_))
      continue;  // discard protons outside kinematic acceptance

    unsigned int line = (*eventParticle)->barcode();
    HepMC::GenParticle* gpart = (*eventParticle);

    if (gpart->pdg_id() != 2212)
      continue;  // only transport stable protons
    if (gpart->status() != 1)
      continue;
    if (m_beamPart.find(line) != m_beamPart.end())
      continue;  // assures this protons has not been already propagated

    transportProton(gpart);
  }
}
bool HectorTransport::transportProton(const HepMC::GenParticle* gpart) {
  edm::LogInfo("ProtonTransport") << "Starting proton transport using HECTOR method\n";

  double px, py, pz, e;
  unsigned int line = (gpart)->barcode();

  double mass = gpart->generatedMass();
  double charge = 1.;

  // Momentum in LHC ref. frame
  px = -gpart->momentum().px();
  py = gpart->momentum().py();
  pz = -gpart->momentum().pz();
  e = gpart->momentum().e();

  int direction = (pz > 0) ? 1 : -1;  // in relation to LHC ref frame

  double XforPosition = -gpart->production_vertex()->position().x();  //mm in the ref. frame of LHC
  double YforPosition = gpart->production_vertex()->position().y();   //mm
  double ZforPosition = -gpart->production_vertex()->position().z();  //mm
  double fCrossingAngleX = (pz < 0) ? fCrossingAngleX_45 : fCrossingAngleX_56;
  double fCrossingAngleY = (pz < 0) ? fCrossingAngleY_45 : fCrossingAngleY_56;

  H_BeamParticle h_p(mass, charge);
  h_p.set4Momentum(px, py, pz, e);
  //
  // shift the starting position of the track to Z=0 if configured so (all the variables below are still in cm)
  XforPosition = XforPosition - ZforPosition * (px / pz + fCrossingAngleX * urad);  // theta ~=tan(theta)
  YforPosition = YforPosition - ZforPosition * (py / pz + fCrossingAngleY * urad);
  ZforPosition = 0.;

  // set position, but do not invert the coordinate for the angles (TX,TY) because it is done by set4Momentum
  h_p.setPosition(XforPosition * mm_to_um, YforPosition * mm_to_um, h_p.getTX(), h_p.getTY(), 0.);

  bool is_stop;
  float x1_ctpps;
  float y1_ctpps;

  H_BeamLine* _beamline = nullptr;
  double _targetZ = 0;
  switch (direction) {
    case 1:
      _beamline = &*m_beamline56;  // negative side propagation
      _targetZ = fPPSRegionStart_56;
      break;
    case -1:
      _beamline = &*m_beamline45;
      _targetZ = fPPSRegionStart_45;
      break;
  }
  // insert protection for NULL beamlines here
  h_p.computePath(&*_beamline);
  is_stop = h_p.stopped(&*_beamline);
  if (verbosity_)
    LogDebug("HectorTransportEventProcessing")
        << "HectorTransport:filterPPS: barcode = " << line << " is_stop=  " << is_stop;
  if (is_stop) {
    return false;
  }
  //
  //propagating
  //
  h_p.propagate(_targetZ);
  x1_ctpps = h_p.getX();
  y1_ctpps = h_p.getY();

  double thx = h_p.getTX();
  double thy = h_p.getTY();

  if (produceHitsRelativeToBeam_) {
    H_BeamParticle p_beam(mass, charge);
    p_beam.set4Momentum(0., 0., beamMomentum_, beamEnergy_);
    p_beam.setPosition(0., 0., fCrossingAngleX * urad, fCrossingAngleY * urad, 0.);
    p_beam.computePath(&*_beamline);
    thx -= p_beam.getTX();
    thy -= p_beam.getTY();
    x1_ctpps -= p_beam.getX();
    y1_ctpps -= p_beam.getY();
  }

  double partP = sqrt(pow(h_p.getE(), 2) - ProtonMassSQ);
  double theta = sqrt(thx * thx + thy * thy) * urad;

  // copy the kinematic changing to CMS ref. frame, only the negative Pz needs to be changed
  TLorentzVector p_out(-tan(thx * urad) * partP * cos(theta),
                       tan(thy * urad) * partP * cos(theta),
                       -direction * partP * cos(theta),
                       h_p.getE());

  m_beamPart[line] = p_out;
  m_xAtTrPoint[line] = -x1_ctpps * um_to_mm;  // move to CMS ref. frame
  m_yAtTrPoint[line] = y1_ctpps * um_to_mm;
  if (verbosity_)
    LogDebug("HectorTransportEventProcessing")
        << "HectorTransport:filterPPS: barcode = " << line << " x=  " << x1_ctpps << " y= " << y1_ctpps;

  return true;
}
bool HectorTransport::setBeamLine() {
  edm::FileInPath b1(beam1Filename_.c_str());
  edm::FileInPath b2(beam2Filename_.c_str());

  // construct beam line for PPS (forward 1 backward 2):
  if (fPPSBeamLineLength_ > 0.) {
    m_beamline45 = std::make_unique<H_BeamLine>(
        -1, fPPSBeamLineLength_ + 0.1);  // it is needed to move too  (direction, length, beamEnergy_)
    m_beamline45->fill(b2.fullPath(), 1, "IP5");
    m_beamline56 = std::make_unique<H_BeamLine>(
        1, fPPSBeamLineLength_ + 0.1);  // the same as above, it requires a change in HECTOR
    m_beamline56->fill(b1.fullPath(), 1, "IP5");
  } else {
    if (verbosity_)
      LogDebug("HectorTransportSetup") << "HectorTransport: WARNING: lengthctpps=  " << fPPSBeamLineLength_;
    return false;
  }
  if (verbosity_) {
    edm::LogInfo("HectorTransportSetup") << "===================================================================\n"
                                         << " * * * * * * * * * * * * * * * * * * * * * * * * * * * *           \n"
                                         << " *                                                         *       \n"
                                         << " *                   --<--<--  A fast simulator --<--<--     *     \n"
                                         << " *                 | --<--<--     of particle   --<--<--     *     \n"
                                         << " *  ----HECTOR----<                                          *     \n"
                                         << " *                 | -->-->-- transport through-->-->--      *     \n"
                                         << " *                   -->-->-- generic beamlines -->-->--     *     \n"
                                         << " *                                                           *     \n"
                                         << " * JINST 2:P09005 (2007)                                     *     \n"
                                         << " *      X Rouby, J de Favereau, K Piotrzkowski (CP3)         *     \n"
                                         << " *       http://www.fynu.ucl.ac.be/hector.html               *     \n"
                                         << " *                                                           *     \n"
                                         << " * Center for Cosmology, Particle Physics and Phenomenology  *     \n"
                                         << " *              Universite catholique de Louvain             *     \n"
                                         << " *                 Louvain-la-Neuve, Belgium                 *     \n"
                                         << " *                                                         *       \n"
                                         << " * * * * * * * * * * * * * * * * * * * * * * * * * * * *           \n"
                                         << " HectorTransport configuration: \n"
                                         << " Beam line length      = " << fPPSBeamLineLength_ << "\n"
                                         << " PPS Region Start 44   =  " << fPPSRegionStart_45 << "\n"
                                         << " PPS Region Start 56   =  " << fPPSRegionStart_56 << "\n"
                                         << "===================================================================\n";
  }
  if (verbosity_) {
    edm::LogInfo("HectorTransportSetup") << "====================================================================\n"
                                         << "                  Forward beam line elements \n";
    m_beamline45->showElements();
    edm::LogInfo("HectorTransportSetup") << "====================================================================\n"
                                         << "                 Backward beam line elements \n";
    m_beamline56->showElements();
  }
  return true;
}
