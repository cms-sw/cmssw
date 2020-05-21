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
  fCrossingAngle_45 = iConfig.getParameter<double>("halfCrossingAngleSector45");
  fCrossingAngle_56 = iConfig.getParameter<double>("halfCrossingAngleSector56");
  beamEnergy_ = iConfig.getParameter<double>("BeamEnergy");
  fVtxMeanX = iConfig.getParameter<double>("VtxMeanX");
  fVtxMeanY = iConfig.getParameter<double>("VtxMeanY");
  fVtxMeanZ = iConfig.getParameter<double>("VtxMeanZ");
  m_sigmaSTX = iConfig.getParameter<double>("BeamDivergenceX");
  m_sigmaSTY = iConfig.getParameter<double>("BeamDivergenceY");
  m_sigmaSX = iConfig.getParameter<double>("BeamSigmaX");
  m_sigmaSY = iConfig.getParameter<double>("BeamSigmaY");
  m_sig_E = iConfig.getParameter<double>("BeamEnergyDispersion");
  fBeamXatIP = iConfig.getUntrackedParameter<double>("BeamXatIP", fVtxMeanX);
  fBeamYatIP = iConfig.getUntrackedParameter<double>("BeamYatIP", fVtxMeanY);

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

  px = gpart->momentum().px();
  py = gpart->momentum().py();
  pz = gpart->momentum().pz();
  e = gpart->momentum().e();

  e = sqrt(pow(mass, 2) + pow(px, 2) + pow(py, 2) + pow(pz, 2));

  int direction = (pz > 0) ? 1 : -1;

  // Apply Beam and Crossing Angle Corrections
  TLorentzVector p_out(px, py, pz, e);
  PPSTools::LorentzBoost(p_out,
                         "LAB",
                         {fCrossingAngle_56,  //Beam1
                          fCrossingAngle_45,  //Beam2
                          beamMomentum_,
                          beamEnergy_});

  ApplyBeamCorrection(p_out);

  // from mm to cm
  double XforPosition = gpart->production_vertex()->position().x() / cm;  //cm
  double YforPosition = gpart->production_vertex()->position().y() / cm;  //cm
  double ZforPosition = gpart->production_vertex()->position().z() / cm;  //cm

  H_BeamParticle h_p(mass, charge);
  h_p.set4Momentum(-direction * p_out.Px(), p_out.Py(), fabs(p_out.Pz()), p_out.E());
  // shift the beam position to the given beam position at IP (in cm)
  XforPosition = (XforPosition - fVtxMeanX) + fBeamXatIP * mm_to_cm;
  YforPosition = (YforPosition - fVtxMeanY) + fBeamYatIP * mm_to_cm;
  //ZforPosition stays the same, move the closed orbit only in the X,Y plane
  //
  // shift the starting position of the track to Z=0 if configured so (all the variables below are still in cm)
  if (bApplyZShift) {
    double fCrossingAngle = (p_out.Pz() > 0) ? fCrossingAngle_45 : -fCrossingAngle_56;
    XforPosition = XforPosition +
                   (tan((long double)fCrossingAngle * urad) - ((long double)p_out.Px()) / ((long double)p_out.Pz())) *
                       ZforPosition;
    YforPosition = YforPosition - ((long double)p_out.Py()) / ((long double)p_out.Pz()) * ZforPosition;
    ZforPosition = 0.;
  }

  //
  // set position, but do not invert the coordinate for the angles (TX,TY) because it is done by set4Momentum
  h_p.setPosition(-direction * XforPosition * cm_to_um,
                  YforPosition * cm_to_um,
                  h_p.getTX(),
                  h_p.getTY(),
                  -direction * ZforPosition * cm_to_m);
  bool is_stop;
  float x1_ctpps;
  float y1_ctpps;

  H_BeamLine* _beamline = nullptr;
  double _targetZ = 0;
  switch (direction) {
    case -1:
      _beamline = &*m_beamline56;  // negative side propagation
      _targetZ = fPPSRegionStart_56;
      break;
    case 1:
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

  p_out = PPSTools::HectorParticle2LorentzVector(h_p, direction);

  p_out.SetPx(direction * p_out.Px());
  x1_ctpps = direction * h_p.getX() * um_to_mm;
  y1_ctpps = h_p.getY() * um_to_mm;

  if (verbosity_)
    LogDebug("HectorTransportEventProcessing")
        << "HectorTransport:filterPPS: barcode = " << line << " x=  " << x1_ctpps << " y= " << y1_ctpps;

  m_beamPart[line] = p_out;
  m_xAtTrPoint[line] = x1_ctpps;
  m_yAtTrPoint[line] = y1_ctpps;
  return true;
}
bool HectorTransport::setBeamLine() {
  edm::FileInPath b1(beam1Filename_.c_str());
  edm::FileInPath b2(beam2Filename_.c_str());

  // construct beam line for PPS (forward 1 backward 2):
  if (fPPSBeamLineLength_ > 0.) {
    m_beamline45 = std::make_unique<H_BeamLine>(-1, fPPSBeamLineLength_ + 0.1);  // (direction, length)
    m_beamline45->fill(b2.fullPath(), 1, "IP5");
    m_beamline56 = std::make_unique<H_BeamLine>(1, fPPSBeamLineLength_ + 0.1);  //
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
