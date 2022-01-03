#include "SimTransport/PPSProtonTransport/interface/HectorTransport.h"
#include "Utilities/PPS/interface/PPSUtilities.h"
#include "TLorentzVector.h"
//Hector headers
#include "H_BeamLine.h"
#include "H_BeamParticle.h"
#include <memory>

#include <string>

HectorTransport::HectorTransport(const edm::ParameterSet& iConfig, edm::ConsumesCollector iC)
    : BaseProtonTransport(iConfig),
      m_beamline45(nullptr),
      m_beamline56(nullptr),
      beamParametersToken_(iC.esConsumes()),
      beamspotToken_(iC.esConsumes()) {
  MODE = TransportMode::HECTOR;
  std::string s1 = iConfig.getParameter<std::string>("Beam1Filename");
  std::string s2 = iConfig.getParameter<std::string>("Beam2Filename");
  setBeamFileNames(s1, s2);
  double cax45 = iConfig.getParameter<double>("halfCrossingAngleXSector45");
  double cax56 = iConfig.getParameter<double>("halfCrossingAngleXSector56");
  double cay45 = iConfig.getParameter<double>("halfCrossingAngleYSector45");
  double cay56 = iConfig.getParameter<double>("halfCrossingAngleYSector56");
  setCrossingAngles(cax45, cax56, cay45, cay56);
  double stx = iConfig.getParameter<double>("BeamDivergenceX");
  double sty = iConfig.getParameter<double>("BeamDivergenceY");
  double sx = iConfig.getParameter<double>("BeamSigmaX");
  double sy = iConfig.getParameter<double>("BeamSigmaY");
  double se = iConfig.getParameter<double>("BeamEnergyDispersion");
  setBeamParameters(stx, sty, sx, sy, se);
  //PPS
  edm::LogVerbatim("ProtonTransport")
      << "=============================================================================\n"
      << "             Bulding LHC Proton transporter based on HECTOR model\n"
      << "=============================================================================\n";
  setBeamLine();
}
HectorTransport::~HectorTransport() {}
//
// this method is the same for all propagator, but since transportProton is different for each derived class
// it needes to be overriden
//
void HectorTransport::process(const HepMC::GenEvent* evt,
                              const edm::EventSetup& iSetup,
                              CLHEP::HepRandomEngine* engine) {
  clear();
  engine_ = engine;  // the engine needs to be updated for each event

  beamspot_ = &iSetup.getData(beamspotToken_);
  beamParameters_ = &iSetup.getData(beamParametersToken_);

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
  edm::LogVerbatim("ProtonTransport") << "Starting proton transport using HECTOR method\n";

  double px, py, pz, e;
  unsigned int line = (gpart)->barcode();

  double mass = gpart->generatedMass();
  double charge = 1.;

  // remove the CMS vertex shift
  double vtxXoffset;
  double vtxYoffset;
  if (useBeamPositionFromLHCInfo_) {
    vtxXoffset = beamParameters_->getVtxOffsetX45() * cm_to_mm;
    vtxYoffset = beamParameters_->getVtxOffsetY45() * cm_to_mm;
  } else {
    vtxXoffset = beamspot_->x() * cm_to_mm;
    vtxYoffset = beamspot_->y() * cm_to_mm;
  }

  // Momentum in LHC ref. frame
  px = -gpart->momentum().px();
  py = gpart->momentum().py();
  pz = -gpart->momentum().pz();
  e = gpart->momentum().e();

  int direction = (pz > 0) ? 1 : -1;  // in relation to LHC ref frame

  double XforPosition = -gpart->production_vertex()->position().x();  //mm in the ref. frame of LHC
  double YforPosition = gpart->production_vertex()->position().y();   //mm
  double ZforPosition = -gpart->production_vertex()->position().z();  //mm
  double fCrossingAngleX = (pz < 0) ? fCrossingAngleX_45_ : fCrossingAngleX_56_;
  double fCrossingAngleY = (pz < 0) ? fCrossingAngleY_45_ : fCrossingAngleY_56_;
  //
  H_BeamParticle h_p(mass, charge);
  h_p.set4Momentum(px, py, pz, e);
  //
  // shift the starting position of the track to Z=0 if configured so (all the variables below are still in cm)
  XforPosition = XforPosition - ZforPosition * (px / pz + fCrossingAngleX * urad);  // theta ~=tan(theta)
  YforPosition = YforPosition - ZforPosition * (py / pz + fCrossingAngleY * urad);
  XforPosition -= (-vtxXoffset);  // X was already in the LHC ref. frame
  YforPosition -= vtxYoffset;

  // set position, but do not invert the coordinate for the angles (TX,TY) because it is done by set4Momentum
  h_p.setPosition(XforPosition * mm_to_um, YforPosition * mm_to_um, h_p.getTX(), h_p.getTY(), 0.);

  bool is_stop;
  float x1_ctpps;
  float y1_ctpps;

  H_BeamLine* beamline = nullptr;
  double targetZ = 0;
  switch (direction) {
    case 1:
      beamline = &*m_beamline56;  // negative side propagation
      targetZ = fPPSRegionStart_56_;
      break;
    case -1:
      beamline = &*m_beamline45;
      targetZ = fPPSRegionStart_45_;
      break;
  }
  // insert protection for NULL beamlines here
  h_p.computePath(&*beamline);
  is_stop = h_p.stopped(&*beamline);
  if (verbosity_)
    LogDebug("HectorTransportEventProcessing")
        << "HectorTransport:filterPPS: barcode = " << line << " is_stop=  " << is_stop;
  if (is_stop) {
    return false;
  }
  //
  //propagating
  //
  h_p.propagate(targetZ);
  x1_ctpps = h_p.getX();
  y1_ctpps = h_p.getY();

  double thx = h_p.getTX();
  double thy = h_p.getTY();

  if (produceHitsRelativeToBeam_) {
    H_BeamParticle p_beam(mass, charge);
    p_beam.set4Momentum(0., 0., beamMomentum(), beamEnergy());
    p_beam.setPosition(0., 0., fCrossingAngleX * urad, fCrossingAngleY * urad, 0.);
    p_beam.computePath(&*beamline);
    thx -= p_beam.getTX();
    thy -= p_beam.getTY();
    x1_ctpps -= p_beam.getX();
    y1_ctpps -= p_beam.getY();
    edm::LogVerbatim("HectorTransportEventProcessing")
        << "Shifting the hit relative to beam :  X = " << p_beam.getX() << "(mm)      Y = " << p_beam.getY() << "(mm)";
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
    m_beamline45 = std::make_unique<H_BeamLine>(-1, fPPSBeamLineLength_ + 0.1);
    m_beamline45->fill(b2.fullPath(), 1, "IP5");
    m_beamline56 = std::make_unique<H_BeamLine>(1, fPPSBeamLineLength_ + 0.1);
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
                                         << " PPS Region Start 44   =  " << fPPSRegionStart_45_ << "\n"
                                         << " PPS Region Start 56   =  " << fPPSRegionStart_56_ << "\n"
                                         << "===================================================================\n";
    edm::LogVerbatim("HectorTransportSetup") << "===================================================================\n"
                                             << "                  Forward beam line elements \n";
    m_beamline45->showElements();
    edm::LogVerbatim("HectorTransportSetup") << "===================================================================\n"
                                             << "                 Backward beam line elements \n";
    m_beamline56->showElements();
  }
  return true;
}
