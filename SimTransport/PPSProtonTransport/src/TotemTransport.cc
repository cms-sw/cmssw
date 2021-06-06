#include "SimTransport/PPSProtonTransport/interface/TotemTransport.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <CLHEP/Units/GlobalSystemOfUnits.h>
#include <CLHEP/Random/RandGauss.h>
#include "TLorentzVector.h"
#include "TFile.h"

#include <cmath>

TotemTransport::TotemTransport(const edm::ParameterSet& iConfig)
    : BaseProtonTransport(iConfig),
      m_model_ip_150_r_name(iConfig.getParameter<std::string>("Model_IP_150_R_Name")),
      m_model_ip_150_l_name(iConfig.getParameter<std::string>("Model_IP_150_L_Name")),
      m_beampipe_aperture_radius(iConfig.getParameter<double>("BeampipeApertureRadius")) {
  MODE = TransportMode::TOTEM;
  beam1Filename_ = iConfig.getParameter<std::string>("Beam1Filename");
  beam2Filename_ = iConfig.getParameter<std::string>("Beam2Filename");
  fCrossingAngleX_45 = iConfig.getParameter<double>("halfCrossingAngleSector45");
  fCrossingAngleX_56 = iConfig.getParameter<double>("halfCrossingAngleSector56");
  beamEnergy_ = iConfig.getParameter<double>("BeamEnergy");
  m_sigmaSTX = iConfig.getParameter<double>("BeamDivergenceX");
  m_sigmaSTY = iConfig.getParameter<double>("BeamDivergenceY");
  m_sigmaSX = iConfig.getParameter<double>("BeamSigmaX");
  m_sigmaSY = iConfig.getParameter<double>("BeamSigmaY");
  m_sig_E = iConfig.getParameter<double>("BeamEnergyDispersion");
  fBeamXatIP = iConfig.getUntrackedParameter<double>("BeamXatIP", 0.);
  fBeamYatIP = iConfig.getUntrackedParameter<double>("BeamYatIP", 0.);

  if (fPPSRegionStart_56 > 0)
    fPPSRegionStart_56 *= -1;  // make sure sector 56 has negative position, as TOTEM convention

  edm::LogInfo("TotemTransport") << "=============================================================================\n"
                                 << "             Bulding LHC Proton transporter based on TOTEM model\n"
                                 << "=============================================================================\n";

  m_aprox_ip_150_r = ReadParameterization(m_model_ip_150_r_name, beam1Filename_);
  m_aprox_ip_150_l = ReadParameterization(m_model_ip_150_l_name, beam2Filename_);

  if (m_aprox_ip_150_r == nullptr || m_aprox_ip_150_l == nullptr) {
    edm::LogError("TotemTransport") << "Parameterisation " << m_model_ip_150_r_name << " or " << m_model_ip_150_l_name
                                    << " missing in file. Cannot proceed. ";
    exit(1);
  }
  edm::LogInfo("TotemTransport") << "Parameterizations read from file, pointers:" << m_aprox_ip_150_r << " "
                                 << m_aprox_ip_150_l << " ";
}
//
// this method is the same for all propagator, but since transportProton is different for each derived class
// it needes to be overriden
//
void TotemTransport::process(const HepMC::GenEvent* evt,
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
    if (m_beamPart.find(line) != m_beamPart.end())  // assures this protons has not been already propagated
      continue;

    transportProton(gpart);
  }
}
//
//
// here comes the real thing
//
//
bool TotemTransport::transportProton(const HepMC::GenParticle* in_trk) {
  //
  //
  edm::LogInfo("TotemTransport") << "Starting proton transport using TOTEM method\n";
  //
  ApplyBeamCorrection(const_cast<HepMC::GenParticle*>(in_trk));

  const HepMC::GenVertex* in_pos = in_trk->production_vertex();
  const HepMC::FourVector in_mom = in_trk->momentum();
  //
  // ATTENTION: HepMC uses mm, vertex config of CMS uses cm and SimTransport uses mm
  //
  double in_position[3] = {in_pos->position().x(), in_pos->position().y(), in_pos->position().z()};  //in LHC ref. frame

  double fCrossingAngleX = (in_mom.z() > 0) ? fCrossingAngleX_45 : fCrossingAngleX_56;

  // Move the position to z=0. Do it in the CMS ref frame. Totem parameterization does the rotation internatlly
  in_position[0] =
      in_position[0] - in_position[2] * (in_mom.x() / in_mom.z() - fCrossingAngleX * urad);  // in CMS ref. frame
  in_position[1] = in_position[1] - in_position[2] * (in_mom.y() / (in_mom.z()));
  in_position[2] = 0.;
  double in_momentum[3] = {in_mom.x(), in_mom.y(), in_mom.z()};
  double out_position[3];
  double out_momentum[3];
  edm::LogInfo("TotemTransport") << "before transport ->"
                                 << " position: " << in_position[0] << ", " << in_position[1] << ", " << in_position[2]
                                 << " momentum: " << in_momentum[0] << ", " << in_momentum[1] << ", " << in_momentum[2];

  LHCOpticsApproximator* approximator_ = nullptr;
  double m_Zin_;
  double m_Zout_;
  if (in_mom.z() > 0) {
    approximator_ = m_aprox_ip_150_l;
    m_Zin_ = 0.0;  // Totem propagations assumes the starting point at 0 (zero)
    m_Zout_ = fPPSRegionStart_45;
  } else {
    approximator_ = m_aprox_ip_150_r;
    m_Zin_ = 0.0;  // Totem propagations assumes the starting point at 0 (zero)
    m_Zout_ = fPPSRegionStart_56;
  }

  bool invert_beam_coord_system =
      true;  // it doesn't matter the option here, it is hard coded as TRUE inside LHCOpticsApproximator!

  bool tracked = approximator_->Transport_m_GeV(
      in_position, in_momentum, out_position, out_momentum, invert_beam_coord_system, m_Zout_ - m_Zin_);

  if (!tracked)
    return false;

  edm::LogInfo("TotemTransport") << "after transport -> "
                                 << "position: " << out_position[0] << ", " << out_position[1] << ", "
                                 << out_position[2] << "momentum: " << out_momentum[0] << ", " << out_momentum[1]
                                 << ", " << out_momentum[2];

  if (out_position[0] * out_position[0] + out_position[1] * out_position[1] >
      m_beampipe_aperture_radius * m_beampipe_aperture_radius) {
    edm::LogInfo("TotemTransport") << "Proton ouside beampipe";
    edm::LogInfo("TotemTransport") << "===== END Transport "
                                   << "====================";
    return false;
  }

  TVector3 out_pos(out_position[0] * meter, out_position[1] * meter, out_position[2] * meter);
  TVector3 out_mom(out_momentum[0], out_momentum[1], out_momentum[2]);

  if (verbosity_) {
    LogDebug("TotemTransport") << "output -> position: ";
    out_pos.Print();
    LogDebug("TotemTransport") << " momentum: ";
    out_mom.Print();
  }

  double px = -out_momentum[0];  // tote calculates px by means of TH_X, which is in the LHC ref. frame.
  double py = out_momentum[1];   // this need to be checked again, since it seems an invertion is occuring in  the prop.
  double pz =
      out_momentum[2];  // totem calculates output pz already in the CMS ref. frame, it doesn't need to be converted
  double e = sqrt(px * px + py * py + pz * pz + ProtonMassSQ);
  TLorentzVector p_out(px, py, pz, e);
  double x1_ctpps = -out_position[0] * meter;  // Totem parameterization uses meter, one need it in millimeter
  double y1_ctpps = out_position[1] * meter;

  unsigned int line = in_trk->barcode();

  if (verbosity_)
    LogDebug("TotemTransport") << "TotemTransport:filterPPS: barcode = " << line << " x=  " << x1_ctpps
                               << " y= " << y1_ctpps;

  m_beamPart[line] = p_out;
  m_xAtTrPoint[line] = x1_ctpps;
  m_yAtTrPoint[line] = y1_ctpps;
  return true;
}
//
LHCOpticsApproximator* TotemTransport::ReadParameterization(const std::string& m_model_name,
                                                            const std::string& rootfile) {
  edm::FileInPath fileName(rootfile.c_str());
  TFile* f = TFile::Open(fileName.fullPath().c_str(), "read");
  if (!f) {
    edm::LogError("TotemTransport") << "File " << fileName << " not found. Exiting.";
    return nullptr;
  }
  edm::LogInfo("TotemTransport") << "Root file opened, pointer:" << f;

  // read parametrization
  LHCOpticsApproximator* aprox = (LHCOpticsApproximator*)f->Get(m_model_name.c_str());
  f->Close();
  return aprox;
}
