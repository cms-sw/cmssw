#include "SimTransport/PPSProtonTransport/interface/TotemTransport.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include <CLHEP/Units/GlobalSystemOfUnits.h>
#include <CLHEP/Random/RandGauss.h>
#include "TLorentzVector.h"
#include "TFile.h"

#include <cmath>

TotemTransport::TotemTransport() : ProtonTransport() { MODE = TransportMode::TOTEM; };
TotemTransport::~TotemTransport() { this->clear(); }
TotemTransport::TotemTransport(const edm::ParameterSet& iConfig, bool verbosity)
    : ProtonTransport(),
      m_parameters(iConfig.getParameter<edm::ParameterSet>("BeamProtTransportSetup")),
      m_verbosity(iConfig.getParameter<bool>("Verbosity")),
      m_model_root_file_r(m_parameters.getParameter<std::string>("ModelRootFile_R")),
      m_model_root_file_l(m_parameters.getParameter<std::string>("ModelRootFile_L")),
      m_model_ip_150_r_name(m_parameters.getParameter<std::string>("Model_IP_150_R_Name")),
      m_model_ip_150_l_name(m_parameters.getParameter<std::string>("Model_IP_150_L_Name")),
      m_model_ip_150_r_zmin(m_parameters.getParameter<double>("Model_IP_150_R_Zmin")),
      m_model_ip_150_r_zmax(m_parameters.getParameter<double>("Model_IP_150_R_Zmax")),
      m_model_ip_150_l_zmin(m_parameters.getParameter<double>("Model_IP_150_L_Zmin")),
      m_model_ip_150_l_zmax(m_parameters.getParameter<double>("Model_IP_150_L_Zmax")),
      m_beampipe_aperture_radius(m_parameters.getParameter<double>("BeampipeApertureRadius")) {
  fBeamEnergy = m_parameters.getParameter<double>("sqrtS");
  m_sigmaSTX = m_parameters.getParameter<double>("beamDivergenceX");
  m_sigmaSTY = m_parameters.getParameter<double>("beamDivergenceY");
  m_sig_E = m_parameters.getParameter<double>("beamEnergyDispersion");
  fCrossingAngle_45 = m_parameters.getParameter<double>("halfCrossingAngleSector45");
  fCrossingAngle_56 = m_parameters.getParameter<double>("halfCrossingAngleSector56");
  fVtxMeanX = iConfig.getParameter<double>("VtxMeanX");
  fVtxMeanY = iConfig.getParameter<double>("VtxMeanY");
  fVtxMeanZ = iConfig.getParameter<double>("VtxMeanZ");
  fBeamXatIP = m_parameters.getUntrackedParameter<double>("BeamXatIP", fVtxMeanX);
  fBeamYatIP = m_parameters.getUntrackedParameter<double>("BeamYatIP", fVtxMeanY);
  bApplyZShift = m_parameters.getParameter<bool>("ApplyZShift");

  MODE = TransportMode::TOTEM;
  edm::LogInfo("ProtonTransport") << "=============================================================================\n"
                                  << "             Bulding LHC Proton transporter based on TOTEM model\n"
                                  << "=============================================================================\n";

  fBeamMomentum = sqrt(fBeamEnergy * fBeamEnergy - ProtonMassSQ);

  fPPSRegionStart_56 = m_model_ip_150_r_zmax;
  fPPSRegionStart_45 = m_model_ip_150_l_zmax;

  m_aprox_ip_150_r = ReadParameterization(m_model_ip_150_r_name, m_model_root_file_r);
  m_aprox_ip_150_l = ReadParameterization(m_model_ip_150_l_name, m_model_root_file_l);

  if (m_aprox_ip_150_r == nullptr || m_aprox_ip_150_l == nullptr) {
    edm::LogError("ProtonTransport") << "Parameterisation " << m_model_ip_150_r_name << " or " << m_model_ip_150_l_name
                                     << " missing in file. Cannot proceed. ";
    exit(1);
  }
  edm::LogInfo("TotemRPProtonTransportSetup")
      << "Parameterizations read from file, pointers:" << m_aprox_ip_150_r << " " << m_aprox_ip_150_l << " ";
}
void TotemTransport::process(const HepMC::GenEvent* evt,
                             const edm::EventSetup& iSetup,
                             CLHEP::HepRandomEngine* _engine) {
  engine = _engine;

  for (HepMC::GenEvent::particle_const_iterator eventParticle = evt->particles_begin();
       eventParticle != evt->particles_end();
       ++eventParticle) {
    if (!((*eventParticle)->status() == 1 && (*eventParticle)->pdg_id() == 2212))
      continue;
    unsigned int line = (*eventParticle)->barcode();
    HepMC::GenParticle* gpart = (*eventParticle);
    if (gpart->pdg_id() != 2212)
      continue;  // only transport stable protons
    if (gpart->status() != 1 /*&& gpart->status()<83 */)
      continue;
    if (m_beamPart.find(line) != m_beamPart.end())
      continue;

    transportProton(gpart);
  }
  addPartToHepMC(const_cast<HepMC::GenEvent*>(evt));
}
bool TotemTransport::transportProton(const HepMC::GenParticle* in_trk) {
  edm::LogInfo("ProtonTransport") << "Starting proton transport using TOTEM method\n";
  ApplyBeamCorrection(const_cast<HepMC::GenParticle*>(in_trk));

  const HepMC::GenVertex* in_pos = in_trk->production_vertex();
  const HepMC::FourVector in_mom = in_trk->momentum();
  //
  // ATTENTION: HepMC uses mm, vertex config of CMS uses cm and SimTransport uses mm
  //
  double in_position[3] = {(in_pos->position().x() - fVtxMeanX * cm) / meter + fBeamXatIP * mm / meter,
                           (in_pos->position().y() - fVtxMeanY * cm) / meter + fBeamYatIP * mm / meter,
                           (in_pos->position().z() - fVtxMeanZ * cm) / meter};  // move to z=0 if configured below

  // (bApplyZShift) -- The TOTEM parameterization requires the shift to z=0
  double fCrossingAngle = (in_mom.z() > 0) ? fCrossingAngle_45 : -fCrossingAngle_56;
  in_position[0] = in_position[0] +
                   (tan((long double)fCrossingAngle * urad) - ((long double)in_mom.x()) / ((long double)in_mom.z())) *
                       in_position[2];
  in_position[1] = in_position[1] - ((long double)in_mom.y()) / ((long double)in_mom.z()) * in_position[2];
  in_position[2] = 0.;
  //
  double in_momentum[3] = {in_mom.x(), in_mom.y(), in_mom.z()};
  double out_position[3];
  double out_momentum[3];
  edm::LogInfo("ProtonTransport") << "before transport ->"
                                  << " position: " << in_position[0] << ", " << in_position[1] << ", " << in_position[2]
                                  << " momentum: " << in_momentum[0] << ", " << in_momentum[1] << ", "
                                  << in_momentum[2];

  LHCOpticsApproximator* approximator_ = nullptr;
  if (in_mom.z() > 0) {
    approximator_ = m_aprox_ip_150_l;
    m_Zin_ = m_model_ip_150_l_zmin;
    m_Zout_ = m_model_ip_150_l_zmax;
  } else {
    approximator_ = m_aprox_ip_150_r;
    m_Zin_ = m_model_ip_150_r_zmin;
    m_Zout_ = m_model_ip_150_r_zmax;
  }

  bool invert_beam_coord_system =
      true;  // it doesn't matter the option here, it is hard coded as TRUE inside LHCOpticsApproximator!

  bool tracked = approximator_->Transport_m_GeV(
      in_position, in_momentum, out_position, out_momentum, invert_beam_coord_system, m_Zout_ - m_Zin_);

  if (!tracked)
    return false;

  edm::LogInfo("ProtonTransport") << "after transport -> "
                                  << "position: " << out_position[0] << ", " << out_position[1] << ", "
                                  << out_position[2] << "momentum: " << out_momentum[0] << ", " << out_momentum[1]
                                  << ", " << out_momentum[2];

  if (out_position[0] * out_position[0] + out_position[1] * out_position[1] >
      m_beampipe_aperture_radius * m_beampipe_aperture_radius) {
    edm::LogInfo("ProtonTransport") << "Proton ouside beampipe";
    edm::LogInfo("ProtonTransport") << "===== END Transport "
                                    << "====================";
    return false;
  }

  TVector3 out_pos(out_position[0] * meter, out_position[1] * meter, out_position[2] * meter);
  TVector3 out_mom(out_momentum[0], out_momentum[1], out_momentum[2]);
  edm::LogInfo("TotemRPProtonTransportModel") << "output -> "
                                              << "position: ";
  out_pos.Print();
  edm::LogInfo("TotemRPProtonTransportModel") << " momentum: ";
  out_mom.Print();
  double px = -out_momentum[0];
  double py = out_momentum[1];  // this need to be checked again, since it seems an invertion is occuring in  the prop.
  double pz = out_momentum[2];
  double e = sqrt(px * px + py * py + pz * pz + ProtonMassSQ);
  TLorentzVector p_out(px, py, pz, e);
  double x1_ctpps = -out_position[0] * meter;  // Totem parameterization uses meter, one need it in millimeter
  double y1_ctpps = -out_position[1] * meter;

  unsigned int line = in_trk->barcode();

  if (m_verbosity)
    LogDebug("ProtonTransportEventProcessing")
        << "ProtonTransport:filterPPS: barcode = " << line << " x=  " << x1_ctpps << " y= " << y1_ctpps;

  m_beamPart[line] = p_out;
  m_xAtTrPoint[line] = x1_ctpps;
  m_yAtTrPoint[line] = y1_ctpps;
  return true;
}
LHCOpticsApproximator* TotemTransport::ReadParameterization(const std::string& m_model_name,
                                                            const std::string& rootfile) {
  edm::FileInPath fileName(rootfile.c_str());
  TFile* f = TFile::Open(fileName.fullPath().c_str(), "read");
  if (!f) {
    edm::LogError("TotemRPProtonTransportSetup") << "File " << fileName << " not found. Exiting.";
    return nullptr;
  }
  edm::LogInfo("TotemRPProtonTransportSetup") << "Root file opened, pointer:" << f;

  // read parametrization
  LHCOpticsApproximator* aprox = (LHCOpticsApproximator*)f->Get(m_model_name.c_str());
  f->Close();
  return aprox;
}
