/****************************************************************************
 *
 * Authors:
 *   Jan Kašpar, Lorenzo Marini, Lorenzo Pagliai, Francesco Turini, Nicola Turini
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//#include "HepPDT/ParticleDataTable.hh"
//#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "CLHEP/Random/RandFlat.h"
#include "CLHEP/Random/RandGauss.h"
#include "CLHEP/Random/RandExponential.h"
#include "CLHEP/Random/RandBreitWigner.h"
#include "CLHEP/Vector/LorentzVector.h"
#include "CLHEP/Vector/ThreeVector.h"

#include "FWCore/Framework/interface/one/EDProducer.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "SimCTPPS/Generators/plugins/particle_ids.h"

//----------------------------------------------------------------------------------------------------
  
class PPXZGenerator : public edm::one::EDProducer<>
{

  public:
    PPXZGenerator(const edm::ParameterSet &);

    virtual ~PPXZGenerator();

  private:
    virtual void produce(edm::Event & e, const edm::EventSetup& es) override;

    // input parameters
    unsigned int verbosity;

    bool decayX;
    bool decayZToElectrons;
    bool decayZToMuons;

    const double m_X;       // mass of the X particle, GeV
    const double m_Z_mean;  // mass of the Z particle, mean, GeV
    const double m_Z_gamma; // mass of the Z particle, gamma, GeV

    const double m_X_pr1;   // mass of the X particle product 1, GeV
    const double m_X_pr2;   // mass of the X particle product 2, GeV

    const double m_e;       // mass of the X electron, GeV
    const double m_mu;      // mass of the X electron, GeV

    const double p_beam;    // beam momentum, GeV

    const double m_XZ_min;  // minimal value of invariant mass of the X-Z system, GeV
    const double c_XZ;      // parameter of the exponential distribution for the invariant mass of the X-Z system, GeV

    const double p_z_LAB_2p_min; // minimum of p_z of the 2-proton system in the LAB frame, GeV
    const double p_z_LAB_2p_max; // maximum of p_z of the 2-proton system in the LAB frame, GeV

    const double p_T_Z_min; // minimum value of Z's pT in the LAB frame, GeV
};

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

using namespace edm;
using namespace std;

//----------------------------------------------------------------------------------------------------

PPXZGenerator::PPXZGenerator(const edm::ParameterSet& pset) :
  verbosity(pset.getUntrackedParameter<unsigned int>("verbosity", 0)),

  decayX(pset.getParameter<bool>("decayX")),
  decayZToElectrons(pset.getParameter<bool>("decayZToElectrons")),
  decayZToMuons(pset.getParameter<bool>("decayZToMuons")),

  m_X(pset.getParameter<double>("m_X")),
  m_Z_mean(pset.getParameter<double>("m_Z_mean")),
  m_Z_gamma(pset.getParameter<double>("m_Z_gamma")),

  m_X_pr1(pset.getParameter<double>("m_X_pr1")),
  m_X_pr2(pset.getParameter<double>("m_X_pr2")),

  m_e(pset.getParameter<double>("m_e")),
  m_mu(pset.getParameter<double>("m_mu")),

  p_beam(pset.getParameter<double>("p_beam")),
  m_XZ_min(pset.getParameter<double>("m_XZ_min")),
  c_XZ(pset.getParameter<double>("c_XZ")),
  p_z_LAB_2p_min(pset.getParameter<double>("p_z_LAB_2p_min")),
  p_z_LAB_2p_max(pset.getParameter<double>("p_z_LAB_2p_max")),

  p_T_Z_min(pset.getParameter<double>("p_T_Z_min"))
{
  produces<HepMCProduct>("unsmeared");
}

//----------------------------------------------------------------------------------------------------

void PPXZGenerator::produce(edm::Event &e, const edm::EventSetup& es)
{
  if (verbosity)
    printf("\n>> PPXZGenerator::produce > event %llu\n", e.id().event());

  // get conditions
  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine* engine = &rng->getEngine(e.streamID());

  //ESHandle<HepPDT::ParticleDataTable> pdgTable;
  //es.getData(pdgTable);

  // prepare HepMC event
  HepMC::GenEvent *fEvt = new HepMC::GenEvent();
  fEvt->set_event_number(e.id().event());

  // generate vertex position
  HepMC::GenVertex *vtx = new HepMC::GenVertex(HepMC::FourVector(0., 0., 0., 0.));
  fEvt->add_vertex(vtx);

  //const HepPDT::ParticleData *pData = pdgTable->particle(HepPDT::ParticleID(particleId));
  //double mass_1 = pData->mass().value();
  //double mass_2 = pData->mass().value();

  // generate mass of Z and mass of the X-Z system
  const double c_XZ_mean = 1. / c_XZ;
  double m_Z = -1., m_XZ = -1.;

  // four-momenta of the outgoing particles in the LAB frame
  CLHEP::HepLorentzVector momentum_Z;
  CLHEP::HepLorentzVector momentum_X;

  CLHEP::HepLorentzVector momentum_p1;
  CLHEP::HepLorentzVector momentum_p2;

  // try to generate event fullfilling all criteria
  bool generationOK = false;
  for (unsigned int n_attempt = 0; n_attempt < 10000; ++n_attempt)
  {
    m_Z = CLHEP::RandBreitWigner::shoot(engine, m_Z_mean, m_Z_gamma);
    m_XZ = m_XZ_min + CLHEP::RandExponential::shoot(engine, c_XZ_mean);

    if (m_Z < 0. || m_XZ < m_Z + m_X)
      continue;

    // generate p_z of the 2-proton system in the LAB frame
    const double p_z_LAB_2p = CLHEP::RandFlat::shoot(engine, p_z_LAB_2p_min, p_z_LAB_2p_max);

    // generate spherical angles in the CMS frame of the X-Z system
    const double cos_theta_c = 2. * CLHEP::RandFlat::shoot(engine) - 1.;
    const double sin_theta_c = sqrt(1. - cos_theta_c * cos_theta_c);
    const double phi_c = CLHEP::RandFlat::shoot(engine) * 2. * M_PI;

    // determine xi's of the protons
    // proton 1: positive z momentum component
    const double xi2 = (p_z_LAB_2p + sqrt(p_z_LAB_2p*p_z_LAB_2p + m_XZ*m_XZ)) / (2. * p_beam);
    const double xi1 = m_XZ * m_XZ / (4. * p_beam * p_beam * xi2);

    if (verbosity)
    {
      printf("  m_Z = %.1f\n", m_Z);
      printf("  m_XZ = %.1f\n", m_XZ);
      printf("  p_z_LAB_2p = %.1f\n", p_z_LAB_2p);
      printf("  xi1 = %.3f, xi2 = %.3f\n", xi1, xi2);
      printf("  p_beam * (xi2 - xi1) = %.1f\n", p_beam * (xi2 - xi1));
    }

    // determine momenta of the X and Z particles in the CMS frame of the X-Z system
    const double p_c_sq = pow(m_XZ*m_XZ - m_X*m_X - m_Z*m_Z, 2.) / (4. * m_XZ * m_XZ) - m_X*m_X * m_Z*m_Z / (m_XZ*m_XZ);
    const double p_c = sqrt(p_c_sq);

    if (verbosity)
      printf("  p_c = %.3f\n", p_c);

    CLHEP::HepLorentzVector momentum_X_CMS(
      + p_c * sin_theta_c * cos(phi_c),
      + p_c * sin_theta_c * sin(phi_c),
      + p_c * cos_theta_c,
      sqrt(p_c*p_c + m_X*m_X)
    );

    CLHEP::HepLorentzVector momentum_Z_CMS(
      - p_c * sin_theta_c * cos(phi_c),
      - p_c * sin_theta_c * sin(phi_c),
      - p_c * cos_theta_c,
      sqrt(p_c*p_c + m_Z*m_Z)
    );

    // determine boost from X-Z CMS frame to the LAB frame
    const double beta = (xi1 - xi2) / (xi1 + xi2);
    const CLHEP::Hep3Vector betaVector(0., 0., beta);

    if (verbosity)
      printf("  beta = %.3f\n", beta);

    // determine four-momenta of the outgoing particles in the LAB frame
    momentum_Z = CLHEP::boostOf(momentum_Z_CMS, betaVector);
    momentum_X = CLHEP::boostOf(momentum_X_CMS, betaVector);

    momentum_p1 = CLHEP::HepLorentzVector(0., 0., +p_beam * (1. - xi1), p_beam * (1. - xi1));
    momentum_p2 = CLHEP::HepLorentzVector(0., 0., -p_beam * (1. - xi2), p_beam * (1. - xi2));

    if (verbosity)
    {
      printf("  p_X_z = %.1f\n", momentum_X.z());
      printf("  p_Z_z = %.1f\n", momentum_Z.z());

      const CLHEP::HepLorentzVector m_tot = momentum_p1 + momentum_p2 + momentum_X + momentum_Z;
      printf("  four-momentum of p + p + X + Z: (%.1f, %.1f, %.1f | %.1f)\n", m_tot.x(), m_tot.y(), m_tot.z(), m_tot.t());
    }

    if (momentum_Z.perp() > p_T_Z_min)
    {
      generationOK = true;
      break;
    }
  }

  if (!generationOK)
    throw cms::Exception("PPXZGenerator") << "Failed to generate event.";

  // fill in the HepMC record
  unsigned int barcode = 0;

  // status codes
  //const int statusInitial = 3;
  const int statusFinal = 1;
  const int statusDecayed = 2;

  int status_X = (decayX) ? statusDecayed : statusFinal;
  int status_Z = (decayZToElectrons || decayZToMuons) ? statusDecayed : statusFinal;

  HepMC::GenParticle* particle_Z = new HepMC::GenParticle(momentum_Z, particleId_Z, status_Z);
  particle_Z->suggest_barcode(++barcode);
  vtx->add_particle_out(particle_Z);

  HepMC::GenParticle* particle_X = new HepMC::GenParticle(momentum_X, particleId_X, status_X);
  particle_X->suggest_barcode(++barcode);
  vtx->add_particle_out(particle_X);

  HepMC::GenParticle* particle_p1 = new HepMC::GenParticle(momentum_p1, particleId_p, statusFinal);
  particle_p1->suggest_barcode(++barcode);
  vtx->add_particle_out(particle_p1);

  HepMC::GenParticle* particle_p2 = new HepMC::GenParticle(momentum_p2, particleId_p, statusFinal);
  particle_p2->suggest_barcode(++barcode);
  vtx->add_particle_out(particle_p2);

  // decay X if desired
  if (decayX)
  {
    // generate decay angles in X's rest frame;
    const double cos_theta_d = 2. * CLHEP::RandFlat::shoot(engine) - 1.;
    const double sin_theta_d = sqrt(1. - cos_theta_d * cos_theta_d);
    const double phi_d = CLHEP::RandFlat::shoot(engine) * 2. * M_PI;

    // product momentum and energy in X's rest frame
    const double M2 = m_X*m_X - m_X_pr1*m_X_pr1 - m_X_pr2*m_X_pr2;
    const double p_pr = sqrt(M2*M2 - 4. * m_X_pr1*m_X_pr1 * m_X_pr2*m_X_pr2) / 2. / m_X;
    const double E_pr1 = sqrt(p_pr*p_pr + m_X_pr1*m_X_pr1);
    const double E_pr2 = sqrt(p_pr*p_pr + m_X_pr2*m_X_pr2);

    // product four-momenta in X's rest frame
    CLHEP::HepLorentzVector momentum_pr1(
      p_pr * sin_theta_d * cos(phi_d),
      p_pr * sin_theta_d * sin(phi_d),
      p_pr * cos_theta_d,
      E_pr1
    );

    CLHEP::HepLorentzVector momentum_pr2(
      -p_pr * sin_theta_d * cos(phi_d),
      -p_pr * sin_theta_d * sin(phi_d),
      -p_pr * cos_theta_d,
      E_pr2
    );

    // apply boost
    double beta = momentum_X.rho() / momentum_X.t();
    CLHEP::Hep3Vector betaVector(momentum_X.x(), momentum_X.y(), momentum_X.z());
    betaVector *= beta / betaVector.mag();
    momentum_pr1 = CLHEP::boostOf(momentum_pr1, betaVector);
    momentum_pr2 = CLHEP::boostOf(momentum_pr2, betaVector);

    if (verbosity)
    {
      const CLHEP::HepLorentzVector m_tot = momentum_p1 + momentum_p2 + momentum_Z + momentum_pr1 + momentum_pr2;
      printf("  four-momentum of p + p + Z + X_pr1 + X_pr2: (%.1f, %.1f, %.1f | %.1f)\n", m_tot.x(), m_tot.y(), m_tot.z(), m_tot.t());
    }

    // add particles to vertex
    HepMC::GenParticle* particle_pr1 = new HepMC::GenParticle(momentum_pr1, particleId_X_pr1, statusFinal);
    particle_pr1->suggest_barcode(++barcode);
    vtx->add_particle_out(particle_pr1);

    HepMC::GenParticle* particle_pr2 = new HepMC::GenParticle(momentum_pr2, particleId_X_pr2, statusFinal);
    particle_pr2->suggest_barcode(++barcode);
    vtx->add_particle_out(particle_pr2);
  }

  // decay Z if desired
  if (decayZToElectrons || decayZToMuons)
  {
    double m_l = 0.;
    signed int particleId_l_mi=0, particleId_l_pl=0;

    if (decayZToElectrons) m_l = m_e, particleId_l_mi = particleId_e_mi, particleId_l_pl = particleId_e_pl;
    if (decayZToMuons) m_l = m_mu, particleId_l_mi = particleId_mu_mi, particleId_l_pl = particleId_mu_pl;

    // generate decay angles in Z's rest frame;
    const double cos_theta_d = 2. * CLHEP::RandFlat::shoot(engine) - 1.;
    const double sin_theta_d = sqrt(1. - cos_theta_d * cos_theta_d);
    const double phi_d = CLHEP::RandFlat::shoot(engine) * 2. * M_PI;

    // lepton momentum and energy in Z's rest frame
    const double E_l = m_Z / 2.;
    const double p_l = sqrt(E_l*E_l - m_l*m_l);

    // lepton four-momenta in Z's rest frame
    CLHEP::HepLorentzVector momentum_l_mi(
      p_l * sin_theta_d * cos(phi_d),
      p_l * sin_theta_d * sin(phi_d),
      p_l * cos_theta_d,
      E_l
    );

    CLHEP::HepLorentzVector momentum_l_pl(
      -p_l * sin_theta_d * cos(phi_d),
      -p_l * sin_theta_d * sin(phi_d),
      -p_l * cos_theta_d,
      E_l
    );

    // apply boost
    double beta = momentum_Z.rho() / momentum_Z.t();
    CLHEP::Hep3Vector betaVector(momentum_Z.x(), momentum_Z.y(), momentum_Z.z());
    betaVector *= beta / betaVector.mag();
    momentum_l_mi = CLHEP::boostOf(momentum_l_mi, betaVector);
    momentum_l_pl = CLHEP::boostOf(momentum_l_pl, betaVector);

    if (verbosity)
    {
      const CLHEP::HepLorentzVector m_tot = momentum_p1 + momentum_p2 + momentum_X + momentum_l_mi + momentum_l_pl;
      printf("  four-momentum of p + p + X + l + l: (%.1f, %.1f, %.1f | %.1f)\n", m_tot.x(), m_tot.y(), m_tot.z(), m_tot.t());
    }

    // add particles to vertex
    HepMC::GenParticle* particle_l_mi = new HepMC::GenParticle(momentum_l_mi, particleId_l_mi, statusFinal);
    particle_l_mi->suggest_barcode(++barcode);
    vtx->add_particle_out(particle_l_mi);

    HepMC::GenParticle* particle_l_pl = new HepMC::GenParticle(momentum_l_pl, particleId_l_pl, statusFinal);
    particle_l_pl->suggest_barcode(++barcode);
    vtx->add_particle_out(particle_l_pl);
  }

  // save output
  std::unique_ptr<HepMCProduct> output(new HepMCProduct()) ;
  output->addHepMCData(fEvt);
  e.put(std::move(output), "unsmeared");
}

//----------------------------------------------------------------------------------------------------

PPXZGenerator::~PPXZGenerator()
{
}

//----------------------------------------------------------------------------------------------------

DEFINE_FWK_MODULE(PPXZGenerator);
