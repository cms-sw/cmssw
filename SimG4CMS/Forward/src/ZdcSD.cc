///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.cc
// Date: 03.01
// Description: Sensitive Detector class for Zdc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include <memory>

#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "SimG4CMS/Forward/interface/ForwardName.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4Cerenkov.hh"
#include "G4ParticleTable.hh"
#include "G4PhysicalConstants.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Randomize.hh"
#include "G4Poisson.hh"
#include "G4TwoVector.hh"

//#define EDM_ML_DEBUG

ZdcSD::ZdcSD(const std::string& name,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : CaloSD(name, clg, p, manager) {
  edm::ParameterSet m_ZdcSD = p.getParameter<edm::ParameterSet>("ZdcSD");
  useShowerLibrary = m_ZdcSD.getParameter<bool>("UseShowerLibrary");
  useShowerHits = m_ZdcSD.getParameter<bool>("UseShowerHits");
  zdcHitEnergyCut = m_ZdcSD.getParameter<double>("ZdcHitEnergyCut") * CLHEP::GeV;
  thFibDir = m_ZdcSD.getParameter<double>("FiberDirection");
  verbosity = m_ZdcSD.getParameter<int>("Verbosity");
  int verbn = verbosity / 10;
  verbosity %= 10;
  numberingScheme = std::make_unique<ZdcNumberingScheme>(verbn);

  edm::LogVerbatim("ForwardSim") << "***************************************************\n"
                                 << "*                                                 *\n"
                                 << "* Constructing a ZdcSD  with name " << name << "   *\n"
                                 << "*                                                 *\n"
                                 << "***************************************************";

  edm::LogVerbatim("ForwardSim") << "\nUse of shower library is set to " << useShowerLibrary
                                 << "\nUse of Shower hits method is set to " << useShowerHits;

  edm::LogVerbatim("ForwardSim") << "\nEnergy Threshold Cut set to " << zdcHitEnergyCut / CLHEP::GeV << " (GeV)";

  if (useShowerLibrary) {
    showerLibrary = std::make_unique<ZdcShowerLibrary>(name, p);
    setParameterized(true);
  } else {
    showerLibrary.reset(nullptr);
  }
}

void ZdcSD::initRun() {
  if (useShowerLibrary) {
    G4ParticleTable* theParticleTable = G4ParticleTable::GetParticleTable();
    showerLibrary->initRun(theParticleTable);
  }
  hits.clear();
}

bool ZdcSD::ProcessHits(G4Step* aStep, G4TouchableHistory*) {
  if (useShowerLibrary)
    getFromLibrary(aStep);

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("ZdcSD") << "ZdcSD::" << GetName() << " ID= " << aStep->GetTrack()->GetTrackID()
                            << " prID= " << aStep->GetTrack()->GetParentID()
                            << " Eprestep= " << aStep->GetPreStepPoint()->GetKineticEnergy()
                            << " step= " << aStep->GetStepLength() << " Edep= " << aStep->GetTotalEnergyDeposit();
#endif
  if (useShowerHits) {
    // check unitID
    unsigned int unitID = setDetUnitId(aStep);
    if (unitID == 0)
      return false;

    auto const theTrack = aStep->GetTrack();
    uint16_t depth = getDepth(aStep);

    double time = theTrack->GetGlobalTime() / nanosecond;
    int primaryID = getTrackID(theTrack);
    currentID[0].setID(unitID, time, primaryID, depth);
    double energy = calculateCherenkovDeposit(aStep);

    // Russian Roulette
    double wt2 = theTrack->GetWeight();
    if (wt2 > 0.0) {
      energy *= wt2;
    }

    if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
      edepositEM = energy;
      edepositHAD = 0;
    } else {
      edepositEM = 0;
      edepositHAD = energy;
    }
    if (!hitExists(aStep, 0) && energy > 0.) {
#ifdef EDM_ML_DEBUG
      G4ThreeVector pre = aStep->GetPreStepPoint()->GetPosition();
      edm::LogVerbatim("ZdcSD") << pre.x() << " " << pre.y() << " " << pre.z();
#endif
      currentHit[0] = CaloSD::createNewHit(aStep, theTrack, 0);
    }
  }
  return true;
}

bool ZdcSD::getFromLibrary(const G4Step* aStep) {
  bool ok = true;

  auto const preStepPoint = aStep->GetPreStepPoint();

  double etrack = preStepPoint->GetKineticEnergy();
  int primaryID = setTrackID(aStep);

  hits.clear();

  // Reset entry point for new primary
  resetForNewPrimary(aStep);

  if (etrack >= zdcHitEnergyCut) {
    // create hits only if above threshold

#ifdef EDM_ML_DEBUG
    auto const theTrack = aStep->GetTrack();
    edm::LogVerbatim("ForwardSim") << "----------------New track------------------------------\n"
                                   << "Incident EnergyTrack: " << etrack << " MeV \n"
                                   << "Zdc Cut Energy for Hits: " << zdcHitEnergyCut << " MeV \n"
                                   << "ZdcSD::getFromLibrary " << hits.size() << " hits for " << GetName() << " of "
                                   << primaryID << " with " << theTrack->GetDefinition()->GetParticleName() << " of "
                                   << etrack << " MeV\n";
#endif
    hits.swap(showerLibrary.get()->getHits(aStep, ok));
  }

  incidentEnergy = etrack;
  entrancePoint = preStepPoint->GetPosition();
  for (unsigned int i = 0; i < hits.size(); i++) {
    posGlobal = hits[i].position;
    entranceLocal = hits[i].entryLocal;
    double time = hits[i].time;
    unsigned int unitID = hits[i].detID;
    edepositHAD = hits[i].DeHad;
    edepositEM = hits[i].DeEM;
    currentID[0].setID(unitID, time, primaryID, 0);
    processHit(aStep);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardSim") << "ZdcSD: Final Hit number:" << i << "-->"
                                   << "New HitID: " << currentHit[0]->getUnitID()
                                   << " New Hit trackID: " << currentHit[0]->getTrackID()
                                   << " New EM Energy: " << currentHit[0]->getEM() / CLHEP::GeV
                                   << " New HAD Energy: " << currentHit[0]->getHadr() / CLHEP::GeV
                                   << " New HitEntryPoint: " << currentHit[0]->getEntryLocal()
                                   << " New IncidentEnergy: " << currentHit[0]->getIncidentEnergy() / CLHEP::GeV
                                   << " New HitPosition: " << posGlobal;
#endif
  }
  return ok;
}

double ZdcSD::getEnergyDeposit(const G4Step* aStep) {
  double NCherPhot = 0.;

  // preStepPoint information
  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();

  const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
  const G4ThreeVector& hit_mom = preStepPoint->GetMomentumDirection();
  G4double stepL = aStep->GetStepLength() / cm;
  G4double beta = preStepPoint->GetBeta();
  G4double charge = preStepPoint->GetCharge();
  if (charge == 0.0)
    return 0.0;

  // theTrack information
  G4Track* theTrack = aStep->GetTrack();
  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  G4ThreeVector localPoint = theTrack->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);

#ifdef EDM_ML_DEBUG
  const G4ThreeVector& vert_mom = theTrack->GetVertexMomentumDirection();

  // calculations
  float costheta =
      vert_mom.z() / sqrt(vert_mom.x() * vert_mom.x() + vert_mom.y() * vert_mom.y() + vert_mom.z() * vert_mom.z());
  float theta = std::acos(std::min(std::max(costheta, -1.f), 1.f));
  float eta = -std::log(std::tan(theta * 0.5f));
  float phi = -100.;
  if (vert_mom.x() != 0)
    phi = std::atan2(vert_mom.y(), vert_mom.x());
  if (phi < 0.)
    phi += twopi;

  // Get the total energy deposit
  double stepE = aStep->GetTotalEnergyDeposit();

  // postStepPoint information
  G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
  G4VPhysicalVolume* postPV = postStepPoint->GetPhysicalVolume();
  std::string postnameVolume = ForwardName::getName(postPV->GetName());
  edm::LogVerbatim("ForwardSim") << "ZdcSD::  getEnergyDeposit: \n"
                                 << "  preStepPoint: " << nameVolume << "," << stepL << "," << stepE << "," << beta
                                 << "," << charge << "\n"
                                 << "  postStepPoint: " << postnameVolume << "," << costheta << "," << theta << ","
                                 << eta << "," << phi << "," << particleType << " id= " << theTrack->GetTrackID()
                                 << " Etot(GeV)= " << theTrack->GetTotalEnergy() / CLHEP::GeV;
#endif
  const double bThreshold = 0.67;
  if (beta > bThreshold) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardSim") << "ZdcSD::  getEnergyDeposit:  pass ";
#endif
    const float nMedium = 1.4925;
    // float photEnSpectrDL = 10714.285714;
    //       photEnSpectrDL = (1./400.nm-1./700.nm)*10000000.cm/nm; /* cm-1  */

    const float photEnSpectrDE = 1.24;
    // E = 2pi*(1./137.)*(eV*cm/370.)/lambda = 12.389184*(eV*cm)/lambda
    // Emax = 12.389184*(eV*cm)/400nm*10-7cm/nm  = 3.01 eV
    // Emin = 12.389184*(eV*cm)/700nm*10-7cm/nm  = 1.77 eV
    // delE = Emax - Emin = 1.24 eV

    const float effPMTandTransport = 0.15;

    // Check these values
    const float thFullRefl = 23.;
    float thFullReflRad = thFullRefl * pi / 180.;

    float thFibDirRad = thFibDir * pi / 180.;

    // at which theta the point is located:
    //   float th1 = hitPoint.theta();

    // theta of charged particle in LabRF(hit momentum direction):
    float costh = hit_mom.z() / sqrt(hit_mom.x() * hit_mom.x() + hit_mom.y() * hit_mom.y() + hit_mom.z() * hit_mom.z());
    float th = acos(std::min(std::max(costh, -1.f), 1.f));
    // just in case (can do both standard ranges of phi):
    if (th < 0.)
      th += CLHEP::twopi;

    // theta of cone with Cherenkov photons w.r.t.direction of charged part.:
    float costhcher = 1. / (nMedium * beta);
    float thcher = acos(std::min(std::max(costhcher, -1.f), 1.f));

    // diff thetas of charged part. and quartz direction in LabRF:
    float DelFibPart = std::abs(th - thFibDirRad);

    // define real distances:
    float d = std::abs(std::tan(th) - std::tan(thFibDirRad));

    float a = std::tan(thFibDirRad) + std::tan(std::abs(thFibDirRad - thFullReflRad));
    float r = std::tan(th) + std::tan(std::abs(th - thcher));

    // define losses d_qz in cone of full reflection inside quartz direction
    float d_qz = -1;
#ifdef EDM_ML_DEBUG
    float variant = -1;
#endif
    // if (d > (r+a))
    if (DelFibPart > (thFullReflRad + thcher)) {
#ifdef EDM_ML_DEBUG
      variant = 0.;
#endif
      d_qz = 0.;
    } else {
      // if ((DelFibPart + thcher) < thFullReflRad )  [(d+r) < a]
      if ((th + thcher) < (thFibDirRad + thFullReflRad) && (th - thcher) > (thFibDirRad - thFullReflRad)) {
#ifdef EDM_ML_DEBUG
        variant = 1.;
#endif
        d_qz = 1.;
      } else {
        // if ((thcher - DelFibPart ) > thFullReflRad )  [(r-d) > a]
        if ((thFibDirRad + thFullReflRad) < (th + thcher) && (thFibDirRad - thFullReflRad) > (th - thcher)) {
#ifdef EDM_ML_DEBUG
          variant = 2.;
#endif
          d_qz = 0.;
        } else {
#ifdef EDM_ML_DEBUG
          variant = 3.;  // d_qz is calculated below
#endif
          // use crossed length of circles(cone projection) - dC1/dC2 :
          float arg_arcos = 0.;
          float tan_arcos = 2. * a * d;
          if (tan_arcos != 0.)
            arg_arcos = (r * r - a * a - d * d) / tan_arcos;
          arg_arcos = std::abs(arg_arcos);
          float th_arcos = acos(std::min(std::max(arg_arcos, -1.f), 1.f));
          d_qz = th_arcos / CLHEP::twopi;
          d_qz = std::abs(d_qz);
#ifdef EDM_ML_DEBUG
          edm::LogVerbatim("ForwardSim") << "  d_qz: " << r << "," << a << "," << d << " " << tan_arcos << " "
                                         << arg_arcos;
          edm::LogVerbatim("ForwardSim") << "," << arg_arcos;
          edm::LogVerbatim("ForwardSim") << " " << d_qz;
          edm::LogVerbatim("ForwardSim") << " " << th_arcos;
          edm::LogVerbatim("ForwardSim") << "," << d_qz;
#endif
        }
      }
    }
    double meanNCherPhot = 0.;
    int poissNCherPhot = 0;
    if (d_qz > 0) {
      meanNCherPhot = 370. * charge * charge * (1. - 1. / (nMedium * nMedium * beta * beta)) * photEnSpectrDE * stepL;

      poissNCherPhot = std::max((int)G4Poisson(meanNCherPhot), 0);
      NCherPhot = poissNCherPhot * effPMTandTransport * d_qz;
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ForwardSim") << "ZdcSD::  getEnergyDeposit:  gED: " << stepE << "," << costh << "," << th << ","
                                   << costhcher << "," << thcher << "," << DelFibPart << "," << d << "," << a << ","
                                   << r << "," << hitPoint << "," << hit_mom << "," << vert_mom << "," << localPoint
                                   << "," << charge << "," << beta << "," << stepL << "," << d_qz << "," << variant
                                   << "," << meanNCherPhot << "," << poissNCherPhot << "," << NCherPhot;
#endif

  } else {
    // determine failure mode: beta, charge, and/or nameVolume
    if (beta <= bThreshold)
      edm::LogVerbatim("ForwardSim") << "ZdcSD::  getEnergyDeposit: fail beta=" << beta;
  }

  return NCherPhot;
}

///////////////////////////////////////
// Functions added by Oliver Suranyi //
///////////////////////////////////////

// Constants as global variables
const double RINDEX = 1.47;
const double NA = 0.22;  // Numerical aperture, characteristic of the specific fiber
const double NAperRINDEX = NA / RINDEX;
const double EMAX = 4.79629 /*eV*/;                                    // Maximum energy of PMT sensitivity range
const double EMIN = 1.75715 /*eV*/;                                    // Minimum energy of PMT sensitivity range
const double ALPHA = /*1/137=*/0.0072973525693;                        // Fine structure constant
const double HBARC = 6.582119514E-16 /*eV*s*/ * 2.99792458E8 /*m/s*/;  // hbar * c

// Calculate the Cherenkov deposit corresponding to a G4Step
double ZdcSD::calculateCherenkovDeposit(const G4Step* aStep) {
  const G4StepPoint* pPreStepPoint = aStep->GetPreStepPoint();
  G4double charge = pPreStepPoint->GetCharge() / CLHEP::eplus;
  if (charge == 0.0 || aStep->GetStepLength() < 1e-9 * CLHEP::mm)
    return 0.0;

  const G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();

  G4ThreeVector pre = pPreStepPoint->GetPosition();
  G4ThreeVector post = pPostStepPoint->GetPosition();

  //Convert step coordinates to local (fiber) coodinates
  const G4ThreeVector localPre = setToLocal(pre, pPreStepPoint->GetTouchable());
  const G4ThreeVector localPost = setToLocal(post, pPreStepPoint->GetTouchable());

  // Calculate the unit direction vector in local coordinates
  const G4ThreeVector particleDirection = (localPost - localPre) / (localPost - localPre).mag();

  double beta = 0.5 * (pPreStepPoint->GetBeta() + pPostStepPoint->GetBeta());
  double stepLength = aStep->GetStepLength() / 1000;  // Geant4 stepLength is in "mm"

  int nPhotons;  // Number of Cherenkov photons

  nPhotons = G4Poisson(calculateMeanNumberOfPhotons(charge, beta, stepLength));

  double totalE = 0.0;

  for (int i = 0; i < nPhotons; ++i) {
    // uniform refractive index in PMT range -> uniform energy distribution
    double photonE = EMIN + G4UniformRand() * (EMAX - EMIN);
    // UPDATE: taking into account dispersion relation -> energy distribution

    if (G4UniformRand() > pmtEfficiency(convertEnergyToWavelength(photonE)))
      continue;

    double omega = G4UniformRand() * twopi;
    double cosTheta = std::min(1.0 / (beta * RINDEX), 1.0);
    double sinTheta = std::sqrt((1. - cosTheta) * (1.0 + cosTheta));

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ZdcSD") << "E_gamma: " << photonE << "\t omega: " << omega << "\t thetaC: " << cosTheta;
#endif
    // Calculate momentum direction w.r.t primary particle (z-direction)
    double px = photonE * sinTheta * std::cos(omega);
    double py = photonE * sinTheta * std::sin(omega);
    double pz = photonE * cosTheta;
    G4ThreeVector photonMomentum(px, py, pz);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ZdcSD") << "pPR = (" << particleDirection.x() << "," << particleDirection.y() << ","
                              << particleDirection.z() << ")";
    edm::LogVerbatim("ZdcSD") << "pCH = (" << px << "," << py << "," << pz << ")";
#endif
    // Rotate to the fiber reference frame
    photonMomentum.rotateUz(particleDirection);

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ZdcSD") << "pLAB = (" << photonMomentum.x() << "," << photonMomentum.y() << ","
                              << photonMomentum.z() << ")";
#endif
    // Get random position along G4Step
    G4ThreeVector photonPosition = localPre + G4UniformRand() * (localPost - localPre);

    // 2D vectors to calculate impact position (x*,y*)
    G4TwoVector r0(photonPosition);
    G4TwoVector v0(photonMomentum);

    double R = 0.3; /*mm, fiber radius*/
    double R2 = 0.3 * 0.3;

    if (r0.mag() < R && photonMomentum.z() < 0.0) {
      // 2nd order polynomial coefficients
      double a = v0.mag2();
      double b = 2.0 * r0 * v0;
      double c = r0.mag2() - R2;

      if (a < 1E-6)
        totalE += 1;  //photonE /*eV*/;
      else {
        // calculate intersection point - solving 2nd order polynomial
        double t = (-b + sqrt(b * b - 4.0 * a * c)) / (2.0 * a);
        G4ThreeVector n(r0.x() + v0.x() * t, r0.y() + v0.y() * t, 0.0);  // surface normal
        double cosTheta = (n * photonMomentum) / (n.mag() * photonE);    // cosine of incident angle

        if (cosTheta >= NAperRINDEX)  // lightguide condition
          totalE += 1;                //photonE /*eV*/;
      }
    }

#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("ZdcSD") << "r = (" << photonPosition.x() << "," << photonPosition.y() << "," << photonPosition.z()
                              << ")" << std::endl;
#endif
  }

#ifdef EDM_ML_DEBUG
  if (nPhotons > 30) {
    edm::LogVerbatim("ZdcSD") << totalE;

    if (totalE > 0)
      edm::LogVerbatim("ZdcSD") << pre.x() << " " << pre.y() << " " << pre.z() << " " << totalE;
  }
#endif
  return totalE;
}

// Calculate mean number of Cherenkov photons in the sensitivity range (300-650 nm)
// for a given step length for a particle with given charge and beta
double ZdcSD::calculateMeanNumberOfPhotons(double charge, double beta, double stepLength) {
  // Return mean number of Cherenkov photons
  return (ALPHA * charge * charge * stepLength) / HBARC * (EMAX - EMIN) * (1.0 - 1.0 / (beta * beta * RINDEX * RINDEX));
}

// Evaluate photon pdf
double ZdcSD::photonEnergyDist(double charge, double beta, double E) {
  const std::vector<double> ENERGY_TAB{1.75715, 1.81902, 1.88311, 1.94944, 2.0183,  2.08939, 2.16302, 2.23919,
                                       2.31789, 2.39954, 2.48416, 2.57175, 2.66232, 2.75643, 2.85349, 2.95411,
                                       3.05756, 3.16528, 3.2774,  3.39218, 3.5123,  3.6359,  3.76394, 3.89642,
                                       4.03332, 4.17596, 4.32302, 4.47596, 4.63319, 4.79629};

  const std::vector<double> RINDEX_TAB{1.45517, 1.45572, 1.45631, 1.45693, 1.45758, 1.45826, 1.45899, 1.45976,
                                       1.46057, 1.46144, 1.46238, 1.46337, 1.46444, 1.46558, 1.4668,  1.46812,
                                       1.46953, 1.47105, 1.4727,  1.47447, 1.4764,  1.47847, 1.48071, 1.48315,
                                       1.48579, 1.48868, 1.49182, 1.49526, 1.499,   1.5031};
  double rIndex = evaluateFunction(ENERGY_TAB, RINDEX_TAB, E);
  return (ALPHA * charge * charge) / HBARC * (1.0 - 1.0 / (beta * beta * rIndex * rIndex));
}

// Generate a photon with the given minimum energy accourding to the energy distribution
double ZdcSD::generatePhotonEnergy(double charge, double beta, double Emin) {
  double photonE;

  // Use rejection method
  do {
    photonE = G4UniformRand() * (EMAX - Emin) + Emin;
  } while (G4UniformRand() > photonEnergyDist(photonE, charge, beta) / photonEnergyDist(EMAX, charge, beta));

  return photonE;
}

// Calculate the integral: int_Emin^Emax 1/n(E)^2 dE
// The integral values are tabulated
double ZdcSD::calculateN2InvIntegral(double Emin) {
  // Hardcoded minimum photon energy table (eV)
  const std::vector<double> EMIN_TAB{1.75715, 1.81902, 1.88311, 1.94944, 2.0183,  2.08939, 2.16302, 2.23919,
                                     2.31789, 2.39954, 2.48416, 2.57175, 2.66232, 2.75643, 2.85349, 2.95411,
                                     3.05756, 3.16528, 3.2774,  3.39218, 3.5123,  3.6359,  3.76394, 3.89642,
                                     4.03332, 4.17596, 4.32302, 4.47596, 4.63319};

  // Hardcoded integral values
  const std::vector<double> INTEGRAL_TAB{1.39756,  1.36835,  1.33812,  1.30686,  1.27443,  1.24099,  1.20638,  1.17061,
                                         1.1337,   1.09545,  1.05586,  1.01493,  0.972664, 0.928815, 0.883664, 0.836938,
                                         0.788988, 0.739157, 0.687404, 0.634547, 0.579368, 0.522743, 0.464256, 0.40393,
                                         0.341808, 0.27732,  0.211101, 0.142536, 0.0723891};
  return evaluateFunction(EMIN_TAB, INTEGRAL_TAB, Emin);
}

double ZdcSD::pmtEfficiency(double lambda) {
  // Hardcoded wavelength values (nm)
  const std::vector<double> LAMBDA_TAB{263.27, 265.98, 268.69, 271.39, 273.20, 275.90, 282.22, 282.22, 293.04,
                                       308.38, 325.52, 346.26, 367.91, 392.27, 417.53, 440.98, 463.53, 484.28,
                                       502.32, 516.75, 528.48, 539.30, 551.93, 564.56, 574.48, 584.41, 595.23,
                                       606.96, 616.88, 625.00, 632.22, 637.63, 642.14, 647.55, 652.96, 656.57,
                                       661.08, 666.49, 669.20, 673.71, 677.32, 680.93, 686.34, 692.65};

  // Hardcoded quantum efficiency values
  const std::vector<double> EFF_TAB{2.215,  2.860,  3.659,  4.724,  5.989,  7.734,  9.806,  9.806,  12.322,
                                    15.068, 17.929, 20.570, 22.963, 24.050, 23.847, 22.798, 20.445, 18.003,
                                    15.007, 12.282, 9.869,  7.858,  6.373,  5.121,  4.077,  3.276,  2.562,
                                    2.077,  1.669,  1.305,  1.030,  0.805,  0.629,  0.492,  0.388,  0.303,
                                    0.239,  0.187,  0.144,  0.113,  0.088,  0.069,  0.054,  0.043};
  //double efficiency = evaluateFunction(LAMBDA_TAB,EFF_TAB,lambda);

  // Using linear interpolation to calculate efficiency
  for (int i = 0; i < 44 - 1; i++) {
    if (lambda > LAMBDA_TAB[i] && lambda < LAMBDA_TAB[i + 1]) {
      double a = (EFF_TAB[i] - EFF_TAB[i + 1]) / (LAMBDA_TAB[i] - LAMBDA_TAB[i + 1]);
      double b = EFF_TAB[i] - a * LAMBDA_TAB[i];
      return (a * lambda + b) / 100.0;
    }
  }

  return 0;
}

// Evaluate a function given by set of datapoints
// Linear interpolation is used to calculate function value between datapoints
double ZdcSD::evaluateFunction(const std::vector<double>& X, const std::vector<double>& Y, double x) {
  for (unsigned int i = 0; i < X.size() - 1; i++) {
    if (x > X[i] && x < X[i + 1]) {
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("ZdcSD") << X[i] << "\t" << Y[i] << "\t" << X[i + 1] << "\t" << Y[i + 1] << "\t" << x << "\t"
                                << linearInterpolation(X[i], Y[i], X[i + 1], Y[i + 1], x);
#endif
      return linearInterpolation(X[i], Y[i], X[i + 1], Y[i + 1], x);
    }
  }

  if (fabs(X[0] - x) < 1E-8)
    return Y[0];
  else if (fabs(X[X.size() - 1] - x) < 1E-8)
    return Y[X.size() - 1];
  else
    return NAN;
}

// Do linear interpolation between two points
double ZdcSD::linearInterpolation(double x1, double y1, double x2, double y2, double z) {
  if (x1 < x2)
    return y1 + (z - x1) * (y2 - y1) / (x2 - x1);
  else
    return y2 + (z - x2) * (y1 - y2) / (x1 - x2);
}

// Energy (eV) to wavelength (nm) conversion
double ZdcSD::convertEnergyToWavelength(double energy) { return (1240.0 / energy); }

/////////////////////////////////////////////////////////////////////

uint32_t ZdcSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme == nullptr ? 0 : numberingScheme->getUnitID(aStep));
}

int ZdcSD::setTrackID(const G4Step* aStep) {
  auto const theTrack = aStep->GetTrack();
  TrackInformation* trkInfo = (TrackInformation*)(theTrack->GetUserInformation());
  int primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
#ifdef EDM_ML_DEBUG
    auto const preStepPoint = aStep->GetPreStepPoint();
    double etrack = preStepPoint->GetKineticEnergy();
    edm::LogVerbatim("ZdcSD") << "ZdcSD: Problem with primaryID **** set by force to TkID **** "
                              << theTrack->GetTrackID() << " E " << etrack;
#endif
    primaryID = theTrack->GetTrackID();
  }
  if (primaryID != previousID[0].trackID())
    resetForNewPrimary(aStep);
  return primaryID;
}
