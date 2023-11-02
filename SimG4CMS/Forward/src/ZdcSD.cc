///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.cc
// Date: 03.01
// Description: Sensitive Detector class for Zdc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include <memory>

#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

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
  zdcHitEnergyCut = m_ZdcSD.getParameter<double>("ZdcHitEnergyCut") * GeV;
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
  NaNTrap(aStep);

  /*
    if (useShowerLibrary)
    getFromLibrary(aStep);
  */
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
    if (G4TrackToParticleID::isGammaElectronPositron(theTrack)) {
      edepositEM = energy;
      edepositHAD = 0;
    } else {
      edepositEM = 0;
      edepositHAD = energy;
    }
    if (!hitExists(aStep, 0) && edepositEM + edepositHAD > 0.) {
#ifdef EDM_ML_DEBUG
      G4ThreeVector pre = aStep->GetPreStepPoint()->GetPosition();
      edm::LogVerbatim("ZdcSD") << pre.x() << " " << pre.y() << " " << pre.z();
#endif
      currentHit[0] = CaloSD::createNewHit(aStep, theTrack, 0);
    }
  }
  return true;
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
  G4Material* material = aStep->GetTrack()->GetMaterial();

  if (material->GetName() != "quartz")
    return 0.0;  // 0 deposit if material is not quartz
  else {
    const G4StepPoint* pPreStepPoint = aStep->GetPreStepPoint();
    const G4StepPoint* pPostStepPoint = aStep->GetPostStepPoint();
    const G4String volumeName = pPreStepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume()->GetName();

    G4ThreeVector pre = pPreStepPoint->GetPosition();
    G4ThreeVector post = pPostStepPoint->GetPosition();

    if ((post - pre).mag() < 1E-9)
      return 0.0;

    //Convert step coordinates to local (fiber) coodinates
    const G4ThreeVector localPre = setToLocal(pre, pPreStepPoint->GetTouchable());
    const G4ThreeVector localPost = setToLocal(post, pPreStepPoint->GetTouchable());
    // Calculate the unit direction vector in local coordinates

    const G4ThreeVector particleDirection = (localPost - localPre) / (localPost - localPre).mag();

    const G4DynamicParticle* aParticle = aStep->GetTrack()->GetDynamicParticle();
    int charge = round(aParticle->GetDefinition()->GetPDGCharge());

    if (charge == 0)
      return 0.0;

    double beta = 0.5 * (pPreStepPoint->GetBeta() + pPostStepPoint->GetBeta());
    double stepLength = aStep->GetStepLength() / 1000;  // Geant4 stepLength is in "mm"

    int nPhotons;  // Number of Cherenkov photons

    nPhotons = G4Poisson(calculateMeanNumberOfPhotons(charge, beta, stepLength));

    double totalE = 0.0;

    for (int i = 0; i < nPhotons; i++) {
      // uniform refractive index in PMT range -> uniform energy distribution
      double photonE = EMIN + G4UniformRand() * (EMAX - EMIN);
      // UPDATE: taking into account dispersion relation -> energy distribution

      if (G4UniformRand() > pmtEfficiency(convertEnergyToWavelength(photonE)))
        continue;

      double omega = G4UniformRand() * twopi;
      double thetaC = acos(1.0 / (beta * RINDEX));

#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("ZdcSD") << "E_gamma: " << photonE << "\t omega: " << omega << "\t thetaC: " << thetaC;
#endif
      // Calculate momentum direction w.r.t primary particle (z-direction)
      double px = photonE * sin(thetaC) * cos(omega);
      double py = photonE * sin(thetaC) * sin(omega);
      double pz = photonE * cos(thetaC);
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
      edm::LogVerbatim("ZdcSD") << "r = (" << photonPosition.x() << "," << photonPosition.y() << ","
                                << photonPosition.z() << ")" << std::endl;
#endif
    }

#ifdef EDM_ML_DEBUG
    if (nPhotons > 30) {
      edm::LogVerbatim("ZdcSD") << totalE;

      if (totalE > 0)
        edm::LogVerbatim("ZdcSD") << pre.x() << " " << pre.y() << " " << pre.z() << " " << totalE << std::endl;
    }
#endif
    return totalE;
  }
}

// Calculate mean number of Cherenkov photons in the sensitivity range (300-650 nm)
// for a given step length for a particle with given charge and beta
double ZdcSD::calculateMeanNumberOfPhotons(int charge, double beta, double stepLength) {
  // Return mean number of Cherenkov photons
  return (ALPHA * charge * charge * stepLength) / HBARC * (EMAX - EMIN) * (1.0 - 1.0 / (beta * beta * RINDEX * RINDEX));
}

// Evaluate photon pdf
double ZdcSD::photonEnergyDist(int charge, double beta, double E) {
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
double ZdcSD::generatePhotonEnergy(int charge, double beta, double Emin) {
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
