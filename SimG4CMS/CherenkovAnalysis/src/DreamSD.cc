
#include <memory>

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDFilteredView.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4LogicalVolume.hh"
#include "G4LogicalVolumeStore.hh"
#include "G4Poisson.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "FWCore/Framework/interface/ESTransientHandle.h"

// Histogramming
#include "FWCore/ServiceRegistry/interface/Service.h"

// Cherenkov
#include "SimG4CMS/CherenkovAnalysis/interface/DreamSD.h"
#include "SimG4CMS/CherenkovAnalysis/interface/PMTResponse.h"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

//#define EDM_ML_DEBUG

//________________________________________________________________________________________
DreamSD::DreamSD(const std::string &name,
                 const edm::EventSetup &es,
                 const SensitiveDetectorCatalog &clg,
                 edm::ParameterSet const &p,
                 const SimTrackManager *manager)
    : CaloSD(name, clg, p, manager) {
  edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ECalSD");
  useBirk_ = m_EC.getParameter<bool>("UseBirkLaw");
  doCherenkov_ = m_EC.getParameter<bool>("doCherenkov");
  birk1_ = m_EC.getParameter<double>("BirkC1") * (CLHEP::g / (CLHEP::MeV * CLHEP::cm2));
  birk2_ = m_EC.getParameter<double>("BirkC2");
  birk3_ = m_EC.getParameter<double>("BirkC3");
  slopeLY_ = m_EC.getParameter<double>("SlopeLightYield");
  readBothSide_ = m_EC.getUntrackedParameter<bool>("ReadBothSide", false);
  dd4hep_ = p.getParameter<bool>("g4GeometryDD4hepSource");

  chAngleIntegrals_.reset(nullptr);

  edm::LogVerbatim("EcalSim") << "Constructing a DreamSD  with name " << GetName()
                              << "\nDreamSD:: Use of Birks law is set to      " << useBirk_
                              << "  with three constants kB = " << birk1_ << ", C1 = " << birk2_ << ", C2 = " << birk3_
                              << "\n          Slope for Light yield is set to " << slopeLY_
                              << "\n          Parameterization of Cherenkov is set to " << doCherenkov_
                              << ", readout both sides is " << readBothSide_ << " and dd4hep flag " << dd4hep_;
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << GetName() << " initialized";
  const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
  unsigned int k(0);
  for (auto lvcite = lvs->begin(); lvcite != lvs->end(); ++lvcite, ++k)
    edm::LogVerbatim("EcalSim") << "Volume[" << k << "] " << (*lvcite)->GetName();
#endif
  initMap(name, es);
}

//________________________________________________________________________________________
double DreamSD::getEnergyDeposit(const G4Step *aStep) {
  // take into account light collection curve for crystals
  double weight = curve_LY(aStep, side_);
  if (useBirk_)
    weight *= getAttenuation(aStep, birk1_, birk2_, birk3_);
  double edep = aStep->GetTotalEnergyDeposit() * weight;

  // Get Cerenkov contribution
  if (doCherenkov_) {
    edep += cherenkovDeposit_(aStep);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "DreamSD:: " << aStep->GetPreStepPoint()->GetPhysicalVolume()->GetName() << " Side "
                              << side_ << " Light Collection Efficiency " << weight << " Weighted Energy Deposit "
                              << edep / CLHEP::MeV << " MeV";
#endif
  return edep;
}

//________________________________________________________________________________________
void DreamSD::initRun() {
  // Get the material and set properties if needed
  DimensionMap::const_iterator ite = xtalLMap_.begin();
  const G4LogicalVolume *lv = (ite->first);
  G4Material *material = lv->GetMaterial();
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "DreamSD::initRun: Initializes for material " << material->GetName() << " in "
                              << lv->GetName();
#endif
  materialPropertiesTable_ = material->GetMaterialPropertiesTable();
  if (!materialPropertiesTable_) {
    if (!setPbWO2MaterialProperties_(material)) {
      edm::LogWarning("EcalSim") << "Couldn't retrieve material properties table\n Material = " << material->GetName();
    }
    materialPropertiesTable_ = material->GetMaterialPropertiesTable();
  }
}

//________________________________________________________________________________________
uint32_t DreamSD::setDetUnitId(const G4Step *aStep) {
  const G4VTouchable *touch = aStep->GetPreStepPoint()->GetTouchable();
  uint32_t id = (touch->GetReplicaNumber(1)) * 10 + (touch->GetReplicaNumber(0));
  side_ = readBothSide_ ? -1 : 1;
  if (side_ < 0) {
    ++id;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "DreamSD:: ID " << id;
#endif
  return id;
}

//________________________________________________________________________________________
void DreamSD::initMap(const std::string &sd, const edm::EventSetup &es) {
  if (dd4hep_) {
    edm::ESTransientHandle<cms::DDCompactView> cpv;
    es.get<IdealGeometryRecord>().get(cpv);
    const cms::DDFilter filter("ReadOutName", sd);
    cms::DDFilteredView fv((*cpv), filter);
    while (fv.firstChild()) {
      std::string name = static_cast<std::string>(dd4hep::dd::noNamespace(fv.name()));
      std::vector<double> paras(fv.parameters());
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalSim") << "DreamSD::initMap (for " << sd << "): Solid " << name << " Shape "
                                  << cms::dd::name(cms::DDSolidShapeMap, fv.shape()) << " Parameter 0 = " << paras[0];
#endif
      // Set length to be the largest size, width the smallest
      std::sort(paras.begin(), paras.end());
      double length = 2.0 * k_ScaleFromDD4HepToG4 * paras.back();
      double width = 2.0 * k_ScaleFromDD4HepToG4 * paras.front();
      fillMap(name, length, width);
    }
  } else {
    edm::ESTransientHandle<DDCompactView> cpv;
    es.get<IdealGeometryRecord>().get(cpv);
    DDSpecificsMatchesValueFilter filter{DDValue("ReadOutName", sd, 0)};
    DDFilteredView fv((*cpv), filter);
    fv.firstChild();
    bool dodet = true;
    const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
    while (dodet) {
      const DDSolid &sol = fv.logicalPart().solid();
      std::vector<double> paras(sol.parameters());
      G4String name = sol.name().name();
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalSim") << "DreamSD::initMap (for " << sd << "): Solid " << name << " Shape " << sol.shape()
                                  << " Parameter 0 = " << paras[0];
#endif
      // Set length to be the largest size, width the smallest
      std::sort(paras.begin(), paras.end());
      double length = 2.0 * k_ScaleFromDDDToG4 * paras.back();
      double width = 2.0 * k_ScaleFromDDDToG4 * paras.front();
      G4LogicalVolume *lv = nullptr;
      for (auto lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++)
        if ((*lvcite)->GetName() == name) {
          lv = (*lvcite);
          break;
        }
      xtalLMap_.insert(std::pair<G4LogicalVolume *, Doubles>(lv, Doubles(length, width)));
#ifdef EDM_ML_DEBUG
      edm::LogVerbatim("EcalSim") << "DreamSD " << name << ":" << lv << ":" << length << ":" << width;
#endif
      dodet = fv.next();
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "DreamSD: Length Table for ReadOutName = " << sd << ":";
#endif
  DimensionMap::const_iterator ite = xtalLMap_.begin();
  int i = 0;
  for (; ite != xtalLMap_.end(); ite++, i++) {
    G4String name = "Unknown";
    if (ite->first != nullptr)
      name = (ite->first)->GetName();
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalSim") << " " << i << " " << ite->first << " " << name << " L = " << ite->second.first
                                << " W = " << ite->second.second;
#endif
  }
}

//________________________________________________________________________________________
void DreamSD::fillMap(const std::string &name, double length, double width) {
  const G4LogicalVolumeStore *lvs = G4LogicalVolumeStore::GetInstance();
  edm::LogVerbatim("EcalSim") << "LV Store with " << lvs->size() << " elements";
  G4LogicalVolume *lv = nullptr;
  for (auto lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) {
    edm::LogVerbatim("EcalSim") << name << " vs " << (*lvcite)->GetName();
    if ((*lvcite)->GetName() == static_cast<G4String>(name)) {
      lv = (*lvcite);
      break;
    }
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "DreamSD::fillMap (for " << name << " Logical Volume " << lv << " Length " << length
                              << " Width " << width;
#endif
  xtalLMap_.insert(std::pair<G4LogicalVolume *, Doubles>(lv, Doubles(length, width)));
}

//________________________________________________________________________________________
double DreamSD::curve_LY(const G4Step *aStep, int flag) {
  auto const stepPoint = aStep->GetPreStepPoint();
  auto const lv = stepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  G4String nameVolume = lv->GetName();

  double weight = 1.;
  G4ThreeVector localPoint = setToLocal(stepPoint->GetPosition(), stepPoint->GetTouchable());
  double crlength = crystalLength(lv);
  double localz = localPoint.x();
  double dapd = 0.5 * crlength - flag * localz;  // Distance from closest APD
  if (dapd >= -0.1 || dapd <= crlength + 0.1) {
    if (dapd <= 100.)
      weight = 1.0 + slopeLY_ - dapd * 0.01 * slopeLY_;
  } else {
    edm::LogWarning("EcalSim") << "DreamSD: light coll curve : wrong distance "
                               << "to APD " << dapd << " crlength = " << crlength << " crystal name = " << nameVolume
                               << " z of localPoint = " << localz << " take weight = " << weight;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "DreamSD, light coll curve : " << dapd << " crlength = " << crlength
                              << " crystal name = " << nameVolume << " z of localPoint = " << localz
                              << " take weight = " << weight;
#endif
  return weight;
}

//________________________________________________________________________________________
double DreamSD::crystalLength(G4LogicalVolume *lv) const {
  double length = -1.;
  DimensionMap::const_iterator ite = xtalLMap_.find(lv);
  if (ite != xtalLMap_.end())
    length = ite->second.first;
  return length;
}

//________________________________________________________________________________________
double DreamSD::crystalWidth(G4LogicalVolume *lv) const {
  double width = -1.;
  DimensionMap::const_iterator ite = xtalLMap_.find(lv);
  if (ite != xtalLMap_.end())
    width = ite->second.second;
  return width;
}

//________________________________________________________________________________________
// Calculate total cherenkov deposit
// Inspired by Geant4's Cherenkov implementation
double DreamSD::cherenkovDeposit_(const G4Step *aStep) {
  double cherenkovEnergy = 0;
  if (!materialPropertiesTable_)
    return cherenkovEnergy;
  G4Material *material = aStep->GetTrack()->GetMaterial();

  // Retrieve refractive index
  G4MaterialPropertyVector *Rindex = materialPropertiesTable_->GetProperty("RINDEX");
  if (Rindex == nullptr) {
    edm::LogWarning("EcalSim") << "Couldn't retrieve refractive index";
    return cherenkovEnergy;
  }

  // V.Ivanchenko - temporary close log output for 9.5
  // Material refraction properties
  int Rlength = Rindex->GetVectorLength() - 1;
  double Pmin = Rindex->Energy(0);
  double Pmax = Rindex->Energy(Rlength);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "Material properties: \n  Pmin = " << Pmin << "  Pmax = " << Pmax;
#endif
  // Get particle properties
  const G4StepPoint *pPreStepPoint = aStep->GetPreStepPoint();
  const G4StepPoint *pPostStepPoint = aStep->GetPostStepPoint();
  const G4ThreeVector &x0 = pPreStepPoint->GetPosition();
  G4ThreeVector p0 = aStep->GetDeltaPosition().unit();
  const G4DynamicParticle *aParticle = aStep->GetTrack()->GetDynamicParticle();
  const double charge = aParticle->GetDefinition()->GetPDGCharge();
  // beta is averaged over step
  double beta = 0.5 * (pPreStepPoint->GetBeta() + pPostStepPoint->GetBeta());
  double BetaInverse = 1.0 / beta;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "Particle properties: \n  charge = " << charge << "  beta   = " << beta;
#endif

  // Now get number of photons generated in this step
  double meanNumberOfPhotons = getAverageNumberOfPhotons_(charge, beta, material, Rindex);
  if (meanNumberOfPhotons <= 0.0) {  // Don't do anything
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalSim") << "Mean number of photons is zero: " << meanNumberOfPhotons << ", stopping here";
#endif
    return cherenkovEnergy;
  }

  // number of photons is in unit of Geant4...
  meanNumberOfPhotons *= aStep->GetStepLength();

  // Now get a poisson distribution
  int numPhotons = static_cast<int>(G4Poisson(meanNumberOfPhotons));
  // edm::LogVerbatim("EcalSim") << "Number of photons = " << numPhotons;
  if (numPhotons <= 0) {
#ifdef EDM_ML_DEBUG
    edm::LogVerbatim("EcalSim") << "Poission number of photons is zero: " << numPhotons << ", stopping here";
#endif
    return cherenkovEnergy;
  }

  // Material refraction properties
  double dp = Pmax - Pmin;
  double maxCos = BetaInverse / (*Rindex)[Rlength];
  double maxSin2 = (1.0 - maxCos) * (1.0 + maxCos);

  // Finally: get contribution of each photon
  for (int iPhoton = 0; iPhoton < numPhotons; ++iPhoton) {
    // Determine photon momentum
    double randomNumber;
    double sampledMomentum, sampledRI;
    double cosTheta, sin2Theta;

    // sample a momentum (not sure why this is needed!)
    do {
      randomNumber = G4UniformRand();
      sampledMomentum = Pmin + randomNumber * dp;
      sampledRI = Rindex->Value(sampledMomentum);
      cosTheta = BetaInverse / sampledRI;

      sin2Theta = (1.0 - cosTheta) * (1.0 + cosTheta);
      randomNumber = G4UniformRand();

    } while (randomNumber * maxSin2 > sin2Theta);

    // Generate random position of photon on cone surface
    // defined by Theta
    randomNumber = G4UniformRand();

    double phi = twopi * randomNumber;
    double sinPhi = sin(phi);
    double cosPhi = cos(phi);

    // Create photon momentum direction vector
    // The momentum direction is still w.r.t. the coordinate system where the
    // primary particle direction is aligned with the z axis
    double sinTheta = sqrt(sin2Theta);
    double px = sinTheta * cosPhi;
    double py = sinTheta * sinPhi;
    double pz = cosTheta;
    G4ThreeVector photonDirection(px, py, pz);

    // Rotate momentum direction back to global (crystal) reference system
    photonDirection.rotateUz(p0);

    // Create photon position and momentum
    randomNumber = G4UniformRand();
    G4ThreeVector photonPosition = x0 + randomNumber * aStep->GetDeltaPosition();
    G4ThreeVector photonMomentum = sampledMomentum * photonDirection;

    // Collect energy on APD
    cherenkovEnergy += getPhotonEnergyDeposit_(photonMomentum, photonPosition, aStep);
  }
  return cherenkovEnergy;
}

//________________________________________________________________________________________
// Returns number of photons produced per GEANT-unit (millimeter) in the current
// medium. From G4Cerenkov.cc
double DreamSD::getAverageNumberOfPhotons_(const double charge,
                                           const double beta,
                                           const G4Material *aMaterial,
                                           const G4MaterialPropertyVector *Rindex) {
  const double rFact = 369.81 / (eV * cm);

  if (beta <= 0.0)
    return 0.0;

  double BetaInverse = 1. / beta;

  // Vectors used in computation of Cerenkov Angle Integral:
  //         - Refraction Indices for the current material
  //        - new G4PhysicsOrderedFreeVector allocated to hold CAI's

  // Min and Max photon momenta
  int Rlength = Rindex->GetVectorLength() - 1;
  double Pmin = Rindex->Energy(0);
  double Pmax = Rindex->Energy(Rlength);

  // Min and Max Refraction Indices
  double nMin = (*Rindex)[0];
  double nMax = (*Rindex)[Rlength];

  // Max Cerenkov Angle Integral
  double CAImax = chAngleIntegrals_.get()->GetMaxEnergy();

  double dp = 0., ge = 0., CAImin = 0.;

  // If n(Pmax) < 1/Beta -- no photons generated
  if (nMax < BetaInverse) {
  }

  // otherwise if n(Pmin) >= 1/Beta -- photons generated
  else if (nMin > BetaInverse) {
    dp = Pmax - Pmin;
    ge = CAImax;
  }
  // If n(Pmin) < 1/Beta, and n(Pmax) >= 1/Beta, then
  // we need to find a P such that the value of n(P) == 1/Beta.
  // Interpolation is performed by the GetPhotonEnergy() and
  // GetProperty() methods of the G4MaterialPropertiesTable and
  // the GetValue() method of G4PhysicsVector.
  else {
    Pmin = Rindex->Value(BetaInverse);
    dp = Pmax - Pmin;
    // need boolean for current implementation of G4PhysicsVector
    // ==> being phased out
    double CAImin = chAngleIntegrals_->Value(Pmin);
    ge = CAImax - CAImin;
  }

  // Calculate number of photons
  double numPhotons = rFact * charge / eplus * charge / eplus * (dp - ge * BetaInverse * BetaInverse);

  edm::LogVerbatim("EcalSim") << "@SUB=getAverageNumberOfPhotons\nCAImin = " << CAImin << "\nCAImax = " << CAImax
                              << "\ndp = " << dp << ", ge = " << ge << "\nnumPhotons = " << numPhotons;
  return numPhotons;
}

//________________________________________________________________________________________
// Set lead tungstate material properties on the fly.
// Values from Ts42 detector construction
bool DreamSD::setPbWO2MaterialProperties_(G4Material *aMaterial) {
  std::string pbWO2Name("E_PbWO4");
  if (pbWO2Name != aMaterial->GetName()) {  // Wrong material!
    edm::LogWarning("EcalSim") << "This is not the right material: "
                               << "expecting " << pbWO2Name << ", got " << aMaterial->GetName();
    return false;
  }

  G4MaterialPropertiesTable *table = new G4MaterialPropertiesTable();

  // Refractive index as a function of photon momentum
  // FIXME: Should somehow put that in the configuration
  const int nEntries = 14;
  double PhotonEnergy[nEntries] = {1.7712 * eV,
                                   1.8368 * eV,
                                   1.90745 * eV,
                                   1.98375 * eV,
                                   2.0664 * eV,
                                   2.15625 * eV,
                                   2.25426 * eV,
                                   2.3616 * eV,
                                   2.47968 * eV,
                                   2.61019 * eV,
                                   2.75521 * eV,
                                   2.91728 * eV,
                                   3.09961 * eV,
                                   3.30625 * eV};
  double RefractiveIndex[nEntries] = {2.17728,
                                      2.18025,
                                      2.18357,
                                      2.18753,
                                      2.19285,
                                      2.19813,
                                      2.20441,
                                      2.21337,
                                      2.22328,
                                      2.23619,
                                      2.25203,
                                      2.27381,
                                      2.30282,
                                      2.34666};

  table->AddProperty("RINDEX", PhotonEnergy, RefractiveIndex, nEntries);
  aMaterial->SetMaterialPropertiesTable(table);  // FIXME: could this leak? What does G4 do?

  // Calculate Cherenkov angle integrals:
  // This is an ad-hoc solution (we hold it in the class, not in the material)
  chAngleIntegrals_ = std::make_unique<G4PhysicsFreeVector>();

  int index = 0;
  double currentRI = RefractiveIndex[index];
  double currentPM = PhotonEnergy[index];
  double currentCAI = 0.0;
  chAngleIntegrals_.get()->PutValue(0, currentPM, currentCAI);
  double prevPM = currentPM;
  double prevCAI = currentCAI;
  double prevRI = currentRI;
  while (++index < nEntries) {
    currentRI = RefractiveIndex[index];
    currentPM = PhotonEnergy[index];
    currentCAI = 0.5 * (1.0 / (prevRI * prevRI) + 1.0 / (currentRI * currentRI));
    currentCAI = prevCAI + (currentPM - prevPM) * currentCAI;

    chAngleIntegrals_.get()->PutValue(index, currentPM, currentCAI);

    prevPM = currentPM;
    prevCAI = currentCAI;
    prevRI = currentRI;
  }

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "Material properties set for " << aMaterial->GetName();
#endif
  return true;
}

//________________________________________________________________________________________
// Calculate energy deposit of a photon on APD
// - simple tracing to APD position (straight line);
// - configurable reflection probability if not straight to APD;
// - APD response function
double DreamSD::getPhotonEnergyDeposit_(const G4ThreeVector &p, const G4ThreeVector &x, const G4Step *aStep) {
  double energy = 0;

  // Crystal dimensions

  // edm::LogVerbatim("EcalSim") << p << x;

  // 1. Check if this photon goes straight to the APD:
  //    - assume that APD is at x=xtalLength/2.0
  //    - extrapolate from x=x0 to x=xtalLength/2.0 using momentum in x-y

  G4StepPoint *stepPoint = aStep->GetPreStepPoint();
  G4LogicalVolume *lv = stepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  G4String nameVolume = lv->GetName();

  double crlength = crystalLength(lv);
  double crwidth = crystalWidth(lv);
  double dapd = 0.5 * crlength - x.x();  // Distance from closest APD
  double y = p.y() / p.x() * dapd;

#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "Distance to APD: " << dapd << " - y at APD: " << y;
#endif
  // Not straight: compute probability
  if (std::abs(y) > crwidth * 0.5) {
  }

  // 2. Retrieve efficiency for this wavelength (in nm, from MeV)
  double waveLength = p.mag() * 1.239e8;

  energy = p.mag() * PMTResponse::getEfficiency(waveLength);
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "Wavelength: " << waveLength << " - Energy: " << energy;
#endif
  return energy;
}
