///////////////////////////////////////////////////////////////////////////////
// File: ZdcSD.cc
// Date: 03.01
// Description: Sensitive Detector class for Zdc
// Modifications:
///////////////////////////////////////////////////////////////////////////////
#include <memory>

#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"
#include "G4ios.hh"
#include "G4Cerenkov.hh"
#include "G4ParticleTable.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "Randomize.hh"
#include "G4Poisson.hh"

ZdcSD::ZdcSD(const std::string& name,
             const edm::EventSetup& es,
             const SensitiveDetectorCatalog& clg,
             edm::ParameterSet const& p,
             const SimTrackManager* manager)
    : CaloSD(name, es, clg, p, manager) {
  edm::ParameterSet m_ZdcSD = p.getParameter<edm::ParameterSet>("ZdcSD");
  useShowerLibrary = m_ZdcSD.getParameter<bool>("UseShowerLibrary");
  useShowerHits = m_ZdcSD.getParameter<bool>("UseShowerHits");
  zdcHitEnergyCut = m_ZdcSD.getParameter<double>("ZdcHitEnergyCut") * GeV;
  thFibDir = m_ZdcSD.getParameter<double>("FiberDirection");
  verbosity = m_ZdcSD.getParameter<int>("Verbosity");
  int verbn = verbosity / 10;
  verbosity %= 10;
  setNumberingScheme(new ZdcNumberingScheme(verbn));

  edm::LogVerbatim("ZdcSD") << "***************************************************\n"
                            << "*                                                 *\n"
                            << "* Constructing a ZdcSD  with name " << name << "   *\n"
                            << "*                                                 *\n"
                            << "***************************************************";

  edm::LogVerbatim("ZdcSD") << "\nUse of shower library is set to " << useShowerLibrary
                            << "\nUse of Shower hits method is set to " << useShowerHits;

  edm::LogVerbatim("ZdcSD") << "\nEnergy Threshold Cut set to " << zdcHitEnergyCut / GeV << " (GeV)";

  if (useShowerLibrary) {
    showerLibrary = std::make_unique<ZdcShowerLibrary>(name, p);
    setParameterized(true);
  } else {
    showerLibrary.reset(nullptr);
  }
}

void ZdcSD::initRun() { hits.clear(); }

bool ZdcSD::getFromLibrary(const G4Step* aStep) {
  bool ok = true;

  auto const preStepPoint = aStep->GetPreStepPoint();
  auto const theTrack = aStep->GetTrack();

  double etrack = preStepPoint->GetKineticEnergy();
  int primaryID = setTrackID(aStep);

  hits.clear();

  // Reset entry point for new primary
  resetForNewPrimary(aStep);

  if (etrack >= zdcHitEnergyCut) {
    // create hits only if above threshold

    LogDebug("ForwardSim") << "----------------New track------------------------------\n"
                           << "Incident EnergyTrack: " << etrack << " MeV \n"
                           << "Zdc Cut Energy for Hits: " << zdcHitEnergyCut << " MeV \n"
                           << "ZdcSD::getFromLibrary " << hits.size() << " hits for " << GetName() << " of "
                           << primaryID << " with " << theTrack->GetDefinition()->GetParticleName() << " of " << etrack
                           << " MeV\n";

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
    currentID.setID(unitID, time, primaryID, 0);
    processHit(aStep);

    LogDebug("ForwardSim") << "ZdcSD: Final Hit number:" << i << "-->"
                           << "New HitID: " << currentHit->getUnitID()
                           << " New Hit trackID: " << currentHit->getTrackID()
                           << " New EM Energy: " << currentHit->getEM() / GeV
                           << " New HAD Energy: " << currentHit->getHadr() / GeV
                           << " New HitEntryPoint: " << currentHit->getEntryLocal()
                           << " New IncidentEnergy: " << currentHit->getIncidentEnergy() / GeV
                           << " New HitPosition: " << posGlobal;
  }
  return ok;
}

double ZdcSD::getEnergyDeposit(const G4Step* aStep) {
  double NCherPhot = 0.;

  // preStepPoint information
  G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  G4VPhysicalVolume* currentPV = preStepPoint->GetPhysicalVolume();
  const G4String& nameVolume = currentPV->GetName();

  const G4ThreeVector& hitPoint = preStepPoint->GetPosition();
  const G4ThreeVector& hit_mom = preStepPoint->GetMomentumDirection();
  G4double stepL = aStep->GetStepLength() / cm;
  G4double beta = preStepPoint->GetBeta();
  G4double charge = preStepPoint->GetCharge();

  // postStepPoint information
  G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
  G4VPhysicalVolume* postPV = postStepPoint->GetPhysicalVolume();
  const G4String& postnameVolume = postPV->GetName();

  // theTrack information
  G4Track* theTrack = aStep->GetTrack();
  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  const G4ThreeVector& vert_mom = theTrack->GetVertexMomentumDirection();
  G4ThreeVector localPoint = theTrack->GetTouchable()->GetHistory()->GetTopTransform().TransformPoint(hitPoint);

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
  LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit: \n"
                         << "  preStepPoint: " << nameVolume << "," << stepL << "," << stepE << "," << beta << ","
                         << charge << "\n"
                         << "  postStepPoint: " << postnameVolume << "," << costheta << "," << theta << "," << eta
                         << "," << phi << "," << particleType << " id= " << theTrack->GetTrackID()
                         << " Etot(GeV)= " << theTrack->GetTotalEnergy() / GeV;

  const double bThreshold = 0.67;
  if ((beta > bThreshold) && (charge != 0) && (nameVolume == "ZDC_EMFiber" || nameVolume == "ZDC_HadFiber")) {
    LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit:  pass ";

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
      th += twopi;

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
    float variant = -1;

    // if (d > (r+a))
    if (DelFibPart > (thFullReflRad + thcher)) {
      variant = 0.;
      d_qz = 0.;
    } else {
      // if ((DelFibPart + thcher) < thFullReflRad )  [(d+r) < a]
      if ((th + thcher) < (thFibDirRad + thFullReflRad) && (th - thcher) > (thFibDirRad - thFullReflRad)) {
        variant = 1.;
        d_qz = 1.;
      } else {
        // if ((thcher - DelFibPart ) > thFullReflRad )  [(r-d) > a]
        if ((thFibDirRad + thFullReflRad) < (th + thcher) && (thFibDirRad - thFullReflRad) > (th - thcher)) {
          variant = 2.;
          d_qz = 0.;
        } else {
          variant = 3.;  // d_qz is calculated below

          // use crossed length of circles(cone projection) - dC1/dC2 :
          float arg_arcos = 0.;
          float tan_arcos = 2. * a * d;
          if (tan_arcos != 0.)
            arg_arcos = (r * r - a * a - d * d) / tan_arcos;
          // std::cout.testOut << "  d_qz: " << r << "," << a << "," << d << " " << tan_arcos << " " << arg_arcos;
          arg_arcos = std::abs(arg_arcos);
          // std::cout.testOut << "," << arg_arcos;
          float th_arcos = acos(std::min(std::max(arg_arcos, -1.f), 1.f));
          // std::cout.testOut << " " << th_arcos;
          d_qz = th_arcos / twopi;
          // std::cout.testOut << " " << d_qz;
          d_qz = std::abs(d_qz);
          // std::cout.testOut << "," << d_qz;
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

    LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit:  gED: " << stepE << "," << costh << "," << th << ","
                           << costhcher << "," << thcher << "," << DelFibPart << "," << d << "," << a << "," << r << ","
                           << hitPoint << "," << hit_mom << "," << vert_mom << "," << localPoint << "," << charge << ","
                           << beta << "," << stepL << "," << d_qz << "," << variant << "," << meanNCherPhot << ","
                           << poissNCherPhot << "," << NCherPhot;
    // --constants-----------------
    // << "," << photEnSpectrDE
    // << "," << nMedium
    // << "," << bThreshold
    // << "," << thFibDirRad
    // << "," << thFullReflRad
    // << "," << effPMTandTransport
    // --other variables-----------
    // << "," << curprocess
    // << "," << nameProcess
    // << "," << name
    // << "," << rad
    // << "," << mat

  } else {
    // determine failure mode: beta, charge, and/or nameVolume
    if (beta <= bThreshold)
      LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit: fail beta=" << beta;
    if (charge == 0)
      LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit: fail charge=0";
    if (!(nameVolume == "ZDC_EMFiber" || nameVolume == "ZDC_HadFiber"))
      LogDebug("ForwardSim") << "ZdcSD::  getEnergyDeposit: fail nv=" << nameVolume;
  }

  return NCherPhot;
}

uint32_t ZdcSD::setDetUnitId(const G4Step* aStep) {
  return (numberingScheme.get() == nullptr ? 0 : numberingScheme.get()->getUnitID(aStep));
}

void ZdcSD::setNumberingScheme(ZdcNumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogVerbatim("ZdcSD") << "ZdcSD: updates numbering scheme for " << GetName();
    numberingScheme.reset(scheme);
  }
}
