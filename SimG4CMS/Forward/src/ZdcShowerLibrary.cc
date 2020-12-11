///////////////////////////////////////////////////////////////////////////////
// File: ZdcShowerLibrary.cc
// Description: Shower library for the Zero Degree Calorimeter
// E. Garcia June 2008
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Forward/interface/ZdcSD.h"
#include "SimG4CMS/Forward/interface/ZdcShowerLibrary.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "Randomize.hh"
#include "CLHEP/Units/GlobalSystemOfUnits.h"

ZdcShowerLibrary::ZdcShowerLibrary(const std::string& name, edm::ParameterSet const& p) {
  edm::ParameterSet m_HS = p.getParameter<edm::ParameterSet>("ZdcShowerLibrary");
  verbose = m_HS.getUntrackedParameter<int>("Verbosity", 0);

  npe = 9;  // number of channels or fibers where the energy will be deposited
  hits.reserve(npe);
}

ZdcShowerLibrary::~ZdcShowerLibrary() {}

std::vector<ZdcShowerLibrary::Hit>& ZdcShowerLibrary::getHits(const G4Step* aStep, bool& ok) {
  const G4StepPoint* preStepPoint = aStep->GetPreStepPoint();
  const G4StepPoint* postStepPoint = aStep->GetPostStepPoint();
  const G4Track* track = aStep->GetTrack();

  const G4DynamicParticle* aParticle = track->GetDynamicParticle();
  const G4ThreeVector& momDir = aParticle->GetMomentumDirection();
  double energy = preStepPoint->GetKineticEnergy();
  G4ThreeVector hitPoint = preStepPoint->GetPosition();
  const G4ThreeVector& hitPointOrig = preStepPoint->GetPosition();
  G4int parCode = track->GetDefinition()->GetPDGEncoding();

  hits.clear();

  ok = false;
  bool isEM = G4TrackToParticleID::isGammaElectronPositron(parCode);
  bool isHad = G4TrackToParticleID::isStableHadronIon(track);
  if (!isEM && !isHad)
    return hits;
  ok = true;

  G4ThreeVector pos;
  G4ThreeVector posLocal;
  double tSlice = (postStepPoint->GetGlobalTime()) / nanosecond;

  int nHit = 0;
  HcalZDCDetId::Section section;
  bool side = false;
  int channel = 0;
  double xx, yy, zz;
  double xxlocal, yylocal, zzlocal;

  ZdcShowerLibrary::Hit oneHit;
  side = (hitPointOrig.z() > 0.) ? true : false;

  float xWidthEM = std::abs(theXChannelBoundaries[0] - theXChannelBoundaries[1]);
  float zWidthEM = std::abs(theZSectionBoundaries[0] - theZSectionBoundaries[1]);
  float zWidthHAD = std::abs(theZHadChannelBoundaries[0] - theZHadChannelBoundaries[1]);

  for (int i = 0; i < npe; i++) {
    if (i < 5) {
      section = HcalZDCDetId::EM;
      channel = i + 1;
      xxlocal = theXChannelBoundaries[i] + (xWidthEM / 2.);
      xx = xxlocal + X0;
      yy = 0.0;
      yylocal = yy + Y0;
      zzlocal = theZSectionBoundaries[0] + (zWidthEM / 2.);
      zz = (hitPointOrig.z() > 0.) ? zzlocal + Z0 : zzlocal - Z0;
      pos = G4ThreeVector(xx, yy, zz);
      posLocal = G4ThreeVector(xxlocal, yylocal, zzlocal);
    }
    if (i > 4) {
      section = HcalZDCDetId::HAD;
      channel = i - 4;
      xxlocal = 0.0;
      xx = xxlocal + X0;
      yylocal = 0;
      yy = yylocal + Y0;
      zzlocal = (hitPointOrig.z() > 0.) ? theZHadChannelBoundaries[i - 5] + (zWidthHAD / 2.)
                                        : theZHadChannelBoundaries[i - 5] - (zWidthHAD / 2.);
      zz = (hitPointOrig.z() > 0.) ? zzlocal + Z0 : zzlocal - Z0;
      pos = G4ThreeVector(xx, yy, zz);
      posLocal = G4ThreeVector(xxlocal, yylocal, zzlocal);
    }

    oneHit.position = pos;
    oneHit.entryLocal = posLocal;
    oneHit.depth = channel;
    oneHit.time = tSlice;
    oneHit.detID = HcalZDCDetId(section, side, channel);

    // Note: coodinates of hit are relative to center of detector (X0,Y0,Z0)
    hitPoint.setX(hitPointOrig.x() - X0);
    hitPoint.setY(hitPointOrig.y() - Y0);
    double setZ = (hitPointOrig.z() > 0.) ? hitPointOrig.z() - Z0 : fabs(hitPointOrig.z()) - Z0;
    hitPoint.setZ(setZ);

    int dE = getEnergyFromLibrary(hitPoint, momDir, energy, parCode, section, side, channel);

    if (isEM) {
      oneHit.DeEM = dE;
      oneHit.DeHad = 0.;
    } else {
      oneHit.DeEM = 0;
      oneHit.DeHad = dE;
    }

    hits.push_back(oneHit);

    LogDebug("ZdcShower") << "\nZdcShowerLibrary:Generated Hit " << nHit << " orig hit pos " << hitPointOrig
                          << " orig hit pos local coord" << hitPoint << " new position " << (hits[nHit].position)
                          << " Channel " << (hits[nHit].depth) << " side " << side << " Time " << (hits[nHit].time)
                          << " DetectorID " << (hits[nHit].detID) << " Had Energy " << (hits[nHit].DeHad)
                          << " EM Energy  " << (hits[nHit].DeEM) << "\n";
    nHit++;
  }
  return hits;
}

int ZdcShowerLibrary::getEnergyFromLibrary(const G4ThreeVector& hitPoint,
                                           const G4ThreeVector& momDir,
                                           double energy,
                                           G4int parCode,
                                           HcalZDCDetId::Section section,
                                           bool side,
                                           int channel) {
  int nphotons = -1;

  energy = energy / GeV;

  LogDebug("ZdcShower") << "\n ZdcShowerLibrary::getEnergyFromLibrary input/output variables:"
                        << " phi: " << 59.2956 * momDir.phi() << " theta: " << 59.2956 * momDir.theta()
                        << " xin : " << hitPoint.x() << " yin : " << hitPoint.y() << " zin : " << hitPoint.z()
                        << " track en: " << energy << "(GeV)"
                        << " section: " << section << " side: " << side << " channel: " << channel
                        << " partID: " << parCode;

  double eav = 0.;
  double esig = 0.;
  double edis = 0.;

  float xin = hitPoint.x();
  float yin = hitPoint.y();
  float fact = 0.;

  bool isEM = G4TrackToParticleID::isGammaElectronPositron(parCode);

  if (section == 1 && !isEM) {
    if (channel < 5)
      if (((theXChannelBoundaries[channel - 1]) < (xin + X0)) && ((xin + X0) <= theXChannelBoundaries[channel]))
        fact = 0.18;
    if (channel == 5)
      if (theXChannelBoundaries[channel - 1] < xin + X0)
        fact = 0.18;
  }

  if (section == 2 && !isEM) {
    if (channel == 1)
      fact = 0.34;
    if (channel == 2)
      fact = 0.24;
    if (channel == 3)
      fact = 0.17;
    if (channel == 4)
      fact = 0.07;
  }
  if (section == 1 && isEM) {
    if (channel < 5)
      if (((theXChannelBoundaries[channel - 1]) < (xin + X0)) && ((xin + X0) <= theXChannelBoundaries[channel]))
        fact = 1.;
    if (channel == 5)
      if (theXChannelBoundaries[channel - 1] < xin + X0)
        fact = 1.0;
  }

  //change to cm for parametrization
  yin = yin / cm;
  xin = xin / cm;

  if (isEM) {
    eav = ((((((-0.0002 * xin - 2.0e-13) * xin + 0.0022) * xin + 1.0e-11) * xin - 0.0217) * xin - 3.0e-10) * xin +
           1.0028) *
          (((0.0001 * yin + 0.0056) * yin + 0.0508) * yin + 1.0) * 300.0 * pow((energy / 300.0), 0.99);  // EM
    esig = ((((((0.0005 * xin - 1.0e-12) * xin - 0.0052) * xin + 5.0e-11) * xin + 0.032) * xin - 2.0e-10) * xin + 1.0) *
           (((0.0006 * yin + 0.0071) * yin - 0.031) * yin + 1.0) * 30.0 * pow((energy / 300.0), 0.54);  // EM
    edis = 1.0;
  } else {
    eav = ((((((-0.0002 * xin - 2.0e-13) * xin + 0.0022) * xin + 1.0e-11) * xin - 0.0217) * xin - 3.0e-10) * xin +
           1.0028) *
          (((0.0001 * yin + 0.0056) * yin + 0.0508) * yin + 1.0) * 300.0 * pow((energy / 300.0), 1.12);  // HD
    esig = ((((((0.0005 * xin - 1.0e-12) * xin - 0.0052) * xin + 5.0e-11) * xin + 0.032) * xin - 2.0e-10) * xin + 1.0) *
           (((0.0006 * yin + 0.0071) * yin - 0.031) * yin + 1.0) * 54.0 * pow((energy / 300.0), 0.93);  //HD
    edis = 3.0;
  }

  if (eav < 0. || esig < 0.) {
    LogDebug("ZdcShower") << " Negative everage energy or esigma from parametrization \n"
                          << " xin: " << xin << "(cm)"
                          << " yin: " << yin << "(cm)"
                          << " track en: " << energy << "(GeV)"
                          << " eaverage: " << eav << " (GeV)"
                          << " esigma: " << esig << "  (GeV)"
                          << " edist: " << edis << " (GeV)";
    return 0;
  }

  // Convert from GeV to MeV for the code
  eav = eav * GeV;
  esig = esig * GeV;

  while (nphotons == -1 || nphotons > int(eav + 5. * esig))
    nphotons = (int)(fact * photonFluctuation(eav, esig, edis));

  LogDebug("ZdcShower")
      //std::cout
      << " track en: " << energy << "(GeV)"
      << " eaverage: " << eav / GeV << " (GeV)"
      << " esigma: " << esig / GeV << "  (GeV)"
      << " edist: " << edis << " (GeV)"
      << " dE hit: " << nphotons / GeV << " (GeV)";

  return nphotons;
}

int ZdcShowerLibrary::photonFluctuation(double eav, double esig, double edis) {
  int nphot = 0;
  double efluct = 0.;
  if (edis == 1.0)
    efluct = eav + esig * CLHEP::RandGaussQ::shoot();
  if (edis == 3.0)
    efluct = eav + esig * CLHEP::RandLandau::shoot();
  nphot = int(efluct);
  if (nphot < 0)
    nphot = 0;
  return nphot;
}
