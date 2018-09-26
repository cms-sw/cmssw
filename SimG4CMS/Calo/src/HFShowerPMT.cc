///////////////////////////////////////////////////////////////////////////////
// File: HFShowerPMT.cc
// Description: Parametrized version of HF hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerPMT.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4NavigationHistory.hh"
#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "CLHEP/Units/GlobalPhysicalConstants.h"
#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include <sstream>

//#define DebugLog

HFShowerPMT::HFShowerPMT(const std::string & name, const DDCompactView & cpv,
                         edm::ParameterSet const & p) : cherenkov(nullptr) {

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShowerPMT");
  pePerGeV                = m_HF.getParameter<double>("PEPerGeVPMT");
  
  //Special Geometry parameters
  std::string attribute = "Volume";
  std::string value     = "HFPMT";
  DDSpecificsMatchesValueFilter filter1{DDValue(attribute,value,0)};
  DDFilteredView fv1(cpv,filter1);
  if (fv1.firstChild()) {
    DDsvalues_type sv1(fv1.mergedSpecifics());
    std::vector<double> neta;
    neta = getDDDArray("indexPMTR",sv1);
    for (unsigned int ii=0; ii<neta.size(); ii++) {
      int index = static_cast<int>(neta[ii]);
      int ir=-1, ifib=-1;
      if (index >= 0) {
        ir   = index/10; ifib = index%10;
      }
      pmtR1.push_back(ir);
      pmtFib1.push_back(ifib);
    }
    neta = getDDDArray("indexPMTL",sv1);
    for (unsigned int ii=0; ii<neta.size(); ii++) {
      int index = static_cast<int>(neta[ii]);
      int ir=-1, ifib=-1;
      if (index >= 0) {
        ir   = index/10; ifib = index%10;
      }
      pmtR2.push_back(ir);
      pmtFib2.push_back(ifib);
    }
#ifdef DebugLog
    edm::LogVerbatim("HFShower") << "HFShowerPMT: gets the Index matches for "
                             << neta.size() << " PMTs";
    for (unsigned int ii=0; ii<neta.size(); ii++) {
      edm::LogVerbatim("HFShower") << "HFShowerPMT: rIndexR[" << ii << "] = "
                               << pmtR1[ii] << " fibreR[" << ii << "] = "
                               << pmtFib1[ii] << " rIndexL[" << ii << "] = "
                               << pmtR2[ii] << " fibreL[" << ii << "] = "
                               << pmtFib2[ii];
    }
#endif
  } else {
    edm::LogWarning("HFShower") << "HFShowerPMT: cannot get filtered "
                                << " view for " << attribute << " matching "
                                << value;
  }

  cherenkov = new HFCherenkov(m_HF);
}

HFShowerPMT::~HFShowerPMT() {
  if (cherenkov) delete cherenkov;
}

void HFShowerPMT::initRun(const HcalDDDSimConstants* hcons) {

  // Special Geometry parameters
  rTable = hcons->getRTableHF();
  std::stringstream sss;
  for (unsigned int ig=0; ig<rTable.size(); ++ig) {
    if(ig/10*10 == ig) { sss << "\n"; }
    sss << "  " << rTable[ig]/cm;
  }
  edm::LogVerbatim("HFShowerPMT") << "HFShowerPMT: " << rTable.size() 
                              << " rTable(cm):" << sss.str();
}

double HFShowerPMT::getHits(const G4Step * aStep) {

  indexR = indexF = -1;

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch   = preStepPoint->GetTouchable();
  int                 boxNo   = touch->GetReplicaNumber(2);
  int                 pmtNo   = touch->GetReplicaNumber(1);
  if (boxNo <= 1) {
    indexR = pmtR1[pmtNo-1];
    indexF = pmtFib1[pmtNo-1];
  } else {
    indexR = pmtR2[pmtNo-1];
    indexF = pmtFib2[pmtNo-1];
  }

#ifdef DebugLog
  double edep = aStep->GetTotalEnergyDeposit();
  LogDebug("HFShower") << "HFShowerPMT: Box " << boxNo << " PMT "
                       << pmtNo << " Mapped Indices " << indexR << ", "
                       << indexF << " Edeposit " << edep/MeV << " MeV; PE "
                       << edep*pePerGeV/GeV;
#endif

  double photons = 0;
  if (indexR >= 0 && indexF > 0) {
    G4Track *aTrack = aStep->GetTrack();
    G4ParticleDefinition *particleDef = aTrack->GetDefinition();
    double stepl = aStep->GetStepLength();
    double beta  = preStepPoint->GetBeta();
    G4ThreeVector pDir = aTrack->GetDynamicParticle()->GetMomentumDirection();
    G4ThreeVector localMom = preStepPoint->GetTouchable()->GetHistory()->
      GetTopTransform().TransformAxis(pDir);
    photons = cherenkov->computeNPEinPMT(particleDef, beta, localMom.x(),
                                         localMom.y(), localMom.z(), stepl);
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerPMT::getHits: for particle " 
                       << particleDef->GetParticleName() << " Step " << stepl
                       << " Beta " << beta << " Direction " << pDir
                       << " Local " << localMom << " p.e. " << photons;
#endif 

  }
  return photons;
}
 
double HFShowerPMT::getRadius() {
   
  double r = 0.;
  if (indexR >= 0 && indexR+1 < (int)(rTable.size()))
    r = 0.5*(rTable[indexR]+rTable[indexR+1]);
#ifdef DebugLog
  else
    LogDebug("HFShower") << "HFShowerPMT::getRadius: R " << indexR
                         << " F " << indexF;
#endif
  if (indexF == 2)  r =-r;
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerPMT: Radius (" << indexR << "/" << indexF 
                       << ") " << r;
#endif
  return r;
}

std::vector<double> HFShowerPMT::getDDDArray(const std::string & str, 
                                             const DDsvalues_type & sv) {

#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerPMT:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("HFShower") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 2) {
      edm::LogError("HFShower") << "HFShowerPMT: # of " << str 
                                << " bins " << nval << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HFShowerPMT")
        << "nval < 2 for array " << str << "\n";
    }

    return fvec;
  } else {
    edm::LogError("HFShower") << "HFShowerPMT: cannot get array " << str;
    throw cms::Exception("Unknown", "HFShowerPMT") 
      << "cannot get array " << str << "\n";
  }
}
