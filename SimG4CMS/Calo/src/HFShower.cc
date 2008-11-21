///////////////////////////////////////////////////////////////////////////////
// File: HFShower.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShower.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDSplit.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4NavigationHistory.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VSolid.hh"
#include "Randomize.hh"
#include "CLHEP/Units/PhysicalConstants.h"

HFShower::HFShower(std::string & name, const DDCompactView & cpv, 
		   edm::ParameterSet const & p) : cherenkov(0), fibre(0) {

  edm::ParameterSet m_HF = p.getParameter<edm::ParameterSet>("HFShower");
  //static SimpleConfigurable<double> cf1(0.5, "HFShower:CFibre");
  //static SimpleConfigurable<float>  pPr(0.7268,"VCalShowerLibrary:ProbMax");
  probMax          = m_HF.getParameter<double>("ProbMax");

  edm::LogInfo("HFShower") << "HFShower:: Maximum probability cut off " 
			   << probMax;

  G4String attribute = "Volume";
  G4String value     = "HFFibre";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
 
  bool dodet = fv.firstChild();
  std::vector<G4String> tmp;
  while (dodet) {
    const DDSolid & sol  = fv.logicalPart().solid();
    const std::vector<double> & paras = sol.parameters();
    G4String name = DDSplit(sol.name()).first;
    bool ok = true;
    for (unsigned int i=0; i<tmp.size(); i++)
      if (name == tmp[i]) ok = false;
    if (ok) { 
      tmp.push_back(name);
      fibreDz2.insert(std::pair<G4String,double>(name,paras[0]));
      edm::LogInfo("HFShower") << "HFShower::initMap (for " << value 
			       << "): Solid " << name << " Shape " 
			       << sol.shape() << " Parameter 0 = " << paras[0];
    }
    dodet = fv.next();
  }

  cherenkov = new HFCherenkov(p);
  fibre     = new HFFibre(name, cpv, p);

  nHit      = 0;
  clearHits();
}

HFShower::~HFShower() { 
  if (cherenkov) delete cherenkov;
  if (fibre)     delete fibre;
}

int HFShower::getHits(G4Step * aStep) {

  clearHits();
  int    nHit    = 0;

  double edep    = aStep->GetTotalEnergyDeposit();
  double stepl   = 0.;
 
  if (aStep->GetTrack()->GetDefinition()->GetPDGCharge() != 0.)
    stepl = aStep->GetStepLength();
  if ((edep == 0.) || (stepl == 0.)) {
    LogDebug("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
    return nHit;
  }

  G4Track *aTrack = aStep->GetTrack();
  const G4DynamicParticle *aParticle = aTrack->GetDynamicParticle();
 
  double energy   = aParticle->GetTotalEnergy();
  double momentum = aParticle->GetTotalMomentum();
  double pBeta    = momentum / energy;
  double dose     = 0.;
  int    npeDose  = 0;

  G4ThreeVector momentumDir = aParticle->GetMomentumDirection();
  G4ParticleDefinition *particleDef = aTrack->GetDefinition();
     
  G4StepPoint *     preStepPoint = aStep->GetPreStepPoint();
  G4ThreeVector     globalPos    = preStepPoint->GetPosition();
  G4String          name         = 
    preStepPoint->GetTouchable()->GetSolid(0)->GetName();
  G4ThreeVector     localPos     = preStepPoint->GetTouchable()->GetHistory()->
    GetTopTransform().TransformPoint(globalPos);
  G4ThreeVector     localMom     = preStepPoint->GetTouchable()->GetHistory()->
    GetTopTransform().TransformAxis(momentumDir);
  int               depth    = 
    (preStepPoint->GetTouchable()->GetReplicaNumber(0))%10;

  double u        = localMom.x();
  double v        = localMom.y();
  double w        = localMom.z();
  double zCoor    = localPos.z();
  double zFibre   = (fibreLength(name)-zCoor);
  double tSlice   = (aStep->GetPostStepPoint()->GetGlobalTime());
  double time     = fibre->tShift(globalPos, depth, false);

  LogDebug("HFShower") << "HFShower::getHits: in " << name << " Z " << zCoor 
		       << " " << fibreLength(name) << " " << zFibre << " Time "
		       << tSlice  << " " << time 
		       << "\n                  Direction " << momentumDir 
		       << " Local " << localMom;
 
  int npe = cherenkov->computeNPE(particleDef, pBeta, u, v, w, stepl, zFibre, 
				  dose, npeDose);
  std::vector<double> wavelength = cherenkov->getWL();
  
  for (int i = 0; i<npe; ++i) {
    double p   = fibre->attLength(wavelength[i]);
    double r1  = G4UniformRand();
    double r2  = G4UniformRand();
    LogDebug("HFShower") << "HFShower::getHits: " << i << " attenuation " << r1
			 <<":" << exp(-p*zFibre) << " r2 " << r2 << ":" 
			 << probMax << " Survive: " 
			 << (r1 <= exp(-p*zFibre) && r2 <= probMax);
    if (r1 <= exp(-p*zFibre) && r2 <= probMax) {
      nHit++;
      wlHit.push_back(wavelength[i]);
      timHit.push_back(tSlice+time);
    }
  }

  LogDebug("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
  for (int i=0; i<nHit; i++)
    LogDebug("HFShower") << "HFShower::Hit " << i << " WaveLength " << wlHit[i]
			 << " Time " << timHit[i];

  return nHit;

} 
 
double HFShower::getTSlice(int i) {
   
  double tim = 0.;
  if (i < nHit) tim = timHit[i];
  LogDebug("HFShower") << "HFShower: Time (" << i << "/" << nHit << ") " <<tim;
  return tim;
}

double HFShower::fibreLength(G4String name) {

  double length = 825.;
  std::map<G4String,double>::const_iterator it = fibreDz2.find(name);
  if (it != fibreDz2.end()) length = it->second;
  return length;
}

void HFShower::clearHits() {

  wlHit.clear();
  timHit.clear();
}
