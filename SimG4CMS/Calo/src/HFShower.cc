///////////////////////////////////////////////////////////////////////////////
// File: HFShower.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShower.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
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
#include "CLHEP/Units/GlobalPhysicalConstants.h"

//#define DebugLog

#include<iostream>

HFShower::HFShower(std::string & name, const DDCompactView & cpv, 
                   edm::ParameterSet const & p, int chk) : cherenkov(0),
                                                           fibre(0),
                                                           chkFibre(chk) {

  edm::ParameterSet m_HF = p.getParameter<edm::ParameterSet>("HFShower");
  applyFidCut            = m_HF.getParameter<bool>("ApplyFiducialCut");
  probMax                = m_HF.getParameter<double>("ProbMax");

  edm::LogInfo("HFShower") << "HFShower:: Maximum probability cut off " 
                           << probMax << " Check flag " << chkFibre;

  G4String attribute = "ReadOutName";
  G4String value     = name;
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();
  if ( dodet ) {
    DDsvalues_type sv(fv.mergedSpecifics());

    //Special Geometry parameters
    int ngpar = 7;
    gpar      = getDDDArray("gparHF", sv,ngpar);
    edm::LogInfo("HFShower") << "HFShower: " << ngpar << " gpar (cm)";
    for (int ig=0; ig<ngpar; ig++)
      edm::LogInfo("HFShower") << "HFShower: gpar[" << ig << "] = "
                               << gpar[ig]/cm << " cm";
  } else {
    edm::LogError("HFShower") << "HFShower: cannot get filtered "
                              << " view for " << attribute << " matching " << name;
    throw cms::Exception("Unknown", "HFShower")
      << "cannot match " << attribute << " to " << name <<"\n";
  }
  cherenkov = new HFCherenkov(m_HF);
  fibre     = new HFFibre(name, cpv, p);
}

HFShower::~HFShower() { 
  if (cherenkov) delete cherenkov;
  if (fibre)     delete fibre;
}

std::vector<HFShower::Hit> HFShower::getHits(G4Step * aStep, double weight) {

  std::vector<HFShower::Hit> hits;
  int    nHit    = 0;

  double edep    = weight*(aStep->GetTotalEnergyDeposit());
#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShower::getHits: energy " << aStep->GetTotalEnergyDeposit() << " weight " << weight << " edep " << edep;
#endif
  double stepl   = 0.;

  if (aStep->GetTrack()->GetDefinition()->GetPDGCharge() != 0.)
    stepl = aStep->GetStepLength();
  if ((edep == 0.) || (stepl == 0.)) {
#ifdef DebugLog
    edm::LogInfo("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
#endif
    return hits;
  }
  G4Track *aTrack = aStep->GetTrack();
  const G4DynamicParticle *aParticle = aTrack->GetDynamicParticle();
 
  HFShower::Hit hit;
  double energy   = aParticle->GetTotalEnergy();
  double momentum = aParticle->GetTotalMomentum();
  double pBeta    = momentum / energy;
  double dose     = 0.;
  int    npeDose  = 0;

  G4ThreeVector momentumDir = aParticle->GetMomentumDirection();
  G4ParticleDefinition *particleDef = aTrack->GetDefinition();
     
  G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
  G4ThreeVector globalPos    = preStepPoint->GetPosition();
  G4String      name         = preStepPoint->GetTouchable()->GetSolid(0)->GetName();
  //double        zv           = std::abs(globalPos.z()) - gpar[4] - 0.5*gpar[1];
  double        zv           = std::abs(globalPos.z()) - gpar[4];
  G4ThreeVector localPos     = G4ThreeVector(globalPos.x(),globalPos.y(), zv);
  G4ThreeVector localMom     = preStepPoint->GetTouchable()->GetHistory()->
                               GetTopTransform().TransformAxis(momentumDir);
  // @@ Here the depth should be changed  (Fibers all long in Geometry!)
  int  depth = 1;
  int  npmt  = 0;
  bool ok    = true;
  if (zv < 0. || zv > gpar[1]) {
    ok = false; // beyond the fiber in Z
  }
  if (ok && applyFidCut) {
    npmt = HFFibreFiducial::PMTNumber(globalPos);
    if (npmt <= 0) {
      ok = false;
    } else if (npmt > 24) { // a short fibre
      if (zv > gpar[0]) {
	depth = 2; 
      } else {
        ok = false;
      }
    }
#ifdef DebugLog
    edm::LogInfo("HFShower") << "HFShower: npmt " << npmt 
                             << " zv " << std::abs(globalPos.z()) 
                             << ":" << gpar[4] << ":" << zv << ":" 
                             << gpar[0] << " ok " << ok << " depth " << depth;
#endif
  } else {
    depth      = (preStepPoint->GetTouchable()->GetReplicaNumber(0))%10; // All LONG!
  }
  G4ThreeVector translation =
                        preStepPoint->GetTouchable()->GetVolume(1)->GetObjectTranslation();

  double u        = localMom.x();
  double v        = localMom.y();
  double w        = localMom.z();
  double zCoor    = localPos.z();
  double zFibre   = (0.5*gpar[1]-zCoor-translation.z());
  double tSlice   = (aStep->GetPostStepPoint()->GetGlobalTime());
  double time     = fibre->tShift(localPos, depth, chkFibre);

#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShower::getHits: in " << name << " Z " 
			   << zCoor << "(" << globalPos.z() << ") " << zFibre 
			   << " Trans " << translation << " Time " << tSlice  
			   << " " << time << "\n                  Direction " 
			   << momentumDir << " Local " << localMom;
#endif 
  // Here npe should be 0 if there is no fiber (@@ M.K.)
  int npe = 1;
  std::vector<double> wavelength;
  std::vector<double> momz;
  if (!applyFidCut) { // _____ Tmp close of the cherenkov function
    if (ok) npe = cherenkov->computeNPE(aStep,particleDef,pBeta,u,v,w,stepl,zFibre,dose, npeDose);
    wavelength = cherenkov->getWL();
    momz       = cherenkov->getMom();
  } // ^^^^^ End of Tmp close of the cherenkov function
  if(ok && npe>0) {
    for (int i = 0; i<npe; ++i) {
      double p=1.;
      if (!applyFidCut) p   = fibre->attLength(wavelength[i]); 
      double r1  = G4UniformRand();
      double r2  = G4UniformRand();
#ifdef DebugLog
      edm::LogInfo("HFShower") << "HFShower::getHits: " << i << " attenuation "
			       << r1 <<":" << exp(-p*zFibre) << " r2 " << r2 
			       << ":" << probMax << " Survive: " 
			       << (r1 <= exp(-p*zFibre) && r2 <= probMax);
#endif
      if (applyFidCut || chkFibre < 0 || (r1 <= exp(-p*zFibre) && r2 <= probMax)) {
	hit.depth      = depth;
	hit.time       = tSlice+time;
	if (!applyFidCut) {
	  hit.wavelength = wavelength[i]; // Tmp
	  hit.momentum   = momz[i]; // Tmp
	} else {
	  hit.wavelength = 300.; // Tmp
	  hit.momentum   = 1.; // Tmp
	}
	hit.position   = globalPos;
	hits.push_back(hit);
	nHit++;
      }
    }
  }

#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
  for (int i=0; i<nHit; ++i)
    edm::LogInfo("HFShower") << "HFShower::Hit " << i << " WaveLength " 
			     << hits[i].wavelength << " Time " << hits[i].time
			     << " Momentum " << hits[i].momentum <<" Position "
			     << hits[i].position;
#endif
  return hits;

} 

std::vector<HFShower::Hit> HFShower::getHits(G4Step * aStep,
					     bool forLibraryProducer,
					     double zoffset) {

  std::vector<HFShower::Hit> hits;
  int    nHit    = 0;

  double edep    = aStep->GetTotalEnergyDeposit();
  double stepl   = 0.;

  if (aStep->GetTrack()->GetDefinition()->GetPDGCharge() != 0.)
    stepl = aStep->GetStepLength();
  if ((edep == 0.) || (stepl == 0.)) {
#ifdef DebugLog
    edm::LogInfo("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
#endif
    return hits;
  }
  G4Track *aTrack = aStep->GetTrack();
  const G4DynamicParticle *aParticle = aTrack->GetDynamicParticle();
 
  HFShower::Hit hit;
  double energy   = aParticle->GetTotalEnergy();
  double momentum = aParticle->GetTotalMomentum();
  double pBeta    = momentum / energy;
  double dose     = 0.;
  int    npeDose  = 0;

  G4ThreeVector momentumDir = aParticle->GetMomentumDirection();
  G4ParticleDefinition *particleDef = aTrack->GetDefinition();
     
  G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
  G4ThreeVector globalPos    = preStepPoint->GetPosition();
  G4String      name         = preStepPoint->GetTouchable()->GetSolid(0)->GetName();
  //double        zv           = std::abs(globalPos.z()) - gpar[4] - 0.5*gpar[1];
  //double        zv           = std::abs(globalPos.z()) - gpar[4];
  double        zv           = gpar[1]-(std::abs(globalPos.z()) - zoffset);
  G4ThreeVector localPos     = G4ThreeVector(globalPos.x(),globalPos.y(), zv);
  G4ThreeVector localMom     = preStepPoint->GetTouchable()->GetHistory()->
                               GetTopTransform().TransformAxis(momentumDir);
  // @@ Here the depth should be changed  (Fibers all long in Geometry!)
  int  depth = 1;
  int  npmt  = 0;
  bool ok    = true;
  if (zv < 0. || zv > gpar[1]) {
    ok = false; // beyond the fiber in Z
  }
  if (ok && applyFidCut) {
    npmt = HFFibreFiducial::PMTNumber(globalPos);
    if (npmt <= 0) {
      ok = false;
    } else if (npmt > 24) { // a short fibre
      if (zv > gpar[0]) {
	depth = 2; 
      } else {
        ok = false;
      }
    }
#ifdef DebugLog
    edm::LogInfo("HFShower") << "HFShower: npmt " << npmt 
                             << " zv " << std::abs(globalPos.z()) 
                             << ":" << gpar[4] << ":" << zv << ":" 
                             << gpar[0] << " ok " << ok << " depth " << depth;
#endif
  } else {
    depth      = (preStepPoint->GetTouchable()->GetReplicaNumber(0))%10; // All LONG!
  }
  G4ThreeVector translation =
                        preStepPoint->GetTouchable()->GetVolume(1)->GetObjectTranslation();

  double u        = localMom.x();
  double v        = localMom.y();
  double w        = localMom.z();
  double zCoor    = localPos.z();
  double zFibre   = (0.5*gpar[1]-zCoor-translation.z());
  double tSlice   = (aStep->GetPostStepPoint()->GetGlobalTime());
  double time     = fibre->tShift(localPos, depth, chkFibre);

#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShower::getHits: in " << name << " Z " <<zCoor
			   << "(" << globalPos.z() <<") " << zFibre <<" Trans "
			   << translation << " Time " << tSlice  << " " << time
			   << "\n                  Direction " << momentumDir 
			   << " Local " << localMom;
#endif 
  // Here npe should be 0 if there is no fiber (@@ M.K.)
  int npe = 1;
  std::vector<double> wavelength;
  std::vector<double> momz;
  if (!applyFidCut) { // _____ Tmp close of the cherenkov function
    if (ok) npe = cherenkov->computeNPE(aStep,particleDef,pBeta,u,v,w,stepl,zFibre,dose, npeDose);
    wavelength = cherenkov->getWL();
    momz       = cherenkov->getMom();
  } // ^^^^^ End of Tmp close of the cherenkov function
  if (ok && npe>0) {
    for (int i = 0; i<npe; ++i) {
      double p=1.;
      if (!applyFidCut) p   = fibre->attLength(wavelength[i]); 
      double r1  = G4UniformRand();
      double r2  = G4UniformRand();
#ifdef DebugLog
      edm::LogInfo("HFShower") << "HFShower::getHits: " << i << " attenuation "
			       << r1 << ":" << exp(-p*zFibre) << " r2 " << r2 
			       << ":" << probMax << " Survive: " 
			       << (r1 <= exp(-p*zFibre) && r2 <= probMax);
#endif
      if (applyFidCut || chkFibre < 0 || (r1 <= exp(-p*zFibre) && r2 <= probMax)) {
	hit.depth      = depth;
	hit.time       = tSlice+time;
	if (!applyFidCut) {
	  hit.wavelength = wavelength[i]; // Tmp
	  hit.momentum   = momz[i]; // Tmp
	} else {
	  hit.wavelength = 300.; // Tmp
	  hit.momentum   = 1.; // Tmp
	}
	hit.position   = globalPos;
	hits.push_back(hit);
	nHit++;
      }
    }
  }

#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
  for (int i=0; i<nHit; ++i)
    edm::LogInfo("HFShower") << "HFShower::Hit " << i << " WaveLength " 
			     << hits[i].wavelength << " Time " << hits[i].time
			     << " Momentum " << hits[i].momentum <<" Position "
			     << hits[i].position;
#endif
  return hits;

} 

std::vector<HFShower::Hit> HFShower::getHits(G4Step * aStep, bool forLibrary) {
  std::vector<HFShower::Hit> hits;
  int    nHit    = 0;

  double edep    = aStep->GetTotalEnergyDeposit();
  double stepl   = 0.;

  if (aStep->GetTrack()->GetDefinition()->GetPDGCharge() != 0.)
    stepl = aStep->GetStepLength();
  if ((edep == 0.) || (stepl == 0.)) {
#ifdef DebugLog
    edm::LogInfo("HFShower") << "HFShower::getHits: Number of Hits " << nHit;
#endif
    return hits;
  }

  G4Track *aTrack = aStep->GetTrack();
  const G4DynamicParticle *aParticle = aTrack->GetDynamicParticle();

  HFShower::Hit hit;
  double energy   = aParticle->GetTotalEnergy();
  double momentum = aParticle->GetTotalMomentum();
  double pBeta    = momentum / energy;
  double dose     = 0.;
  int    npeDose  = 0;

  G4ThreeVector momentumDir = aParticle->GetMomentumDirection();
  G4ParticleDefinition *particleDef = aTrack->GetDefinition();

  G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
  G4ThreeVector globalPos    = preStepPoint->GetPosition();
  G4String      name         = preStepPoint->GetTouchable()->GetSolid(0)->GetName();
  double        zv           = std::abs(globalPos.z()) - gpar[4] - 0.5*gpar[1];
  G4ThreeVector localPos     = G4ThreeVector(globalPos.x(),globalPos.y(), zv);
  G4ThreeVector localMom     = preStepPoint->GetTouchable()->GetHistory()->
    GetTopTransform().TransformAxis(momentumDir);
  // @@ Here the depth should be changed (Fibers are all long in Geometry!)
  int depth  = 1;
  int npmt   = 0;
  bool ok    = true;
  if (zv < 0 || zv > gpar[1]) {
    ok = false; // beyond the fiber in Z
  }
  if (ok && applyFidCut) {
    npmt = HFFibreFiducial:: PMTNumber(globalPos);
    if (npmt <= 0) {
      ok = false;
    } else if (npmt > 24) { // a short fibre
      if (zv > gpar[0]) {
	depth = 2; 
      } else {
        ok = false;
      }
    }
#ifdef DebugLog
    edm::LogInfo("HFShower") << "HFShower:getHits(SL): npmt " << npmt 
                             << " zv " << std::abs(globalPos.z()) 
                             << ":" << gpar[4] << ":" << zv << ":" 
                             << gpar[0] << " ok " << ok << " depth " << depth;
#endif
  } else {
    depth      = (preStepPoint->GetTouchable()->GetReplicaNumber(0))%10; // All LONG!
  }
  G4ThreeVector translation  =
                        preStepPoint->GetTouchable()->GetVolume(1)->GetObjectTranslation();

  double u        = localMom.x();
  double v        = localMom.y();
  double w        = localMom.z();
  double zCoor    = localPos.z();
  double zFibre   = (0.5*gpar[1]-zCoor-translation.z());
  double tSlice   = (aStep->GetPostStepPoint()->GetGlobalTime());
  double time     = fibre->tShift(localPos, depth, chkFibre);

#ifdef DebugLog
  edm::LogInfo("HFShower") << "HFShower::getHits(SL): in " << name << " Z " 
			   << zCoor << "(" << globalPos.z() << ") " << zFibre 
			   << " Trans " << translation << " Time " << tSlice  
			   << " " << time << "\n                  Direction " 
			   << momentumDir << " Local " << localMom;
#endif
  // npe should be 0
  int npe = 0;
  if(ok) npe = cherenkov->computeNPE(aStep,particleDef,pBeta,u,v,w, stepl,zFibre, dose, npeDose);
  std::vector<double> wavelength = cherenkov->getWL();
  std::vector<double> momz       = cherenkov->getMom();

  for (int i = 0; i<npe; ++i) {
    hit.time = tSlice+time;
    hit.wavelength = wavelength[i];
    hit.momentum   = momz[i];
    hit.position   = globalPos;
    hits.push_back(hit);
    nHit++;
  }

  return hits;
}

std::vector<double> HFShower::getDDDArray(const std::string & str, 
                                          const DDsvalues_type & sv, 
					  int & nmin) {
#ifdef DebugLog
  LogDebug("HFShower") << "HFShower:getDDDArray called for " << str 
                       << " with nMin " << nmin;
#endif
  DDValue value(str);
  if (DDfetch(&sv, value)) {
#ifdef DebugLog
    LogDebug("HFShower") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nmin > 0) {
      if (nval < nmin) {
        edm::LogError("HFShower") << "HFShower : # of " << str << " bins "
                                  << nval << " < " << nmin  << " ==> illegal";
        throw cms::Exception("Unknown", "HFShower")
                                  << "nval < nmin for array " << str << "\n";
      }
    } else {
      if (nval < 2) {
        edm::LogError("HFShower") << "HFShower : # of " << str << " bins " << nval
                                  << " < 2 ==> illegal" << " (nmin=" << nmin << ")";
        throw cms::Exception("Unknown", "HFShower")
	  << "nval < 2 for array " << str << "\n";
      }
    }
    nmin = nval;

    return fvec;
  } else {
    edm::LogError("HFShower") << "HFShower : cannot get array " << str;
    throw cms::Exception("Unknown", "HFShower") << "cannot get array " << str << "\n";
  }
}

void HFShower::initRun(G4ParticleTable *) {// Define PDG codes
}
