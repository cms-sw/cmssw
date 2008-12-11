///////////////////////////////////////////////////////////////////////////////
// File: HFShowerParam.cc
// Description: Parametrized version of HF hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerParam.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4NavigationHistory.hh"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

//#define DebugLog

HFShowerParam::HFShowerParam(std::string & name, const DDCompactView & cpv,
			     edm::ParameterSet const & p) : showerLibrary(0),
							    fibre(0) {

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  pePerGeV                = m_HF.getParameter<double>("PEPerGeV");
  trackEM                 = m_HF.getParameter<bool>("TrackEM");
  bool useShowerLibrary   = m_HF.getParameter<bool>("UseShowerLibrary");
  edMin                   = m_HF.getParameter<double>("EminLibrary");
  edm::LogInfo("HFShower") << "HFShowerParam::Use of shower library is set to "
			   << useShowerLibrary << " P.E. per GeV " << pePerGeV
			   << " and Track EM Flag " << trackEM << " edMin "
			   << edMin << " GeV";
  
  G4String attribute = "ReadOutName";
  G4String value     = name;
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,value,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  bool dodet = fv.firstChild();
  if (dodet) {
    DDsvalues_type sv(fv.mergedSpecifics());

    //Special Geometry parameters
    gpar      = getDDDArray("gparHF",sv);
    edm::LogInfo("HFShower") << "HFShowerParam: " <<gpar.size() <<" gpar (cm)";
    for (unsigned int ig=0; ig<gpar.size(); ig++)
      edm::LogInfo("HFShower") << "HFShowerParam: gpar[" << ig << "] = "
			       << gpar[ig]/cm << " cm";
  } else {
    edm::LogError("HFShower") << "HFShowerParam: cannot get filtered "
			      << " view for " << attribute << " matching "
			      << name;
    throw cms::Exception("Unknown", "HFShowerParam")
      << "cannot match " << attribute << " to " << name <<"\n";
  }
  
  if (useShowerLibrary) showerLibrary = new HFShowerLibrary(name, cpv, p);
  fibre = new HFFibre(name, cpv, p);
}

HFShowerParam::~HFShowerParam() {
  if (fibre)         delete fibre;
  if (showerLibrary) delete showerLibrary;
}

void HFShowerParam::initRun(G4ParticleTable * theParticleTable) {

  emPDG = theParticleTable->FindParticle("e-")->GetPDGEncoding();
  epPDG = theParticleTable->FindParticle("e+")->GetPDGEncoding();
  gammaPDG = theParticleTable->FindParticle("gamma")->GetPDGEncoding();
#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerParam: Particle code for e- = " << emPDG
		       << " for e+ = " << epPDG << " for gamma = " << gammaPDG;
#endif
  if (showerLibrary) showerLibrary->initRun(theParticleTable);
}

std::vector<HFShowerParam::Hit> HFShowerParam::getHits(G4Step * aStep) {

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4Track *     track    = aStep->GetTrack();   
  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4int         particleCode = track->GetDefinition()->GetPDGEncoding();
  G4ThreeVector localPoint = (preStepPoint->GetTouchable()->GetHistory()->GetTopTransform()).TransformPoint(hitPoint);

  double pin    = (preStepPoint->GetTotalEnergy())/GeV;
  double zint   = hitPoint.z(); 
  double zz     = std::abs(zint) - gpar[4];

#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerParam: getHits " 
		       << track->GetDefinition()->GetParticleName()
		       << " of energy " << pin << " GeV" 
                       << " Pos x,y,z = " << hitPoint.x() << "," 
		       << hitPoint.y() << "," << zint << " (" << zz << ","
		       << localPoint.z() << ", " << (localPoint.z()+0.5*gpar[1]) << ")";
#endif
  std::vector<HFShowerParam::Hit> hits;
  HFShowerParam::Hit hit;
  hit.position = hitPoint;
  // take only e+-/gamma
  if (particleCode == emPDG || particleCode == epPDG ||
      particleCode == gammaPDG ) {
    // Leave out the last part
    double edep = 0.;
    bool   kill = false;
    if ((!trackEM) && (zz < (gpar[1]-gpar[2]))) {
      edep = pin;
      kill = true;
    } else {
      edep = (aStep->GetTotalEnergyDeposit())/GeV;
    }
    if (edep > 0) {
      if (showerLibrary && kill && pin > edMin) {
	std::vector<HFShowerLibrary::Hit> hitSL = showerLibrary->getHits(aStep,kill);
	for (unsigned int i=0; i<hitSL.size(); i++) {
	  hit.position = hitSL[i].position;
	  hit.depth    = hitSL[i].depth;
	  hit.time     = hitSL[i].time;
	  hit.edep     = 1;
	  hits.push_back(hit);
	}
      } else {
	edep         *= pePerGeV;
	double tSlice = (aStep->GetPostStepPoint()->GetGlobalTime());
	double time = fibre->tShift(localPoint,1,0); // remaining part
	hit.depth   = 1;
	hit.time    = tSlice+time;
	hit.edep    = edep;
	hits.push_back(hit);
	if (zz >= gpar[0]) {
	  time      = fibre->tShift(hitPoint,2,0);
	  hit.depth = 2;
	  hit.time  = tSlice+time;
	  hits.push_back(hit);
	}
      }
      if (kill) {
	track->SetTrackStatus(fStopAndKill);
	G4TrackVector tv = *(aStep->GetSecondary());
	for (unsigned int kk=0; kk<tv.size(); kk++) {
	  if (tv[kk]->GetVolume() == preStepPoint->GetPhysicalVolume())
	    tv[kk]->SetTrackStatus(fStopAndKill);
	}
      }
#ifdef DebugLog
      LogDebug("HFShower") << "HFShowerParam: getHits kill (" << kill
			   << ") track " << track->GetTrackID() 
			   << " at " << hitPoint
			   << " and deposit " << edep << " " << hits.size()
			   << " times" << " ZZ " << zz << " " << gpar[0];
#endif
    }
  }
    
  return hits;
}

std::vector<double> HFShowerParam::getDDDArray(const std::string & str, 
					       const DDsvalues_type & sv) {

#ifdef DebugLog
  LogDebug("HFShower") << "HFShowerParam:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    LogDebug("HFShower") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 2) {
      edm::LogError("HFShower") << "HFShowerParam : # of " << str 
				<< " bins " << nval << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HFShowerParam")
	<< "nval < 2 for array " << str << "\n";
    }

    return fvec;
  } else {
    edm::LogError("HFShower") << "HFShowerParam : cannot get array " << str;
    throw cms::Exception("Unknown", "HFShowerParam") 
      << "cannot get array " << str << "\n";
  }
}
