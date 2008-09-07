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

HFShowerParam::HFShowerParam(std::string & name, const DDCompactView & cpv,
			     edm::ParameterSet const & p) : fibre(0) {

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  pePerGeV                = m_HF.getParameter<double>("PEPerGeV");
  trackEM                 = m_HF.getParameter<bool>("TrackEM");
  edm::LogInfo("HFShower") << "HFShowerParam:: P.E. per GeV " << pePerGeV
			   << " and Track EM Flag " << trackEM;
  
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
  
  fibre = new HFFibre(name, cpv, p);
}

HFShowerParam::~HFShowerParam() {
  if (fibre) delete fibre;
}

std::vector<double> HFShowerParam::getHits(G4Step * aStep) {

  std::vector<double> edeps;
  hits.clear();

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4Track *     track    = aStep->GetTrack();   
  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4String      partType = track->GetDefinition()->GetParticleName();
  G4ThreeVector localPoint = (preStepPoint->GetTouchable()->GetHistory()->GetTopTransform()).TransformPoint(hitPoint);
  double pin    = (preStepPoint->GetTotalEnergy())/GeV;
  double zint   = hitPoint.z(); 
  double zz     = std::abs(zint) - gpar[4];
  
  edm::LogInfo("HFShower") << "HFShowerParam: getHits " << partType
		       << " of energy " << pin << " GeV" 
                       << " Pos x,y,z = " << hitPoint.x() << "," 
		       << hitPoint.y() << "," << zint << " (" << zz << ","
			   << localPoint.z() << ", " << (localPoint.z()+0.5*gpar[1]) << ")";

  HFShowerParam::Hit hit;
  hit.position = hitPoint;
  // take only e+-/gamma
  if (partType == "e-" || partType == "e+" || partType == "gamma" ) {
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
      edep         *= 0.5*pePerGeV;
      double tSlice = (aStep->GetPostStepPoint()->GetGlobalTime());

      double time = fibre->tShift(hitPoint,1,false); // remaining part
      hit.depth   = 1;
      hit.time    = tSlice+time;
      edeps.push_back(edep);
      hits.push_back(hit);
      if (zz >= gpar[0]) {
	time      = fibre->tShift(hitPoint,2,false);
	hit.depth = 2;
	hit.time  = tSlice+time;
	edeps.push_back(edep);
	hits.push_back(hit);
      }
      if (kill) track->SetTrackStatus(fStopAndKill);
      edm::LogInfo("HFShower") << "HFShowerParam: getHits kill (" << kill
			       << ") track " << track->GetTrackID() 
			       << " and deposit " << edep << " " <<edeps.size()
			       << " times" << " ZZ " << zz << " " << gpar[0];
    }
  }
    
  return edeps;

}
G4ThreeVector HFShowerParam::getPosHit(int i) {

  G4ThreeVector pos;
  if (i < static_cast<int>(hits.size())) pos = (hits[i].position);
  LogDebug("HFShower") << "HFShowerParam::getPosHit (" << i << "/" 
		       << hits.size() << ") " << pos;
  return pos;
}

int HFShowerParam::getDepth(int i) {

  int depth = 0;
  if (i < static_cast<int>(hits.size())) depth = (hits[i].depth);
  LogDebug("HFShower") << "HFShowerParam::getDepth (" << i << "/" 
		       << hits.size() << ") "  << depth;
  return depth;
}
 
double HFShowerParam::getTSlice(int i) {
   
  double tim = 0.;
  if (i < static_cast<int>(hits.size())) tim = (hits[i].time);
  LogDebug("HFShower") << "HFShowerParam::getTSlice (" << i << "/" 
		       << hits.size()<< ") " << tim;
  return tim;
}

std::vector<double> HFShowerParam::getDDDArray(const std::string & str, 
					       const DDsvalues_type & sv) {

  LogDebug("HFShower") << "HFShowerParam:getDDDArray called for " << str;

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    LogDebug("HFShower") << value;
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
