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

std::vector<HFShowerParam::Hit> HFShowerParam::getHits(G4Step * aStep) {

  G4StepPoint * preStepPoint  = aStep->GetPreStepPoint(); 
  G4Track *     track    = aStep->GetTrack();   
  G4ThreeVector hitPoint = preStepPoint->GetPosition();   
  G4String      partType = track->GetDefinition()->GetParticleName();

  double pin    = (preStepPoint->GetTotalEnergy())/GeV;
  double zint   = hitPoint.z(); 
  double zz     = std::abs(zint) - gpar[4];
  
  LogDebug("HFShower") << "HFShowerParam: getHits " << partType
		       << " of energy " << pin << " GeV" 
                       << " Pos x,y,z = " << hitPoint.x() << "," 
		       << hitPoint.y() << "," << zint << " (" << zz << ")";

  std::vector<HFShowerParam::Hit> hits;
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
      hit.edep    = edep;
      hits.push_back(hit);
      if (zz >= gpar[0]) {
	time      = fibre->tShift(hitPoint,2,false);
	hit.depth = 2;
	hit.time  = tSlice+time;
	hits.push_back(hit);
      }
      if (kill) track->SetTrackStatus(fStopAndKill);
      LogDebug("HFShower") << "HFShowerParam: getHits kill track " 
			   << track->GetTrackID() << " and deposit "
			   << edep << " " << hits.size() << " times";
    }
  }
    
  return hits;

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
