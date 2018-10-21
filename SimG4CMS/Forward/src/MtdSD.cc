#include "SimG4CMS/Forward/interface/MtdSD.h"

#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "Geometry/MTDCommonData/interface/BTLNumberingScheme.h"
#include "Geometry/MTDCommonData/interface/ETLNumberingScheme.h"
#include "DataFormats/ForwardDetId/interface/MTDDetId.h"

#include "G4Track.hh"
#include "G4Step.hh"
#include "G4StepPoint.hh"

#include <iostream>

//#define EDM_ML_DEBUG
//-------------------------------------------------------------------
MtdSD::MtdSD(const std::string& name, const DDCompactView & cpv,
	     const SensitiveDetectorCatalog & clg, 
	     edm::ParameterSet const & p, 
	     const SimTrackManager* manager) :
  TimingSD(name, cpv, clg, p, manager), numberingScheme(nullptr) {
    
  //Parameters
  edm::ParameterSet m_p = p.getParameter<edm::ParameterSet>("MtdSD");
  int verbn = m_p.getUntrackedParameter<int>("Verbosity");
    
  SetVerboseLevel(verbn);
        
  std::string attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute,name,0)};
  DDFilteredView fv(cpv,filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());
  std::vector<int> temp = dbl_to_int(getDDDArray("Type",sv));
  int type = temp[0];

  MTDNumberingScheme* scheme=nullptr;
  if (name == "FastTimerHitsBarrel") {
    scheme = dynamic_cast<MTDNumberingScheme*>(new BTLNumberingScheme());
    isBTL=true;
  } else if (name == "FastTimerHitsEndcap") { 
    scheme = dynamic_cast<MTDNumberingScheme*>(new ETLNumberingScheme());
    isETL=true;
  } else {
    scheme = nullptr;
    edm::LogWarning("MtdSim") << "MtdSD: ReadoutName not supported";
  }
  if (scheme)  setNumberingScheme(scheme);

  double newTimeFactor = 1./m_p.getParameter<double>("TimeSliceUnit");
  edm::LogInfo("MtdSim") << "New time factor = " << newTimeFactor;
  setTimeFactor(newTimeFactor);

  edm::LogVerbatim("MtdSim") << "MtdSD: Instantiation completed for "
			     << name << " of type " << type;
}

MtdSD::~MtdSD() { 
}

uint32_t MtdSD::setDetUnitId(const G4Step * aStep) { 
  if (numberingScheme == nullptr) {
    return MTDDetId();
  } else {
    getBaseNumber(aStep);
    return numberingScheme->getUnitID(theBaseNumber);
  }
}

std::vector<double> MtdSD::getDDDArray(const std::string & str, 
				       const DDsvalues_type & sv) {

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("MtdSim") << "MtdSD : # of " << str
				    << " bins " << nval << " < 1 ==> illegal";
      throw cms::Exception("DDException") << "MtdSD: cannot get array " << str;
    }
    return fvec;
  } else {
    edm::LogError("MtdSim") << "MtdSD: cannot get array " << str;
    throw cms::Exception("DDException") << "MtdSD: cannot get array " << str;
  }
}

void MtdSD::setNumberingScheme(MTDNumberingScheme* scheme) {
  if (scheme != nullptr) {
    edm::LogInfo("MtdSim") << "MtdSD: updates numbering scheme for " 
                            << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

void MtdSD::getBaseNumber(const G4Step* aStep) {

  theBaseNumber.reset();
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int theSize = touch->GetHistoryDepth()+1;
  if ( theBaseNumber.getCapacity() < theSize ) theBaseNumber.setSize(theSize);
  //Get name and copy numbers
  if ( theSize > 1 ) {
    for (int ii = 0; ii < theSize ; ii++) {
      theBaseNumber.addLevel(touch->GetVolume(ii)->GetName(),touch->GetReplicaNumber(ii));
#ifdef EDM_ML_DEBUG
      edm::LogInfo("MtdSim") << "MtdSD::getBaseNumber(): Adding level " << ii
                              << ": " << touch->GetVolume(ii)->GetName() << "["
                              << touch->GetReplicaNumber(ii) << "]";
#endif
    }
  }
}
