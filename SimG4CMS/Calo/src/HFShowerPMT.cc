///////////////////////////////////////////////////////////////////////////////
// File: HFShowerPMT.cc
// Description: Parametrized version of HF hits
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HFShowerPMT.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "FWCore/Utilities/interface/Exception.h"

#include "G4VPhysicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "CLHEP/Units/PhysicalConstants.h"
#include "CLHEP/Units/SystemOfUnits.h"

HFShowerPMT::HFShowerPMT(std::string & name, const DDCompactView & cpv,
			 edm::ParameterSet const & p) {

  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  pePerGeV                = m_HF.getUntrackedParameter<double>("PEPerGeVPMT",1.0);
  
  G4String attribute = "ReadOutName";
  G4String value     = name;
  DDSpecificsFilter filter0;
  DDValue           ddv0(attribute,value,0);
  filter0.setCriteria(ddv0,DDSpecificsFilter::equals);
  DDFilteredView fv0(cpv);
  fv0.addFilter(filter0);
  if (fv0.firstChild()) {
    DDsvalues_type sv0(fv0.mergedSpecifics());

    //Special Geometry parameters
    rTable   = getDDDArray("rTable",sv0);
    edm::LogInfo("HFShower") << "HFShowerPMT: " << rTable.size() 
			     << " rTable (cm)";
    for (unsigned int ig=0; ig<rTable.size(); ig++)
      edm::LogInfo("HFShower") << "HFShowerPMT: rTable[" << ig << "] = "
			       << rTable[ig]/cm << " cm";
  } else {
    edm::LogError("HFShower") << "HFShowerPMT: cannot get filtered "
			      << " view for " << attribute << " matching "
			      << value;
    throw cms::Exception("Unknown", "HFShowerPMT")
      << "cannot match " << attribute << " to " << name <<"\n";
  }

  attribute = "Volume";
  value     = "HFPMT";
  DDSpecificsFilter filter1;
  DDValue           ddv1(attribute,value,0);
  filter1.setCriteria(ddv1,DDSpecificsFilter::equals);
  DDFilteredView fv1(cpv);
  fv1.addFilter(filter1);
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
    edm::LogInfo("HFShower") << "HFShowerPMT: gets the Index matches for "
			     << neta.size() << " PMTs";
    for (unsigned int ii=0; ii<neta.size(); ii++) 
      edm::LogInfo("HFShower") << "HFShowerPMT: rIndexR[" << ii << "] = "
			       << pmtR1[ii] << " fibreR[" << ii << "] = "
			       << pmtFib1[ii] << " rIndexL[" << ii << "] = "
			       << pmtR2[ii] << " fibreL[" << ii << "] = "
			       << pmtFib2[ii];
  } else {
    edm::LogWarning("HFShower") << "HFShowerPMT: cannot get filtered "
				<< " view for " << attribute << " matching "
				<< value;
  }

}

HFShowerPMT::~HFShowerPMT() {}

double HFShowerPMT::getHits(G4Step * aStep) {

  double edep = aStep->GetTotalEnergyDeposit();
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

  LogDebug("HFShower") << "HFShowerPMT: Box " << boxNo << " PMT "
		       << pmtNo << " Mapped Indices " << indexR << ", "
		       << indexF << " Edeposit " << edep/MeV << " MeV; PE "
		       << edep*pePerGeV/GeV;
  if (indexR >= 0 && indexF > 0) return edep*pePerGeV/GeV;
  else                           return 0;
}
 
double HFShowerPMT::getRadius() {
   
  double r = 0.;
  if (indexR >= 0 && indexR+1 < (int)(rTable.size()))
    r = 0.5*(rTable[indexR]+rTable[indexR+1]);
  else
    LogDebug("HFShower") << "HFShowerPMT::getRadius: R " << indexR
			 << " F " << indexF;
  if (indexF > 3)  r =-r;
  LogDebug("HFShower") << "HFShower: Radius (" << indexR << "/" << indexF 
		       << ") " << r;
  return r;
}

std::vector<double> HFShowerPMT::getDDDArray(const std::string & str, 
					     const DDsvalues_type & sv) {

  LogDebug("HFShower") << "HFShowerPMT:getDDDArray called for " << str;

  DDValue value(str);
  if (DDfetch(&sv,value)) {
    LogDebug("HFShower") << value;
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 2) {
      edm::LogError("HFShower") << "HFShowerPMT : # of " << str 
				<< " bins " << nval << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HFShowerPMT")
	<< "nval < 2 for array " << str << "\n";
    }

    return fvec;
  } else {
    edm::LogError("HFShower") << "HFShowerPMT : cannot get array " << str;
    throw cms::Exception("Unknown", "HFShowerPMT") 
      << "cannot get array " << str << "\n";
  }
}
