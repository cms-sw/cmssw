///////////////////////////////////////////////////////////////////////////////
// File: ShashlikSD.cc
// Description: Sensitive Detector class for electromagnetic calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/ShashlikSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DataFormats/EcalDetId/interface/EKDetId.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include<algorithm>

//#define DebugLog

template <class T>
bool any(const std::vector<T> & v, const T &what) {
  return std::find(v.begin(), v.end(), what) != v.end();
}

ShashlikSD::ShashlikSD(G4String name, const DDCompactView & cpv,
		       SensitiveDetectorCatalog & clg, 
		       edm::ParameterSet const & p, 
		       const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager, 
	 p.getParameter<edm::ParameterSet>("ShashlikSD").getParameter<int>("TimeSliceUnit"),
	 p.getParameter<edm::ParameterSet>("ShashlikSD").getParameter<bool>("IgnoreTrackID")) {

#ifdef DebugLog
  std::cout << "Constructing a ShashlikSD  with name " << GetName() <<std::endl;
#endif

  edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ShashlikSD");
  useBirk      = m_EC.getParameter<bool>("UseBirkLaw");
  birk1        = m_EC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2        = m_EC.getParameter<double>("BirkC2");
  birk3        = m_EC.getParameter<double>("BirkC3");
  if (useBirk) {
    edm::LogInfo("EcalSim")  << "ShashlikSD:: Use of Birks law is set to      " 
			     << useBirk << "        with three constants kB = "
			     << birk1 << ", C1 = " << birk2 << ", C2 = " 
			     << birk3;
  } else {
    edm::LogInfo("EcalSim")  << "ShashlikSD:: energy deposit is not corrected "
			     << " by Birks law";
  }
  useWeight             = m_EC.getParameter<bool>("UseWeight");
  edm::FileInPath fp    = m_EC.getParameter<edm::FileInPath>("FileName");
  std::string     fname = fp.fullPath();
 
  ///Open map by Sasha and assign the histos 
  TFile *f = TFile::Open(fname.c_str());
  hFibre[0] = (TH2D*)f->Get("h1");
  hFibre[1] = (TH2D*)f->Get("h3");
  hFibre[2] = (TH2D*)f->Get("h2");
  hFibre[3] = (TH2D*)f->Get("h4");
  edm::LogInfo("EcalSim") << "ShashlikSD:: use weight " << useWeight 
			  << " from the four histos read from " << fname;
  for (int i=0; i<4; ++i) 
    edm::LogInfo("EcalSim") << "Histo[" << i << "] " << hFibre[i] << " with x "
			    << hFibre[i]->GetNbinsX() << ":" 
			    << hFibre[i]->GetXaxis()->GetXmin() << ":"
			    << hFibre[i]->GetXaxis()->GetXmax()
			    << " with y " << hFibre[i]->GetNbinsY() << ":" 
			    << hFibre[i]->GetYaxis()->GetXmin() << ":"
			    << hFibre[i]->GetYaxis()->GetXmax();

  roType       = m_EC.getParameter<int>("ReadOutType");
  useAtt       = m_EC.getParameter<bool>("AttCorrection");
  attL         = m_EC.getParameter<double>("AttLength")*CLHEP::mm;
  edm::LogInfo("EcalSim") << "ShashlikSD:: RO Type " << roType << " and "
			  << " correct for attenuation " << useAtt << " using "
			  << attL;

  sdc = new ShashlikDDDConstants(cpv);

  //Length of fiber for each layer
  std::string attribute = "ReadOutName";
  DDSpecificsFilter filter;
  DDValue           ddv(attribute,name,0);
  filter.setCriteria(ddv,DDSpecificsFilter::equals);
  DDFilteredView fv(cpv);
  fv.addFilter(filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());
  fiberL  = getDDDArray("FiberLength",sv);
  std::vector<double> temp = getDDDArray("ModuleLength",sv);
  moduleL = temp[0];
  edm::LogInfo("EcalSim") << "ShashlikSD:: Module length " << moduleL
			  << " distance of layers from back";
  for (unsigned int k=0; k<fiberL.size(); ++k)
    edm::LogInfo("EcalSim") << "Fiber[" << k << "] = " << fiberL[k];
}

ShashlikSD::~ShashlikSD() { }

bool ShashlikSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    preStepPoint     = aStep->GetPreStepPoint(); 
    theTrack         = aStep->GetTrack();   
    double time      = (aStep->GetPostStepPoint()->GetGlobalTime())/nanosecond;
    int    primaryID = getTrackID(theTrack);
    G4int  particleCode = theTrack->GetDefinition()->GetPDGEncoding();
    if (particleCode == emPDG ||
	particleCode == epPDG ||
	particleCode == gammaPDG ) {
      edepositEM  = getEnergyDeposit(aStep);
      edepositHAD = 0.;
    } else {
      edepositEM  = 0.;
      edepositHAD = getEnergyDeposit(aStep);
    }
    if (edepositEM+edepositHAD>0.) {
      double edepEM(edepositEM), edepHad(edepositHAD);
      int            fibMax     = (useWeight) ? 5 : 1;
      int            roMax      = (roType == 0) ? 1 : 2;
      int            layer      = preStepPoint->GetTouchable()->GetReplicaNumber(0);
      G4ThreeVector  localPoint = setToLocal(preStepPoint->GetPosition(),
                                             preStepPoint->GetTouchable());
#ifdef DebugLog
      std::cout << "Global " << preStepPoint->GetPosition() << " Local "
		<< localPoint << std::endl;
#endif
      uint16_t       depth      = getDepth(aStep);
      uint32_t       id0        = setDetUnitId(aStep);
      for (int fib = 0; fib < fibMax; ++fib) {
	double wt1 = (useWeight) ? fiberWt(fib, localPoint) : 1;
	for (int rx = 0; rx < roMax; ++rx) {
	  double   wt2    = fiberLoss(layer, rx);
	  int      ro     = (roType == 0) ? rx : rx+1;
	  EKDetId  id     = EKDetId(id0);
	  id.setFiber(fib,ro);
	  uint32_t unitID = (id.rawId());
	  currentID.setID(unitID, time, primaryID, depth);
	  edepositEM      = edepEM*wt1*wt2;
	  edepositHAD     = edepHad*wt1*wt2;
#ifdef DebugLog
	  std::cout << "ShashlikSD: " << EKDetId(unitID) << " depth " << depth
		    << " Track " << primaryID << " time " << time << " wts "
		    << wt1 << ":" << wt2 << " edep " << (edepEM+edepHad)
		    << std::endl;
#endif
	  // check if it is in the same unit and timeslice as the previous one
	  if (currentID == previousID) {
	    updateHit(currentHit);
	  } else {
	    if (!checkHit()) currentHit = createNewHit();
	  }
	} // end of loop over RO type
      } // end of loop over fibers
    } // Make hits
    return true;
  } // Good step
}

double ShashlikSD::getEnergyDeposit(G4Step * aStep) {
  
  if (aStep == NULL) {
    return 0;
  } else {
    preStepPoint        = aStep->GetPreStepPoint();
    G4Track* theTrack   = aStep->GetTrack();
    double wt2          = theTrack->GetWeight();

    // take into account Birk's correction for crystals
    double weight = 1.;
    if (useBirk) {
      weight *= getAttenuation(aStep, birk1, birk2, birk3);
    }
    double edep = aStep->GetTotalEnergyDeposit()*weight;
    if (wt2 > 0.0)  
      edep *= wt2;
#ifdef DebugLog
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();
    std::cout << "ShashlikSD:: " << nameVolume << " Birk correction factor "
	      << weight << " track wt " << wt2 << " Weighted Energy Deposit " 
	      << edep/CLHEP::MeV << " MeV" << std::endl;
#endif
    return edep;
  } 
}

uint16_t ShashlikSD::getDepth(G4Step *aStep) {

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  uint16_t ret = (uint16_t)(touch->GetReplicaNumber(0));
#ifdef DebugLog
  std::cout << "ShashlikSD::Volume " << touch->GetVolume(0)->GetName() 
	    << " Depth " << ret << std::endl;
#endif
  return ret;
}

uint32_t ShashlikSD::setDetUnitId(G4Step *aStep) { 

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int module = touch->GetReplicaNumber(1);
  int ism    = touch->GetReplicaNumber(2);
  int iz     = ((aStep->GetPreStepPoint()->GetPosition()).z() > 0) ? 1 : -1;
  std::pair<int,int> ixy = sdc->getXY(ism,module);
#ifdef DebugLog
  int theSize = touch->GetHistoryDepth()+1;
  std::cout << "ShaslikSD:: ISM|Module|IZ " << ism << ":" << module << ":"
	    << iz << " touchable size " << theSize;
  for (int ii = 0; ii < theSize ; ii++)
    std::cout << " [" << ii << "]: " << touch->GetVolume(ii)->GetName() 
	      << "(" << touch->GetReplicaNumber(ii) << ")";
  std::cout << std::endl;
#endif
  return EKDetId(ixy.first,ixy.second,0,0,iz).rawId();
}

G4double ShashlikSD::fiberWt(G4int k, G4ThreeVector localPoint) {

  double wt(0);
  double xx = localPoint.getX()/CLHEP::cm;
  double yy = localPoint.getY()/CLHEP::cm;
  if (k > 0) {
    int ibinx = hFibre[k-1]->GetXaxis()->FindBin(xx);
    int ibiny = hFibre[k-1]->GetYaxis()->FindBin(yy);
    wt = hFibre[k-1]->GetBinContent(ibinx,ibiny);
#ifdef DebugLog
    std::cout << "Localpoint.X and localPoint.Y and wt : "
	      << xx << ":" << yy << ":" << localPoint.getZ()/cm << ":" 
	      << ibinx << ":" << ibiny << ":" << wt << std::endl;
#endif
  } else {
    for (int i=0; i<4; ++i) {
      int ibinx = hFibre[i]->GetXaxis()->FindBin(xx);
      int ibiny = hFibre[i]->GetYaxis()->FindBin(yy);
      double wt0= hFibre[i]->GetBinContent(ibinx,ibiny);
      wt += wt0;
#ifdef DebugLog
      std::cout << "Localpoint.X and localPoint.Y and wt : "
		<< xx << ":" << yy << ":" << localPoint.getZ()/cm << ":" 
		<< ibinx << ":"	<< ibiny << ":" << wt0 << ":" << wt <<std::endl;
#endif
    }
  }
  return wt;
}

G4double ShashlikSD::fiberLoss(G4int layer, G4int rx) {
  
  double wt(1.0);
  if (useAtt) {
    if (roType == 0) {
      wt = 0.5*(exp(-(moduleL-fiberL[layer-1])/attL)+
		exp(-(moduleL+fiberL[layer-1])/attL));
    } else {
      double fbl = (rx == 0) ? moduleL-fiberL[layer-1] : fiberL[layer-1];
      wt = exp(-fbl/attL);
    }
  } else {
    if (roType != 0) wt = 0.5;
  }
  return wt;
}

std::vector<double> ShashlikSD::getDDDArray(const std::string & str,
					    const DDsvalues_type & sv) {

#ifdef DebugLog
  std::cout << "ShashlikSD:getDDDArray called for " << str << std::endl;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef DebugLog
    std::cout << value << std::endl;
#endif
    const std::vector<double> & fvec = value.doubles();
    return fvec;
  } else {
    std::vector<double> fvec;
    return fvec;
  }
}
