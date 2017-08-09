///////////////////////////////////////////////////////////////////////////////
// File: ECalSD.cc
// Description: Sensitive Detector class for electromagnetic calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/ECalSD.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "Geometry/EcalCommonData/interface/EcalBarrelNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "Geometry/EcalCommonData/interface/EcalEndcapNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/EcalPreshowerNumberingScheme.h"
#include "Geometry/EcalCommonData/interface/ESTBNumberingScheme.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDSolid.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "Geometry/EcalCommonData/interface/EcalBaseNumber.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4SystemOfUnits.hh"

#include<algorithm>

//#define EDM_ML_DEBUG

template <class T>
bool any(const std::vector<T> & v, const T &what) {
  return std::find(v.begin(), v.end(), what) != v.end();
}

ECalSD::ECalSD(G4String name, const DDCompactView & cpv,
	       const SensitiveDetectorCatalog & clg,
	       edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager, 
	 (float)(p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("TimeSliceUnit")),
	 p.getParameter<edm::ParameterSet>("ECalSD").getParameter<bool>("IgnoreTrackID")), 
  numberingScheme(0){

  //   static SimpleConfigurable<bool>   on1(false,  "ECalSD:UseBirkLaw");
  //   static SimpleConfigurable<double> bk1(0.00463,"ECalSD:BirkC1");
  //   static SimpleConfigurable<double> bk2(-0.03,  "ECalSD:BirkC2");
  //   static SimpleConfigurable<double> bk3(1.0,    "ECalSD:BirkC3");
  // Values from NIM A484 (2002) 239-244: as implemented in Geant3
  //   useBirk          = on1.value();
  //   birk1            = bk1.value()*(g/(MeV*cm2));
  //   birk2            = bk2.value()*(g/(MeV*cm2))*(g/(MeV*cm2));

  edm::ParameterSet m_EC = p.getParameter<edm::ParameterSet>("ECalSD");
  useBirk      = m_EC.getParameter<bool>("UseBirkLaw");
  useBirkL3    = m_EC.getParameter<bool>("BirkL3Parametrization");
  birk1        = m_EC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2        = m_EC.getParameter<double>("BirkC2");
  birk3        = m_EC.getParameter<double>("BirkC3");
  birkSlope    = m_EC.getParameter<double>("BirkSlope");
  birkCut      = m_EC.getParameter<double>("BirkCut");
  slopeLY      = m_EC.getParameter<double>("SlopeLightYield");
  storeTrack   = m_EC.getParameter<bool>("StoreSecondary");
  crystalMat   = m_EC.getUntrackedParameter<std::string>("XtalMat","E_PbWO4");
  bool isItTB  = m_EC.getUntrackedParameter<bool>("TestBeam", false);
  bool nullNS  = m_EC.getUntrackedParameter<bool>("NullNumbering", false);
  storeRL      = m_EC.getUntrackedParameter<bool>("StoreRadLength", false);
  scaleRL      = m_EC.getUntrackedParameter<double>("ScaleRadLength",1.0);

  //Changes for improved timing simulation
  storeLayerTimeSim = m_EC.getUntrackedParameter<bool>("StoreLayerTimeSim", false);
  
  ageingWithSlopeLY   = m_EC.getUntrackedParameter<bool>("AgeingWithSlopeLY", false);
  if(ageingWithSlopeLY) ageing.setLumies(p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("DelivLuminosity"),
					 p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("InstLuminosity"));

  //Material list for HB/HE/HO sensitive detectors
  std::string attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute,name,0)};
  DDFilteredView fv(cpv,filter);
  fv.firstChild();
  DDsvalues_type sv(fv.mergedSpecifics());
  // Use of Weight
  useWeight= true;
  std::vector<double>      tempD = getDDDArray("EnergyWeight",sv);
  if (tempD.size() > 0) { if (tempD[0] < 0.1) useWeight = false; }
#ifdef EDM_ML_DEBUG
  edm::LogInfo("EcalSim") << "ECalSD:: useWeight " << tempD.size() << ":" 
			  << useWeight << std::endl;
#endif
  std::vector<std::string> tempS = getStringArray("Depth1Name",sv);
  if (tempS.size() > 0) depth1Name = tempS[0];
  else                  depth1Name = " ";
  tempS = getStringArray("Depth2Name",sv);
  if (tempS.size() > 0) depth2Name = tempS[0];
  else                  depth2Name = " ";
#ifdef EDM_ML_DEBUG
  edm::LogInfo("EcalSim") << "Names (Depth 1):" << depth1Name << " (Depth 2):" 
			  << depth2Name << std::endl;
#endif
  EcalNumberingScheme* scheme=0;
  if (nullNS) {
    scheme = 0;
  } else if (name == "EcalHitsEB") {
    scheme = dynamic_cast<EcalNumberingScheme*>(new EcalBarrelNumberingScheme());
    isEB=1;
  } else if (name == "EcalHitsEE") { 
    scheme = dynamic_cast<EcalNumberingScheme*>(new EcalEndcapNumberingScheme());
    isEE=1;
  } else if (name == "EcalHitsES") {
    if (isItTB) scheme = dynamic_cast<EcalNumberingScheme*>(new ESTBNumberingScheme());
    else        scheme = dynamic_cast<EcalNumberingScheme*>(new EcalPreshowerNumberingScheme());
    useWeight = false;
  } else {
    edm::LogWarning("EcalSim") << "ECalSD: ReadoutName not supported";
  }

  if (scheme)  setNumberingScheme(scheme);
#ifdef EDM_ML_DEBUG
  edm::LogInfo("EcalSim") << "Constructing a ECalSD  with name " << GetName();
#endif
  if (useWeight) {
    edm::LogInfo("EcalSim")  << "ECalSD:: Use of Birks law is set to      " 
			     << useBirk << "        with three constants kB = "
			     << birk1 << ", C1 = " << birk2 << ", C2 = " 
			     << birk3 <<"\n         Use of L3 parametrization "
			     << useBirkL3 << " with slope " << birkSlope
			     << " and cut off " << birkCut << "\n"
			     << "         Slope for Light yield is set to "
			     << slopeLY;
  } else {
    edm::LogInfo("EcalSim")  << "ECalSD:: energy deposit is not corrected "
			     << " by Birk or light yield curve";
  }

  edm::LogInfo("EcalSim") << "ECalSD:: Suppression Flag " << suppressHeavy
			  << "\tprotons below " << kmaxProton << " MeV,"
			  << "\tneutrons below " << kmaxNeutron << " MeV"
			  << "\tions below " << kmaxIon << " MeV"
			  << "\n\tDepth1 Name = " << depth1Name
			  << "\tDepth2 Name = " << depth2Name
			  << "\n\tstoreRL" << storeRL << ":" << scaleRL
			  << "\tstoreLayerTimeSim " << storeLayerTimeSim
			  << "\n\ttime Granularity " << p.getParameter<edm::ParameterSet>("ECalSD").getParameter<double>("TimeSliceUnit") << " ns"; 
  if (useWeight) initMap(name,cpv);
#ifdef plotDebug
  edm::Service<TFileService> tfile;
  if ( tfile.isAvailable() ) {
    TFileDirectory ecDir = tfile->mkdir("ProfileFromECalSD");
    static const std::string ctype[4] = {"EB","EBref","EE","EERef"};
    for (int k=0; k<4; ++k) {
      std::string name = "ECLL_"+ctype[k];
      std::string title= "Local vs Global for "+ctype[k];
      double xmin = (k > 1) ? 3000.0 : 1000.0;
      g2L_[k] = ecDir.make<TH2F>(name.c_str(),title.c_str(),100,xmin,
				 xmin+1000.,100,0.0,3000.);
    }
  } else {
    for (int k=0; k<4; ++k) g2L_[k] = 0;
  }
#endif

}

ECalSD::~ECalSD() {
  if (numberingScheme) delete numberingScheme;
}

double ECalSD::getEnergyDeposit(G4Step * aStep) {
  
  if (aStep == NULL) {
    return 0;
  } else {
    preStepPoint        = aStep->GetPreStepPoint();
    G4Track* theTrack   = aStep->GetTrack();
    double wt2  = theTrack->GetWeight();
    G4String nameVolume = preStepPoint->GetPhysicalVolume()->GetName();

    // take into account light collection curve for crystals
    double weight = 1.;
    if (suppressHeavy) {
      TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
      if (trkInfo) {
	int pdg = theTrack->GetDefinition()->GetPDGEncoding();
	if (!(trkInfo->isPrimary())) { // Only secondary particles
	  double ke = theTrack->GetKineticEnergy()/MeV;
	  if (((pdg/1000000000 == 1 && ((pdg/10000)%100) > 0 &&
		((pdg/10)%100) > 0)) && (ke<kmaxIon)) weight = 0;
	  if ((pdg == 2212) && (ke < kmaxProton))     weight = 0;
	  if ((pdg == 2112) && (ke < kmaxNeutron))    weight = 0;
#ifdef EDM_ML_DEBUG
	  if (weight == 0) 
	    edm::LogInfo("EcalSim") << "Ignore Track " << theTrack->GetTrackID()
				    << " Type " << theTrack->GetDefinition()->GetParticleName()
				    << " Kinetic Energy " << ke << " MeV";
#endif
	}
      }
    }
    G4LogicalVolume* lv   = aStep->GetPreStepPoint()->GetTouchable()->GetVolume(0)->GetLogicalVolume();
    if (useWeight && !any(noWeight,lv)) {
      weight *= curve_LY(aStep);
      if (useBirk) {
	if (useBirkL3) weight *= getBirkL3(aStep);
	else           weight *= getAttenuation(aStep, birk1, birk2, birk3);
      }
    }
    double wt1  = getResponseWt(theTrack);
    double edep = aStep->GetTotalEnergyDeposit()*weight*wt1;
    /*
    if(wt2 != 1.0) { 
      edm::LogInfo("EcalSim") << "ECalSD:: " << nameVolume
			      <<" LightWeight= " <<weight << " wt1= " <<wt1
			      << "  wt2= " << wt2 << "  "
			      << " Weighted Energy Deposit " << edep/MeV
			      << " MeV";
      const G4VProcess* pr = theTrack->GetCreatorProcess();
      if (pr) 
	edm::LogInfo("EcalSim") << theTrack->GetDefinition()->GetParticleName()
				<< " " << theTrack->GetKineticEnergy()
				<< " Id=" << theTrack->GetTrackID()
				<< " IdP=" << theTrack->GetParentID()
				<< " from  " << pr->GetProcessName();
      else
	edm::LogInfo("EcalSim") << theTrack->GetDefinition()->GetParticleName()
				<< " " << theTrack->GetKineticEnergy()
				<< " Id=" << theTrack->GetTrackID()
				<< " IdP=" << theTrack->GetParentID();
    }
    */
    if(wt2 > 0.0) { edep *= wt2; }
#ifdef EDM_ML_DEBUG
    edm::LogInfo("EcalSim") << "ECalSD:: " << nameVolume
			    << " Light Collection Efficiency " << weight << ":"
			    << wt1 << " Weighted Energy Deposit " << edep/MeV 
			    << " MeV";
#endif
    return edep;
  } 
}

int ECalSD::getTrackID(G4Track* aTrack) {

  int  primaryID(0);
  bool flag(false);
  if (storeTrack) {
    G4LogicalVolume* lv  = preStepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
    if (any(useDepth1,lv)) {
      flag = true;
    } else if (any(useDepth2,lv)) {
      flag = true;
    }
  }
  if (flag) {
    forceSave = true;
    primaryID = aTrack->GetTrackID();
  } else {
    primaryID = CaloSD::getTrackID(aTrack);
  }
  return primaryID;
}

uint16_t ECalSD::getDepth(G4Step * aStep) {
  G4LogicalVolume* lv   = aStep->GetPreStepPoint()->GetTouchable()->GetVolume(0)->GetLogicalVolume();
  uint16_t depth  = any(useDepth1,lv) ? 1 : (any(useDepth2,lv) ? 2 : 0);
  if (storeRL) {
    auto ite        = xtalLMap.find(lv);
    uint16_t depth1 = (ite == xtalLMap.end()) ? 0 : (((ite->second) >= 0) ? 0 :
						     PCaloHit::kEcalDepthRefz);
    uint16_t depth2 = getRadiationLength(aStep);
    depth          |= (((depth2&PCaloHit::kEcalDepthMask) << PCaloHit::kEcalDepthOffset) | depth1);
  } else if (storeLayerTimeSim) {
    uint16_t depth2 = getLayerIDForTimeSim(aStep);
    depth          |= ((depth2&PCaloHit::kEcalDepthMask) << PCaloHit::kEcalDepthOffset);
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("EcalSim") << "ECalSD::Depth " << std::hex << depth1 << ":"
			      << depth2 << ":" << depth << std::dec << " L "
			      << (ite == xtalLMap.end()) << ":" <<ite->second;
#endif
  return depth;
}

uint16_t ECalSD::getRadiationLength(G4Step * aStep) {
  
  uint16_t thisX0 = 0;
  if (aStep != NULL) {
    G4StepPoint* hitPoint = aStep->GetPreStepPoint();
    G4LogicalVolume* lv   = hitPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
#ifdef EDM_ML_DEBUG
    edm::LogInfo("EcalSim") << lv->GetName() << " useWight " << useWeight;
#endif
    if (useWeight) {
      G4ThreeVector  localPoint = setToLocal(hitPoint->GetPosition(),
					     hitPoint->GetTouchable());
      double radl     = hitPoint->GetMaterial()->GetRadlen();
      double depth    = crystalDepth(lv,localPoint);
      thisX0 = (uint16_t)floor(scaleRL*depth/radl);
#ifdef plotDebug
      std::string lvname = lv->GetName();
      int k1 = (lvname.find("EFRY")!=std::string::npos) ? 2 : 0;
      int k2 = (lvname.find("refl")!=std::string::npos) ? 1 : 0;
      int kk = k1+k2;
      double rz = (k1 == 0) ? (hitPoint->GetPosition()).rho() : 
	std::abs((hitPoint->GetPosition()).z());
      edm::LogVerbatim("EcalSim") << lvname << " # " << k1 << ":" << k2 << ":" 
				  << kk << " rz " << rz << " D " << thisX0;
      g2L_[kk]->Fill(rz,thisX0);
#endif
#ifdef EDM_ML_DEBUG
      double crlength = crystalLength(lv);
      edm::LogVerbatim("EcalSim") << lv->GetName() << " Global " 
				  << hitPoint->GetPosition() << ":" 
				  << (hitPoint->GetPosition()).rho() 
				  << " Local " << localPoint 
				  << " Crystal Length " << crlength 
				  << " Radl " << radl << " DetZ " << detz 
				  << " Index " << thisX0 
				  << " : " << getLayerIDForTimeSim(aStep);
#endif
    } 
  }
  return thisX0;
}

uint16_t ECalSD::getLayerIDForTimeSim(G4Step * aStep) 
{
  float    layerSize = 1*cm; //layer size in cm
  if (!isEB && !isEE)  return 0;

  if (aStep != NULL ) {
    G4StepPoint* hitPoint = aStep->GetPreStepPoint();
    G4LogicalVolume* lv   = hitPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();
    G4ThreeVector  localPoint = setToLocal(hitPoint->GetPosition(),
					   hitPoint->GetTouchable());
    double detz     = crystalDepth(lv,localPoint);
    if (detz<0) detz= 0;
    return (int)detz/layerSize;
  }
  return 0;
}

uint32_t ECalSD::setDetUnitId(G4Step * aStep) { 
  if (numberingScheme == 0) {
    return EBDetId(1,1)();
  } else {
    getBaseNumber(aStep);
    return numberingScheme->getUnitID(theBaseNumber);
  }
}

void ECalSD::setNumberingScheme(EcalNumberingScheme* scheme) {
  if (scheme != 0) {
    edm::LogInfo("EcalSim") << "EcalSD: updates numbering scheme for " 
			    << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}


void ECalSD::initMap(G4String sd, const DDCompactView & cpv) {

  G4String attribute = "ReadOutName";
  DDSpecificsMatchesValueFilter filter{DDValue(attribute,sd,0)};
  DDFilteredView fv(cpv,filter);
  fv.firstChild();

  std::vector<G4LogicalVolume*> lvused;
  const G4LogicalVolumeStore *  lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  std::map<std::string, G4LogicalVolume *> nameMap;
  for (auto lvi = lvs->begin(), lve = lvs->end(); lvi != lve; ++lvi)
    nameMap.insert(std::make_pair((*lvi)->GetName(), *lvi));

  bool dodet=true;
  while (dodet) {
    const std::string &matname = fv.logicalPart().material().name().name();
    const std::string &lvname = fv.logicalPart().name().name();
    G4LogicalVolume* lv = nameMap[lvname];
    int ibec = (lvname.find("EFRY") == std::string::npos) ? 0 : 1;
    int iref = (lvname.find("refl") == std::string::npos) ? 0 : 1;
    int type = (ibec+iref == 1) ? 1 : -1;
    if (depth1Name != " ") {
      if (strncmp(lvname.c_str(), depth1Name.c_str(), 4) == 0) {
	if (!any(useDepth1, lv)) {
 	  useDepth1.push_back(lv);
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("EcalSim") << "ECalSD::initMap Logical Volume " 
				  << lvname <<" in Depth 1 volume list";
#endif
	}
	G4LogicalVolume* lvr = nameMap[lvname + "_refl"];
	if (lvr != 0 && !any(useDepth1, lvr)) {
	  useDepth1.push_back(lvr);
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("EcalSim") << "ECalSD::initMap Logical Volume " 
				  << lvname << "_refl"
				  <<" in Depth 1 volume list";
#endif
	}
      }
    }
    if (depth2Name != " ") {
      if (strncmp(lvname.c_str(), depth2Name.c_str(), 4) == 0) {
	if (!any(useDepth2, lv)) {
	  useDepth2.push_back(lv);
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("EcalSim") << "ECalSD::initMap Logical Volume " 
				  << lvname <<" in Depth 2 volume list";
#endif
	}
	G4LogicalVolume* lvr = nameMap[lvname + "_refl"];
	if (lvr != 0 && !any(useDepth2,lvr)) {
	  useDepth2.push_back(lvr);
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("EcalSim") << "ECalSD::initMap Logical Volume " 
				  << lvname << "_refl"
				  <<" in Depth 2 volume list";
#endif
	}
      }
    }
    if (lv != 0) {
      if (crystalMat.size() == matname.size() && !strcmp(crystalMat.c_str(), matname.c_str())) {
	if (!any(lvused,lv)) {
	  lvused.push_back(lv);
	  const DDSolid & sol  = fv.logicalPart().solid();
	  const std::vector<double> & paras = sol.parameters();
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("EcalSim") << "ECalSD::initMap (for " << sd 
				  << "): Solid " << lvname << " Shape " 
				  << sol.shape() << " Parameter 0 = "
				  << paras[0] << " Logical Volume " << lv;
#endif
	  if (sol.shape() == ddtrap) {
	    double dz = 2*paras[0];
	    xtalLMap.insert(std::pair<G4LogicalVolume*,double>(lv,dz*type));
	    lv = nameMap[lvname + "_refl"];
	    if (lv != 0) {
	      xtalLMap.insert(std::pair<G4LogicalVolume*,double>(lv,-dz*type));
	    }
	  }
	}
      } else {
	if (!any(noWeight,lv)) {
	  noWeight.push_back(lv);
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("EcalSim") << "ECalSD::initMap Logical Volume "
				  << lvname << " Material " << matname 
				  << " in noWeight list";
#endif
	}
	lv = nameMap[lvname];
	if (lv != 0 && !any(noWeight,lv)) {
	  noWeight.push_back(lv);
#ifdef EDM_ML_DEBUG
	  edm::LogInfo("EcalSim") << "ECalSD::initMap Logical Volume " 
				  << lvname << " Material " << matname 
				  << " in noWeight list";
#endif
	}
      }
    }
    dodet = fv.next();
  }
#ifdef EDM_ML_DEBUG
  edm::LogInfo("EcalSim") << "ECalSD: Length Table for " << attribute << " = " 
			  << sd << ":";   
  int i=0;
  for (auto ite : xtalLMap) {
    G4String name("Unknown");
    if (ite.first != 0) name = (ite.first)->GetName();
    edm::LogInfo("EcalSim") << " " << i << " " << ite.first << " " << name 
			    << " L = " << ite.second;
    ++i;
  }
#endif
}

double ECalSD::curve_LY(G4Step* aStep) {

  G4StepPoint*     stepPoint = aStep->GetPreStepPoint();
  G4LogicalVolume* lv        = stepPoint->GetTouchable()->GetVolume(0)->GetLogicalVolume();

  double weight = 1.;
  G4ThreeVector  localPoint = setToLocal(stepPoint->GetPosition(),
					 stepPoint->GetTouchable());

  double crlength = crystalLength(lv);
  double depth    = crystalDepth(lv,localPoint);

  if (ageingWithSlopeLY) {
    //position along the crystal in mm from 0 to 230 (in EB)
    if (depth >= -0.1 || depth <= crlength+0.1)
      weight = ageing.calcLightCollectionEfficiencyWeighted(currentID.unitID(), depth/crlength);
  } else {
    double dapd = crlength - depth;
    if (dapd >= -0.1 || dapd <= crlength+0.1) {
      if (dapd <= 100.)
	weight = 1.0 + slopeLY - dapd * 0.01 * slopeLY;
    } else {
      edm::LogWarning("EcalSim") << "ECalSD: light coll curve : wrong distance "
				 << "to APD " << dapd << " crlength = " 
				 << crlength <<" crystal name = " <<lv->GetName()
				 << " z of localPoint = " << localPoint.z() 
				 << " take weight = " << weight;
    }
  }
  return weight;
}

double ECalSD::crystalLength(G4LogicalVolume* lv) {

  auto ite = xtalLMap.find(lv);
  double length = (ite == xtalLMap.end()) ? 230.0 : std::abs(ite->second);
  return length;
}

double ECalSD::crystalDepth(G4LogicalVolume* lv, 
			    const G4ThreeVector& localPoint) {

  auto ite = xtalLMap.find(lv);
  double depth = (ite == xtalLMap.end()) ? 0 :
    (std::abs(0.5*(ite->second)+localPoint.z()));
//  (0.5*std::abs(ite->second)+localPoint.z());
  return depth;
}

void ECalSD::getBaseNumber(const G4Step* aStep) {

  theBaseNumber.reset();
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int theSize = touch->GetHistoryDepth()+1;
  if ( theBaseNumber.getCapacity() < theSize ) theBaseNumber.setSize(theSize);
  //Get name and copy numbers
  if ( theSize > 1 ) {
    for (int ii = 0; ii < theSize ; ii++) {
      theBaseNumber.addLevel(touch->GetVolume(ii)->GetName(),touch->GetReplicaNumber(ii));
#ifdef EDM_ML_DEBUG
      edm::LogInfo("EcalSim") << "ECalSD::getBaseNumber(): Adding level " << ii
			      << ": " << touch->GetVolume(ii)->GetName() << "["
			      << touch->GetReplicaNumber(ii) << "]";
#endif
    }
  }

}

double ECalSD::getBirkL3(G4Step* aStep) {

  double weight = 1.;
  double charge = aStep->GetPreStepPoint()->GetCharge();

  if (charge != 0. && aStep->GetStepLength() > 0) {
    G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    double density = mat->GetDensity();
    double dedx    = aStep->GetTotalEnergyDeposit()/aStep->GetStepLength();
    double rkb     = birk1/density;
    if (dedx > 0) {
      weight         = 1. - birkSlope*log(rkb*dedx);
      if (weight < birkCut) weight = birkCut;
      else if (weight > 1.) weight = 1.;
    }
#ifdef EDM_ML_DEBUG
    edm::LogInfo("EcalSim") << "ECalSD::getBirkL3 in " << mat->GetName()
			    << " Charge " << charge << " dE/dx " << dedx
			    << " Birk Const " << rkb << " Weight = " << weight 
			    << " dE " << aStep->GetTotalEnergyDeposit();
#endif
  }
  return weight;

}

std::vector<double> ECalSD::getDDDArray(const std::string & str,
					const DDsvalues_type & sv) {

#ifdef EDM_ML_DEBUG
  edm::LogInfo("EcalSim") << "ECalSD:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef EDM_ML_DEBUG
    edm::LogInfo("EcalSim") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    return fvec;
  } else {
    std::vector<double> fvec;
    return fvec;
  }
}

std::vector<std::string> ECalSD::getStringArray(const std::string & str,
						const DDsvalues_type & sv) {

#ifdef EDM_ML_DEBUG
  edm::LogInfo("EcalSim") << "ECalSD:getStringArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef EDM_ML_DEBUG
    edm::LogInfo("EcalSim") << value;
#endif
    const std::vector<std::string> & fvec = value.strings();
    return fvec;
  } else {
    std::vector<std::string> fvec;
    return fvec;
  }
}
