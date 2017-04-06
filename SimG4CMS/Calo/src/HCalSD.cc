///////////////////////////////////////////////////////////////////////////////
// File: HCalSD.cc
// Description: Sensitive Detector class for calorimeters
///////////////////////////////////////////////////////////////////////////////

#include "SimG4CMS/Calo/interface/HCalSD.h"
#include "SimG4CMS/Calo/interface/HcalTestNumberingScheme.h"
#include "SimG4CMS/Calo/interface/HFFibreFiducial.h"
#include "SimG4CMS/Calo/interface/HcalTestNS.h"
#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DetectorDescription/Core/interface/DDFilter.h"
#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDValue.h"

#include "Geometry/Records/interface/HcalSimNumberingRecord.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "CondFormats/DataRecord/interface/HBHEDarkeningRecord.h"

#include "G4LogicalVolumeStore.hh"
#include "G4LogicalVolume.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4ParticleTable.hh"
#include "G4VProcess.hh"

#include "G4SystemOfUnits.hh"
#include "G4PhysicalConstants.hh"

#include <iostream>
#include <fstream>
#include <iomanip>

//#define EDM_ML_DEBUG
//#define plotDebug

HCalSD::HCalSD(G4String name, const DDCompactView & cpv,
               const SensitiveDetectorCatalog & clg,
               edm::ParameterSet const & p, const SimTrackManager* manager) : 
  CaloSD(name, cpv, clg, p, manager,
         (float)(p.getParameter<edm::ParameterSet>("HCalSD").getParameter<double>("TimeSliceUnit")),
         p.getParameter<edm::ParameterSet>("HCalSD").getParameter<bool>("IgnoreTrackID")), 
  hcalConstants(0), numberingFromDDD(0), numberingScheme(0), showerLibrary(0), 
  hfshower(0), showerParam(0), showerPMT(0), showerBundle(0), m_HBDarkening(nullptr), m_HEDarkening(nullptr),
  m_HFDarkening(nullptr), hcalTestNS_(0), depth_(1) {

  //static SimpleConfigurable<double> bk1(0.013, "HCalSD:BirkC1");
  //static SimpleConfigurable<double> bk2(0.0568,"HCalSD:BirkC2");
  //static SimpleConfigurable<double> bk3(1.75,  "HCalSD:BirkC3");
  // Values from NIM 80 (1970) 239-244: as implemented in Geant3

  edm::ParameterSet m_HC = p.getParameter<edm::ParameterSet>("HCalSD");
  useBirk          = m_HC.getParameter<bool>("UseBirkLaw");
  birk1            = m_HC.getParameter<double>("BirkC1")*(g/(MeV*cm2));
  birk2            = m_HC.getParameter<double>("BirkC2");
  birk3            = m_HC.getParameter<double>("BirkC3");
  useShowerLibrary = m_HC.getParameter<bool>("UseShowerLibrary");
  useParam         = m_HC.getParameter<bool>("UseParametrize");
  testNumber       = m_HC.getParameter<bool>("TestNumberingScheme");
  neutralDensity   = m_HC.getParameter<bool>("doNeutralDensityFilter");
  usePMTHit        = m_HC.getParameter<bool>("UsePMTHits");
  betaThr          = m_HC.getParameter<double>("BetaThreshold");
  eminHitHB        = m_HC.getParameter<double>("EminHitHB")*MeV;
  eminHitHE        = m_HC.getParameter<double>("EminHitHE")*MeV;
  eminHitHO        = m_HC.getParameter<double>("EminHitHO")*MeV;
  eminHitHF        = m_HC.getParameter<double>("EminHitHF")*MeV;
  useFibreBundle   = m_HC.getParameter<bool>("UseFibreBundleHits");
  deliveredLumi    = m_HC.getParameter<double>("DelivLuminosity");
  agingFlagHB      = m_HC.getParameter<bool>("HBDarkening");
  agingFlagHE      = m_HC.getParameter<bool>("HEDarkening");
  bool agingFlagHF = m_HC.getParameter<bool>("HFDarkening");
  useHF            = m_HC.getUntrackedParameter<bool>("UseHF",true);
  bool forTBH2     = m_HC.getUntrackedParameter<bool>("ForTBH2",false);
  useLayerWt       = m_HC.getUntrackedParameter<bool>("UseLayerWt",false);
  std::string file = m_HC.getUntrackedParameter<std::string>("WtFile","None");
  testNS_          = m_HC.getUntrackedParameter<bool>("TestNS",false);
  edm::ParameterSet m_HF  = p.getParameter<edm::ParameterSet>("HFShower");
  applyFidCut             = m_HF.getParameter<bool>("ApplyFiducialCut");

#ifdef EDM_ML_DEBUG
  LogDebug("HcalSim") << "***************************************************" 
                      << "\n"
                      << "*                                                 *"
                      << "\n"
                      << "* Constructing a HCalSD  with name " << name << "\n"
                      << "*                                                 *"
                      << "\n"
                      << "***************************************************";
#endif
  edm::LogInfo("HcalSim") << "HCalSD:: Use of HF code is set to " << useHF
                          << "\nUse of shower parametrization set to "
                          << useParam << "\nUse of shower library is set to " 
                          << useShowerLibrary << "\nUse PMT Hit is set to "
                          << usePMTHit << " with beta Threshold "<< betaThr
                          << "\nUSe of FibreBundle Hit set to "<<useFibreBundle
                          << "\n         Use of Birks law is set to      " 
                          << useBirk << "  with three constants kB = "
                          << birk1 << ", C1 = " << birk2 << ", C2 = " << birk3;
  edm::LogInfo("HcalSim") << "HCalSD:: Suppression Flag " << suppressHeavy
                          << " protons below " << kmaxProton << " MeV,"
                          << " neutrons below " << kmaxNeutron << " MeV and"
                          << " ions below " << kmaxIon << " MeV\n"
                          << "         Threshold for storing hits in HB: "
                          << eminHitHB << " HE: " << eminHitHE << " HO: "
                          << eminHitHO << " HF: " << eminHitHF << "\n"
			  << "Delivered luminosity for Darkening " 
			  << deliveredLumi << " Flag (HE) " << agingFlagHE
              << " Flag (HB) " << agingFlagHB
			  << " Flag (HF) " << agingFlagHF << "\n"
			  << "Application of Fiducial Cut " << applyFidCut
			  << "Flag for test number|neutral density filter "
			  << testNumber << " " << neutralDensity;

  HcalNumberingScheme* scheme;
  if (testNumber || forTBH2) 
    scheme = dynamic_cast<HcalNumberingScheme*>(new HcalTestNumberingScheme(forTBH2));
  else 
    scheme = new HcalNumberingScheme();
  setNumberingScheme(scheme);

  const G4LogicalVolumeStore * lvs = G4LogicalVolumeStore::GetInstance();
  std::vector<G4LogicalVolume *>::const_iterator lvcite;
  G4LogicalVolume* lv;
  std::string attribute, value;
  if (useHF) {
    if (useParam) {
      showerParam = new HFShowerParam(name, cpv, p);
    }  else {
      if (useShowerLibrary) showerLibrary = new HFShowerLibrary(name, cpv, p);
      hfshower  = new HFShower(name, cpv, p, 0);
    }

    // HF volume names
    attribute = "Volume";
    value     = "HF";
    DDSpecificsMatchesValueFilter filter0{DDValue(attribute,value,0)};
    DDFilteredView fv0(cpv,filter0);
    hfNames = getNames(fv0);
    fv0.firstChild();
    DDsvalues_type sv0(fv0.mergedSpecifics());
    std::vector<double> temp =  getDDDArray("Levels",sv0);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute 
                            << " = " << value << " has " << hfNames.size() 
			    << " elements";
    for (unsigned int i=0; i < hfNames.size(); ++i) {
      G4String namv = hfNames[i];
      lv            = 0;
      for(lvcite=lvs->begin(); lvcite!=lvs->end(); lvcite++) 
	if((*lvcite)->GetName()==namv) {
	  lv = (*lvcite);
	  break;
	}
      hfLV.push_back(lv);
      int level = static_cast<int>(temp[i]);
      hfLevels.push_back(level);
      edm::LogInfo("HcalSim") << "HCalSD:  HF[" << i << "] = " << hfNames[i]
                              << " LV " << hfLV[i] << " at level " 
			      << hfLevels[i];
    }
  
    // HF Fibre volume names
    value     = "HFFibre";
    DDSpecificsMatchesValueFilter filter1{DDValue(attribute,value,0)};
    DDFilteredView fv1(cpv,filter1);
    fibreNames = getNames(fv1);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute 
                            << " = " << value << ":";
    for (unsigned int i=0; i<fibreNames.size(); ++i) {
      G4String namv = fibreNames[i];
      lv            = 0;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); ++lvcite) {
        if ((*lvcite)->GetName() == namv) {
          lv = (*lvcite);
          break;
        }
      }
      fibreLV.push_back(lv);
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << fibreNames[i]
                              << " LV " << fibreLV[i];
    }
  
    // HF PMT volume names
    value     = "HFPMT";
    DDSpecificsMatchesValueFilter filter3{DDValue(attribute,value,0)};
    DDFilteredView fv3(cpv,filter3);
    std::vector<G4String> pmtNames = getNames(fv3);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute 
			    << " = " << value << " have " << pmtNames.size() 
			    << " entries";
    for (unsigned int i=0; i<pmtNames.size(); ++i)  {
      G4String namv = pmtNames[i];
      lv            = 0;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); ++lvcite) 
        if ((*lvcite)->GetName() == namv) {
	  lv = (*lvcite);
	  break;
	}
      pmtLV.push_back(lv);
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << pmtNames[i]
                              << " LV " << pmtLV[i];
    }
    if (pmtNames.size() > 0) showerPMT = new HFShowerPMT (name, cpv, p);
  
    // HF Fibre bundles
    value     = "HFFibreBundleStraight";
    DDSpecificsMatchesValueFilter filter4{DDValue(attribute,value,0)};
    DDFilteredView fv4(cpv,filter4);
    std::vector<G4String> fibreNames = getNames(fv4);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute
                            << " = " << value << " have " << fibreNames.size()
                            << " entries";
    for (unsigned int i=0; i<fibreNames.size(); ++i) {
      G4String namv = fibreNames[i];
      lv            = 0;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); lvcite++) 
        if ((*lvcite)->GetName() == namv) {
	  lv = (*lvcite);
	  break;
	}
      fibre1LV.push_back(lv);
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << fibreNames[i]
                              << " LV " << fibre1LV[i];
    }

    // Geometry parameters for HF
    value     = "HFFibreBundleConical";
    DDSpecificsMatchesValueFilter filter5{DDValue(attribute,value,0)};
    DDFilteredView fv5(cpv,filter5);
    fibreNames = getNames(fv5);
    edm::LogInfo("HcalSim") << "HCalSD: Names to be tested for " << attribute
			    << " = " << value << " have " << fibreNames.size() 
			    << " entries";
    for (unsigned int i=0; i<fibreNames.size(); ++i) {
      G4String namv = fibreNames[i];
      lv            = 0;
      for (lvcite = lvs->begin(); lvcite != lvs->end(); ++lvcite) 
	if ((*lvcite)->GetName() == namv) {
	  lv = (*lvcite);
	  break;
	}
      fibre2LV.push_back(lv);
      edm::LogInfo("HcalSim") << "HCalSD:  (" << i << ") " << fibreNames[i]
                              << " LV " << fibre2LV[i];
    }
    if (fibre1LV.size() > 0 || fibre2LV.size() > 0) 
      showerBundle = new HFShowerFibreBundle (name, cpv, p);
  }

  //Material list for HB/HE/HO sensitive detectors
  const G4MaterialTable * matTab = G4Material::GetMaterialTable();
  std::vector<G4Material*>::const_iterator matite;
  attribute = "OnlyForHcalSimNumbering"; 
  DDSpecificsHasNamedValueFilter filter2{attribute};
  DDFilteredView fv2(cpv,filter2);
  bool dodet = fv2.firstChild();
  DDsvalues_type sv(fv2.mergedSpecifics());

  while (dodet) {
    const DDLogicalPart & log = fv2.logicalPart();
    G4String namx = log.name().name();
    if (!isItHF(namx) && !isItFibre(namx)) {
      bool notIn = true;
      for (unsigned int i=0; i<matNames.size(); ++i) {
        if (!strcmp(matNames[i].c_str(),log.material().name().name().c_str())){
          notIn = false;
          break;
        }
      }
      if (notIn) {
        namx = log.material().name().name();
        matNames.push_back(namx);
        G4Material* mat = nullptr;
        for (matite = matTab->begin(); matite != matTab->end(); ++matite) {
          if ((*matite)->GetName() == namx) {
            mat = (*matite);
            break;
          }
        }
        materials.push_back(mat);
      }
    }
    dodet = fv2.next();
  }

  edm::LogInfo("HcalSim") << "HCalSD: Material names for " << attribute 
                          << " = " << name << ":";
  for (unsigned int i=0; i<matNames.size(); ++i)
    edm::LogInfo("HcalSim") << "HCalSD: (" << i << ") " << matNames[i]
                            << " pointer " << materials[i];

  mumPDG = mupPDG = 0;
  
  if (useLayerWt) readWeightFromFile(file);

  for (int i=0;  i<9; ++i) hit_[i] = time_[i]= dist_[i] = 0;
  hzvem = hzvhad = 0;

  if (agingFlagHF) m_HFDarkening.reset(new HFDarkening(m_HC.getParameter<edm::ParameterSet>("HFDarkeningParameterBlock")));
#ifdef plotDebug
  edm::Service<TFileService> tfile;

  if ( tfile.isAvailable() ) {
    static const char * const labels[] = {"HB", "HE", "HO", "HF Absorber", "HF PMT",
                                          "HF Absorber Long", "HF Absorber Short",
                                          "HF PMT Long", "HF PMT Short"};
    TFileDirectory hcDir = tfile->mkdir("ProfileFromHCalSD");
    char name[20], title[60];
    for (int i=0; i<9; ++i) {
      sprintf (title, "Hit energy in %s", labels[i]);
      sprintf (name, "HCalSDHit%d", i);
      hit_[i] = hcDir.make<TH1F>(name, title, 2000, 0., 2000.);
      sprintf (title, "Energy (MeV)");
      hit_[i]->GetXaxis()->SetTitle(title);
      hit_[i]->GetYaxis()->SetTitle("Hits");
      sprintf (title, "Time of the hit in %s", labels[i]);
      sprintf (name, "HCalSDTime%d", i);
      time_[i] = hcDir.make<TH1F>(name, title, 2000, 0., 2000.);
      sprintf (title, "Time (ns)");
      time_[i]->GetXaxis()->SetTitle(title);
      time_[i]->GetYaxis()->SetTitle("Hits");
      sprintf (title, "Longitudinal profile in %s", labels[i]);
      sprintf (name, "HCalSDDist%d", i);
      dist_[i] = hcDir.make<TH1F>(name, title, 2000, 0., 2000.);
      sprintf (title, "Distance (mm)");
      dist_[i]->GetXaxis()->SetTitle(title);
      dist_[i]->GetYaxis()->SetTitle("Hits");
    }
    if (useHF && (!useParam)) {
      hzvem  = hcDir.make<TH1F>("hzvem", "Longitudinal Profile (EM Part)",330,0.0,1650.0);
      hzvem->GetXaxis()->SetTitle("Longitudinal Profile (EM Part)");
      hzvhad = hcDir.make<TH1F>("hzvhad","Longitudinal Profile (Had Part)",330,0.0,1650.0);
      hzvhad->GetXaxis()->SetTitle("Longitudinal Profile (Hadronic Part)");
    }
  }
#endif
}

HCalSD::~HCalSD() { 

  if (numberingFromDDD) delete numberingFromDDD;
  if (numberingScheme)  delete numberingScheme;
  if (showerLibrary)    delete showerLibrary;
  if (hfshower)         delete hfshower;
  if (showerParam)      delete showerParam;
  if (showerPMT)        delete showerPMT;
  if (showerBundle)     delete showerBundle;
  if (hcalTestNS_)      delete hcalTestNS_;
}

bool HCalSD::ProcessHits(G4Step * aStep, G4TouchableHistory * ) {

  NaNTrap( aStep ) ;
  
  if (aStep == NULL) {
    return true;
  } else {
    depth_ = (aStep->GetPreStepPoint()->GetTouchable()->GetReplicaNumber(0))%10;
    G4LogicalVolume* lv =
      aStep->GetPreStepPoint()->GetPhysicalVolume()->GetLogicalVolume();
    G4String nameVolume = lv->GetName();
    if (isItHF(aStep)) {
      G4int parCode = aStep->GetTrack()->GetDefinition()->GetPDGEncoding();
      double weight(1.0);
      if (m_HFDarkening) {
	G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
	double r = hitPoint.perp()/CLHEP::cm;
	double z = std::abs(hitPoint.z())/CLHEP::cm;
	double dose_acquired = 0.;
  if (z>=HFDarkening::lowZLimit && z <= HFDarkening::upperZLimit) {
    unsigned int hfZLayer = (int)((z - HFDarkening::lowZLimit)/5);
    if (hfZLayer >= HFDarkening::upperZLimit) hfZLayer = (HFDarkening::upperZLimit-1);
	  float normalized_lumi = m_HFDarkening->int_lumi(deliveredLumi);
    for (int i = hfZLayer; i != HFDarkening::numberOfZLayers; ++i) {
	    dose_acquired = m_HFDarkening->dose(i,r);
	    weight *= m_HFDarkening->degradation(normalized_lumi*dose_acquired);
	  }
	}
#ifdef EDM_ML_DEBUG
	LogDebug("HcalSim") << "HCalSD: HFLumiDarkening at r = " << r 
			    << ", z = " << z << " Dose " << dose_acquired 
			    << " weight " << weight;
#endif
      }
      if (useParam) {
#ifdef EDM_ML_DEBUG
        LogDebug("HcalSim") << "HCalSD: " << getNumberOfHits()
			    << " hits from parametrization in " << nameVolume 
			    << " for Track " << aStep->GetTrack()->GetTrackID()
			    <<" (" << aStep->GetTrack()->GetDefinition()->GetParticleName() 
			    <<")";
#endif
        getFromParam(aStep, weight);
#ifdef EDM_ML_DEBUG
        LogDebug("HcalSim") << "HCalSD: " << getNumberOfHits() 
			    << " hits afterParamS*";
#endif 
      } else {
        bool notaMuon = true;
        if (parCode == mupPDG || parCode == mumPDG ) notaMuon = false;
        if (useShowerLibrary && notaMuon) {
#ifdef EDM_ML_DEBUG
          LogDebug("HcalSim") << "HCalSD: Starts shower library from " 
                              << nameVolume << " for Track " 
                              << aStep->GetTrack()->GetTrackID() << " ("
                              << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
          getFromLibrary(aStep, weight);
        } else if (isItFibre(lv)) {
#ifdef EDM_ML_DEBUG
          LogDebug("HcalSim") << "HCalSD: Hit at Fibre in " << nameVolume 
                              << " for Track " 
                              << aStep->GetTrack()->GetTrackID() <<" ("
                              << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
          hitForFibre(aStep, weight);
        }
      }
    } else if (isItPMT(lv)) {
#ifdef EDM_ML_DEBUG
      LogDebug("HcalSim") << "HCalSD: Hit from PMT parametrization from " 
                          <<  nameVolume << " for Track " 
                          << aStep->GetTrack()->GetTrackID() << " ("
                          << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
      if (usePMTHit && showerPMT) getHitPMT(aStep);
    } else if (isItStraightBundle(lv) || isItConicalBundle(lv)) {
#ifdef EDM_ML_DEBUG
      LogDebug("HcalSim") << "HCalSD: Hit from FibreBundle from "
                          << nameVolume << " for Track " 
                          << aStep->GetTrack()->GetTrackID() << " ("
                          << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
      if (useFibreBundle && showerBundle) 
	getHitFibreBundle(aStep, isItConicalBundle(lv));
    } else {
#ifdef EDM_ML_DEBUG
      LogDebug("HcalSim") << "HCalSD: Hit from standard path from " 
                          <<  nameVolume << " for Track " 
                          << aStep->GetTrack()->GetTrackID() << " ("
                          << aStep->GetTrack()->GetDefinition()->GetParticleName() << ")";
#endif
      if (getStepInfo(aStep)) {
#ifdef plotDebug
        if (edepositEM+edepositHAD > 0)
          plotProfile(aStep, aStep->GetPreStepPoint()->GetPosition(),
                      edepositEM+edepositHAD,aStep->GetPostStepPoint()->GetGlobalTime(),0);
#endif
        if (hitExists() == false && edepositEM+edepositHAD>0.) currentHit = createNewHit();
      }
    }
    return true;
  }
} 

double HCalSD::getEnergyDeposit(G4Step* aStep) {
  double destep = aStep->GetTotalEnergyDeposit();
  double weight = 1;
  G4Track* theTrack = aStep->GetTrack();

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  uint32_t detid = setDetUnitId(aStep);
  int det(0), ieta(0), phi(0), z(0), lay, depth(-1);
  if (testNumber) {
    HcalTestNumbering::unpackHcalIndex(detid,det,z,depth,ieta,phi,lay);
    if (z==0) z = -1;
  } else {
    HcalDetId hcid(detid);
    det  = hcid.subdetId();
    ieta = hcid.ietaAbs();
    phi  = hcid.iphi();
    z    = hcid.zside();
  }
  lay   = (touch->GetReplicaNumber(0)/10)%100 + 1;
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HcalSim") << "HCalSD: det: " << det << " ieta: "<< ieta 
			  << " iphi: " << phi << " zside " << z << "  lay: " 
			  << lay-2;
#endif 
  if (depth_==0 && (det==1 || det==2) && ((!testNumber) || neutralDensity))
    weight = hcalConstants->getLayer0Wt(det,phi,z);
  if (useLayerWt) {
    G4ThreeVector hitPoint = aStep->GetPreStepPoint()->GetPosition();
    weight = layerWeight(det+2, hitPoint, depth_, lay);
  }

  if (m_HBDarkening && det == 1) {
    float dweight = m_HBDarkening->degradation(deliveredLumi,ieta,lay);
    weight *= dweight;
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD:         >>> HB Lumi: " << deliveredLumi
			    << "    coefficient = " << dweight;
#endif  
  }

  if (m_HEDarkening && det == 2) {
    float dweight = m_HEDarkening->degradation(deliveredLumi,ieta,lay);
    weight *= dweight;
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD:         >>> HE Lumi: " << deliveredLumi
			    << "    coefficient = " << dweight;
#endif  
  }

  if (suppressHeavy) {
    TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
    if (trkInfo) {
      int pdg = theTrack->GetDefinition()->GetPDGEncoding();
      if (!(trkInfo->isPrimary())) {         // Only secondary particles
        double ke = theTrack->GetKineticEnergy()/MeV;
        if ( pdg/1000000000 == 1 && (pdg/10000)%100 > 0 &&
	     (pdg/10)%100 > 0    && ke <kmaxIon   ) weight = 0;
        if ((pdg == 2212) && (ke < kmaxProton))     weight = 0;
        if ((pdg == 2112) && (ke < kmaxNeutron))    weight = 0;
#ifdef EDM_ML_DEBUG
        if (weight == 0) 
          edm::LogInfo("HcalSim") << "HCalSD:Ignore Track " 
                                  << theTrack->GetTrackID() << " Type " 
                                  << theTrack->GetDefinition()->GetParticleName()
                                  << " Kinetic Energy " << ke << " MeV";
#endif
      }
    }
  }
#ifdef EDM_ML_DEBUG
  double weight0 = weight;
#endif
  if (useBirk) {
    G4Material* mat = aStep->GetPreStepPoint()->GetMaterial();
    if (isItScintillator(mat)) weight *= getAttenuation(aStep, birk1, birk2, birk3);
  }
  double wt1 = getResponseWt(theTrack);
  double wt2 = theTrack->GetWeight();
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HcalSim") << "HCalSD: Detector " << det+2 << " Depth " << depth_
                          << " weight " << weight0 << " " << weight << " " << wt1 
			  << " " << wt2; 
#endif
  double edep = weight*wt1*destep;
  if (wt2 > 0.0) { edep *= wt2; }
  return edep;
}

uint32_t HCalSD::setDetUnitId(G4Step * aStep) { 

  G4StepPoint* preStepPoint = aStep->GetPreStepPoint(); 
  const G4VTouchable* touch = preStepPoint->GetTouchable();
  G4ThreeVector hitPoint    = preStepPoint->GetPosition();

  int depth = (touch->GetReplicaNumber(0))%10 + 1;
  int lay   = (touch->GetReplicaNumber(0)/10)%100 + 1;
  int det   = (touch->GetReplicaNumber(1))/1000;

  return setDetUnitId (det, hitPoint, depth, lay);
}

void HCalSD::setNumberingScheme(HcalNumberingScheme * scheme) {
  if (scheme != 0) {
    edm::LogInfo("HcalSim") << "HCalSD: updates numbering scheme for " << GetName();
    if (numberingScheme) delete numberingScheme;
    numberingScheme = scheme;
  }
}

void HCalSD::update(const BeginOfJob * job) {

  const edm::EventSetup* es = (*job)();
  edm::ESHandle<HcalDDDSimConstants>    hdc;
  es->get<HcalSimNumberingRecord>().get(hdc);
  if (hdc.isValid()) {
    hcalConstants = (HcalDDDSimConstants*)(&(*hdc));
  } else {
    edm::LogError("HcalSim") << "HCalSD : Cannot find HcalDDDSimConstant";
    throw cms::Exception("Unknown", "HCalSD") << "Cannot find HcalDDDSimConstant" << "\n";
  }

  numberingFromDDD = new HcalNumberingFromDDD(hcalConstants);

  edm::LogInfo("HcalSim") << "Maximum depth for HF " << hcalConstants->getMaxDepth(2);

  //Special Geometry parameters
  gpar      = hcalConstants->getGparHF();
  edm::LogInfo("HcalSim") << "HCalSD: " << gpar.size()<< " gpar (cm)";
  for (unsigned int ig=0; ig<gpar.size(); ig++)
    edm::LogInfo("HcalSim") << "HCalSD: gpar[" << ig << "] = "
			    << gpar[ig]/cm << " cm";
  //Test Hcal Numbering Scheme
  if (testNS_) hcalTestNS_ = new HcalTestNS(es);

  if (agingFlagHB) {
    edm::ESHandle<HBHEDarkening> hdark;
    es->get<HBHEDarkeningRecord>().get("HB",hdark);
    m_HBDarkening = &*hdark;
  }
  if (agingFlagHE) {
    edm::ESHandle<HBHEDarkening> hdark;
    es->get<HBHEDarkeningRecord>().get("HE",hdark);
    m_HEDarkening = &*hdark;
  }
}

void HCalSD::initRun() {
  G4ParticleTable * theParticleTable = G4ParticleTable::GetParticleTable();
  G4String          particleName;
  mumPDG = theParticleTable->FindParticle(particleName="mu-")->GetPDGEncoding();
  mupPDG = theParticleTable->FindParticle(particleName="mu+")->GetPDGEncoding();
#ifdef EDM_ML_DEBUG
  LogDebug("HcalSim") << "HCalSD: Particle code for mu- = " << mumPDG
		      << " for mu+ = " << mupPDG;
#endif
  if (showerLibrary) showerLibrary->initRun(theParticleTable,hcalConstants);
  if (showerParam)   showerParam->initRun(theParticleTable,hcalConstants);
  if (hfshower)      hfshower->initRun(theParticleTable,hcalConstants);
  if (showerPMT)     showerPMT->initRun(theParticleTable,hcalConstants);
  if (showerBundle)  showerBundle->initRun(theParticleTable,hcalConstants);
}

bool HCalSD::filterHit(CaloG4Hit* aHit, double time) {
  double threshold=0;
  DetId theId(aHit->getUnitID());
  switch (theId.subdetId()) {
  case HcalBarrel:
    threshold = eminHitHB; break;
  case HcalEndcap:
    threshold = eminHitHE; break;
  case HcalOuter:
    threshold = eminHitHO; break;
  case HcalForward:
    threshold = eminHitHF; break;
  default:
    break;
  }
  return ((time <= tmaxHit) && (aHit->getEnergyDeposit() > threshold));
}

uint32_t HCalSD::setDetUnitId (int det, const G4ThreeVector& pos, int depth, int lay=1) { 
  uint32_t id = 0;
  if (numberingFromDDD) {
    //get the ID's as eta, phi, depth, ... indices
    HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, pos, depth, lay);
    id = setDetUnitId(tmp);
  }
  return id;
}

uint32_t HCalSD::setDetUnitId (HcalNumberingFromDDD::HcalID& tmp) { 
  modifyDepth(tmp);
  uint32_t id = (numberingScheme) ? numberingScheme->getUnitID(tmp) : 0;
  if ((!testNumber) && hcalTestNS_) {
    bool ok = hcalTestNS_->compare(tmp,id);
    if (!ok)
      edm::LogWarning("HcalSim") << "Det ID from HCalSD " << HcalDetId(id) 
				 << " " << std::hex << id << std::dec 
				 << " does not match one from relabller for " 
				 << tmp.subdet << ":" << tmp.etaR << ":"
				 << tmp.phi << ":" << tmp.phis << ":" 
				 << tmp.depth << ":" << tmp.lay << std::endl;
  }
  return id;
}

std::vector<double> HCalSD::getDDDArray(const std::string & str,
                                        const DDsvalues_type & sv) {
#ifdef EDM_ML_DEBUG
  LogDebug("HcalSim") << "HCalSD:getDDDArray called for " << str;
#endif
  DDValue value(str);
  if (DDfetch(&sv,value)) {
#ifdef EDM_ML_DEBUG
    LogDebug("HcalSim") << value;
#endif
    const std::vector<double> & fvec = value.doubles();
    int nval = fvec.size();
    if (nval < 1) {
      edm::LogError("HcalSim") << "HCalSD : # of " << str << " bins " << nval
          << " < 2 ==> illegal";
      throw cms::Exception("Unknown", "HCalSD") << "nval < 2 for array " << str << "\n";
    }
    
    return fvec;
  } else {
    edm::LogError("HcalSim") << "HCalSD :  cannot get array " << str;
    throw cms::Exception("Unknown", "HCalSD") << "cannot get array " << str << "\n";
  }
}

std::vector<G4String> HCalSD::getNames(DDFilteredView& fv) {

  std::vector<G4String> tmp;
  bool dodet = fv.firstChild();
  while (dodet) {
    const DDLogicalPart & log = fv.logicalPart();
    bool ok = true;

    for (unsigned int i=0; i<tmp.size(); ++i) {
      if (!strcmp(tmp[i].c_str(), log.name().name().c_str())) {
        ok = false;
        break;
      }
    }
    if (ok) tmp.push_back(log.name().name());
    dodet = fv.next();
  }
  return tmp;
}

bool HCalSD::isItHF(G4Step * aStep) {
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int levels = (touch->GetHistoryDepth()) + 1;
  for (unsigned int it=0; it < hfNames.size(); ++it) {
    if (levels >= hfLevels[it]) {
      G4LogicalVolume* lv = touch->GetVolume(levels-hfLevels[it])->GetLogicalVolume();
      if (lv == hfLV[it]) return true;
    }
  }
  return false;
}

bool HCalSD::isItHF (G4String name) {
  std::vector<G4String>::const_iterator it = hfNames.begin();
  for (; it != hfNames.end(); ++it) if (name == *it) return true;
  return false;
}

bool HCalSD::isItFibre (G4LogicalVolume* lv) {
  std::vector<G4LogicalVolume*>::const_iterator ite = fibreLV.begin();
  for (; ite != fibreLV.end(); ++ite) if (lv == *ite) return true;
  return false;
}

bool HCalSD::isItFibre (G4String name) {
  std::vector<G4String>::const_iterator it = fibreNames.begin();
  for (; it != fibreNames.end(); ++it) if (name == *it) return true;
  return false;
}

bool HCalSD::isItPMT (G4LogicalVolume* lv) {
  std::vector<G4LogicalVolume*>::const_iterator ite = pmtLV.begin();
  for (; ite != pmtLV.end(); ++ite) if (lv == *ite) return true;
  return false;
}

bool HCalSD::isItStraightBundle (G4LogicalVolume* lv) {
  std::vector<G4LogicalVolume*>::const_iterator ite = fibre1LV.begin();
  for (; ite != fibre1LV.end(); ++ite) if (lv == *ite) return true;
  return false;
}

bool HCalSD::isItConicalBundle (G4LogicalVolume* lv) {
  std::vector<G4LogicalVolume*>::const_iterator ite = fibre2LV.begin();
  for (; ite != fibre2LV.end(); ++ite) if (lv == *ite) return true;
  return false;
}

bool HCalSD::isItScintillator (G4Material* mat) {
  std::vector<G4Material*>::const_iterator ite = materials.begin();
  for (; ite != materials.end(); ++ite) if (mat == *ite) return true;
  return false;
}

bool HCalSD::isItinFidVolume (G4ThreeVector& hitPoint) {
  bool flag = true;
  if (applyFidCut) {
    int npmt = HFFibreFiducial:: PMTNumber(hitPoint);
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD::isItinFidVolume:#PMT= " << npmt 
			    << " for hit point " << hitPoint;
#endif
    if (npmt <= 0) flag = false;
  }
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD::isItinFidVolume: point " << hitPoint
			    << " return flag " << flag;
#endif
  return flag;
}

void HCalSD::getFromLibrary (G4Step* aStep, double weight) {
  preStepPoint  = aStep->GetPreStepPoint(); 
  theTrack      = aStep->GetTrack();   
  int det       = 5;
  bool ok;

  std::vector<HFShowerLibrary::Hit> hits = showerLibrary->getHits(aStep, ok, weight, false);

  double etrack    = preStepPoint->GetKineticEnergy();
  int    primaryID = setTrackID(aStep);

  // Reset entry point for new primary
  posGlobal = preStepPoint->GetPosition();
  resetForNewPrimary(posGlobal, etrack);

  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  if (particleCode==emPDG || particleCode==epPDG || particleCode==gammaPDG) {
    edepositEM  = 1.*GeV;
    edepositHAD = 0.;
  } else {
    edepositEM  = 0.;
    edepositHAD = 1.*GeV;
  }
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HcalSim") << "HCalSD::getFromLibrary " <<hits.size() 
                          << " hits for " << GetName() << " of " << primaryID 
                          << " with " << theTrack->GetDefinition()->GetParticleName() 
                          << " of " << preStepPoint->GetKineticEnergy()/GeV << " GeV";
#endif
  for (unsigned int i=0; i<hits.size(); ++i) {
    G4ThreeVector hitPoint = hits[i].position;
    if (isItinFidVolume (hitPoint)) {
      int depth              = hits[i].depth;
      double time            = hits[i].time;
      unsigned int unitID    = setDetUnitId(det, hitPoint, depth);
      currentID.setID(unitID, time, primaryID, 0);
#ifdef plotDebug
      plotProfile(aStep, hitPoint, 1.0*GeV, time, depth);
      bool emType = false;
      if (particleCode==emPDG || particleCode==epPDG || particleCode==gammaPDG)
	emType = true;
      plotHF(hitPoint,emType);
#endif
   
      // check if it is in the same unit and timeslice as the previous one
      if (currentID == previousID) {
	updateHit(currentHit);
      } else {
	if (!checkHit()) currentHit = createNewHit();
      }
    }
  }

  //Now kill the current track
  if (ok) {
    theTrack->SetTrackStatus(fStopAndKill);
    G4TrackVector tv = *(aStep->GetSecondary());
    for (unsigned int kk=0; kk<tv.size(); ++kk)
      if (tv[kk]->GetVolume() == preStepPoint->GetPhysicalVolume())
        tv[kk]->SetTrackStatus(fStopAndKill);
  }
}

void HCalSD::hitForFibre (G4Step* aStep, double weight) { // if not ParamShower

  preStepPoint  = aStep->GetPreStepPoint();
  theTrack      = aStep->GetTrack();
  int primaryID = setTrackID(aStep);

  int det   = 5;
  std::vector<HFShower::Hit> hits = hfshower->getHits(aStep, weight);

  G4int particleCode = theTrack->GetDefinition()->GetPDGEncoding();
  if (particleCode==emPDG || particleCode==epPDG || particleCode==gammaPDG) {
    edepositEM  = 1.*GeV;
    edepositHAD = 0.;
  } else {
    edepositEM  = 0.;
    edepositHAD = 1.*GeV;
  }
 
#ifdef EDM_ML_DEBUG
  edm::LogInfo("HcalSim") << "HCalSD::hitForFibre " << hits.size() 
			  << " hits for " << GetName() << " of " << primaryID 
			  << " with " << theTrack->GetDefinition()->GetParticleName() 
			  << " of " << preStepPoint->GetKineticEnergy()/GeV 
			  << " GeV in detector type " << det;
#endif
  if (hits.size() > 0) {
    for (unsigned int i=0; i<hits.size(); ++i) {
      G4ThreeVector hitPoint = hits[i].position;
      if (isItinFidVolume (hitPoint)) {
	int depth              = hits[i].depth;
	double time            = hits[i].time;
	unsigned int unitID = setDetUnitId(det, hitPoint, depth);
	currentID.setID(unitID, time, primaryID, 0);
#ifdef plotDebug
	plotProfile(aStep, hitPoint, edepositEM, time, depth);
	bool emType = false;
	if (particleCode==emPDG || particleCode==epPDG || particleCode==gammaPDG)
	  emType = true;
	plotHF(hitPoint,emType);
#endif
	// check if it is in the same unit and timeslice as the previous one
	if (currentID == previousID) {
	  updateHit(currentHit);
	} else {
	  posGlobal = preStepPoint->GetPosition();
	  if (!checkHit()) currentHit = createNewHit();
	}
      }
    }
  }
}

void HCalSD::getFromParam (G4Step* aStep, double weight) {
  std::vector<HFShowerParam::Hit> hits = showerParam->getHits(aStep, weight);
  int nHit = static_cast<int>(hits.size());

  if (nHit > 0) {
    preStepPoint  = aStep->GetPreStepPoint();
    int primaryID = setTrackID(aStep);
   
    int det   = 5;
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD::getFromParam " << nHit << " hits for " 
                            << GetName() << " of " << primaryID << " with " 
                            <<  aStep->GetTrack()->GetDefinition()->GetParticleName()
                            << " of " << preStepPoint->GetKineticEnergy()/GeV 
                            << " GeV in detector type " << det;
#endif
    for (int i = 0; i<nHit; ++i) {
      G4ThreeVector hitPoint = hits[i].position;
      int depth              = hits[i].depth;
      double time            = hits[i].time;
      unsigned int unitID    = setDetUnitId(det, hitPoint, depth);
      currentID.setID(unitID, time, primaryID, 0);
      edepositEM             = hits[i].edep*GeV; 
      edepositHAD            = 0.;
#ifdef plotDebug
      plotProfile(aStep, hitPoint, edepositEM, time, depth);
#endif

      // check if it is in the same unit and timeslice as the previous one
      if (currentID == previousID) {
	updateHit(currentHit);
      } else {
        posGlobal = preStepPoint->GetPosition();
        if (!checkHit()) currentHit = createNewHit();
      }
    }
  }
}

void HCalSD::getHitPMT (G4Step * aStep) {

  preStepPoint = aStep->GetPreStepPoint();
  theTrack     = aStep->GetTrack();
  double edep  = showerPMT->getHits(aStep);

  if (edep >= 0) {
    double etrack    = preStepPoint->GetKineticEnergy();
    int    primaryID = 0;
    if (etrack >= energyCut) {
      primaryID    = theTrack->GetTrackID();
    } else {
      primaryID    = theTrack->GetParentID();
      if (primaryID == 0) primaryID = theTrack->GetTrackID();
    }
    // Reset entry point for new primary
    posGlobal = preStepPoint->GetPosition();
    resetForNewPrimary(posGlobal, etrack);

    //
    int    det      = static_cast<int>(HcalForward);
    G4ThreeVector hitPoint = preStepPoint->GetPosition();   
    double rr       = (hitPoint.x()*hitPoint.x() + hitPoint.y()*hitPoint.y());
    double phi      = (rr == 0. ? 0. :atan2(hitPoint.y(),hitPoint.x()));
    double etaR     = showerPMT->getRadius();
    int depth       = 1;
    if (etaR < 0) {
      depth         = 2;
      etaR          =-etaR;
    }
    if (hitPoint.z() < 0) etaR =-etaR;
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD::Hit for Detector " << det << " etaR "
                            << etaR << " phi " << phi/deg << " depth " <<depth;
#endif
    double time = (aStep->GetPostStepPoint()->GetGlobalTime());
    uint32_t unitID = 0;
    if (numberingFromDDD) {
      HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det,etaR,phi,
								  depth,1);
      unitID = setDetUnitId(tmp);
    }
    currentID.setID(unitID, time, primaryID, 1);

    edepositHAD = aStep->GetTotalEnergyDeposit();
    edepositEM  =-edepositHAD + (edep*GeV);
#ifdef plotDebug
    plotProfile(aStep, hitPoint, edep*GeV, time, depth);
#endif
#ifdef EDM_ML_DEBUG
    double beta = preStepPoint->GetBeta();
    LogDebug("HcalSim") << "HCalSD::getHitPMT 1 hit for " << GetName() 
                        << " of " << primaryID << " with " 
                        << theTrack->GetDefinition()->GetParticleName()
                        << " of " << preStepPoint->GetKineticEnergy()/GeV 
                        << " GeV with velocity " << beta << " UnitID "
                        << std::hex << unitID << std::dec;
#endif
    // check if it is in the same unit and timeslice as the previous one
    if      (currentID == previousID) {
      updateHit(currentHit);
    } else {
      if (!checkHit()) currentHit = createNewHit();
    }
  }
}

void HCalSD::getHitFibreBundle (G4Step* aStep, bool type) {
  preStepPoint = aStep->GetPreStepPoint();
  theTrack     = aStep->GetTrack();
  double edep  = showerBundle->getHits(aStep, type);

  if (edep >= 0) {
    double etrack    = preStepPoint->GetKineticEnergy();
    int    primaryID = 0;
    if (etrack >= energyCut) {
      primaryID    = theTrack->GetTrackID();
    } else {
      primaryID    = theTrack->GetParentID();
      if (primaryID == 0) primaryID = theTrack->GetTrackID();
    }
    // Reset entry point for new primary
    posGlobal = preStepPoint->GetPosition();
    resetForNewPrimary(posGlobal, etrack);

    //
    int    det      = static_cast<int>(HcalForward);
    G4ThreeVector hitPoint = preStepPoint->GetPosition();   
    double rr       = (hitPoint.x()*hitPoint.x() + hitPoint.y()*hitPoint.y());
    double phi      = (rr == 0. ? 0. :atan2(hitPoint.y(),hitPoint.x()));
    double etaR     = showerBundle->getRadius();
    int depth       = 1;
    if (etaR < 0) {
      depth         = 2;
      etaR          =-etaR;
    }
    if (hitPoint.z() < 0) etaR =-etaR;
#ifdef EDM_ML_DEBUG
    LogDebug("HcalSim") << "HCalSD::Hit for Detector " << det << " etaR "
			<< etaR << " phi " << phi/deg << " depth " <<depth;
#endif
    double time = (aStep->GetPostStepPoint()->GetGlobalTime());
    uint32_t unitID = 0;
    if (numberingFromDDD) {
      HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det,etaR,phi,depth,1);
      unitID = setDetUnitId(tmp);
    }
    if (type) currentID.setID(unitID, time, primaryID, 3);
    else      currentID.setID(unitID, time, primaryID, 2);

    edepositHAD = aStep->GetTotalEnergyDeposit();
    edepositEM  =-edepositHAD + (edep*GeV);
#ifdef plotDebug
    plotProfile(aStep, hitPoint, edep*GeV, time, depth);
#endif
#ifdef EDM_ML_DEBUG
    double beta = preStepPoint->GetBeta();
    LogDebug("HcalSim") << "HCalSD::getHitFibreBundle 1 hit for " << GetName() 
                        << " of " << primaryID << " with " 
                        << theTrack->GetDefinition()->GetParticleName()
                        << " of " << preStepPoint->GetKineticEnergy()/GeV 
                        << " GeV with velocity " << beta << " UnitID "
                        << std::hex << unitID << std::dec;
#endif
    // check if it is in the same unit and timeslice as the previous one
    if (currentID == previousID) updateHit(currentHit);
    else if (!checkHit()) currentHit = createNewHit();
  } // non-zero energy deposit
}

int HCalSD::setTrackID (G4Step* aStep) {
  theTrack     = aStep->GetTrack();

  double etrack = preStepPoint->GetKineticEnergy();
  TrackInformation * trkInfo = (TrackInformation *)(theTrack->GetUserInformation());
  int      primaryID = trkInfo->getIDonCaloSurface();
  if (primaryID == 0) {
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD: Problem with primaryID **** set by "
			    << "force to TkID **** " <<theTrack->GetTrackID();
#endif
    primaryID = theTrack->GetTrackID();
  }

  if (primaryID != previousID.trackID())
    resetForNewPrimary(preStepPoint->GetPosition(), etrack);

  return primaryID;
}

void HCalSD::readWeightFromFile(std::string fName) {

  std::ifstream infile;
  int entry=0;
  infile.open(fName.c_str(), std::ios::in);
  if (infile) {
    int    det, zside, etaR, phi, lay;
    double wt;
    while (infile >> det >> zside >> etaR >> phi >> lay >> wt) {
      uint32_t id = HcalTestNumbering::packHcalIndex(det,zside,1,etaR,phi,lay);
      layerWeights.insert(std::pair<uint32_t,double>(id,wt));
      ++entry;
#ifdef EDM_ML_DEBUG
      edm::LogInfo("HcalSim") << "HCalSD::readWeightFromFile:Entry " << entry
                              << " ID " << std::hex << id << std::dec << " ("
                              << det << "/" << zside << "/1/" << etaR << "/"
                              << phi << "/" << lay << ") Weight " << wt;
#endif
    }
    infile.close();
  }
  edm::LogInfo("HcalSim") << "HCalSD::readWeightFromFile: reads " << entry
                          << " weights from " << fName;
  if (entry <= 0) useLayerWt = false;
}

double HCalSD::layerWeight(int det, const G4ThreeVector& pos, int depth, int lay) { 

  double wt = 1.;
  if (numberingFromDDD) {
    //get the ID's as eta, phi, depth, ... indices
    HcalNumberingFromDDD::HcalID tmp = numberingFromDDD->unitID(det, pos, 
								depth, lay);
    modifyDepth(tmp);
    uint32_t id = HcalTestNumbering::packHcalIndex(tmp.subdet, tmp.zside, 1,
                                                   tmp.etaR, tmp.phis,tmp.lay);
    std::map<uint32_t,double>::const_iterator ite = layerWeights.find(id);
    if (ite != layerWeights.end()) wt = ite->second;
#ifdef EDM_ML_DEBUG
    edm::LogInfo("HcalSim") << "HCalSD::layerWeight: ID " << std::hex << id 
                            << std::dec << " (" << tmp.subdet << "/"  
                            << tmp.zside << "/1/" << tmp.etaR << "/" 
                            << tmp.phis << "/"  << tmp.lay << ") Weight " <<wt;
#endif
  }
  return wt;
}

void HCalSD::plotProfile(G4Step* aStep,const G4ThreeVector& global, double edep,
                         double time, int id) { 

  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  static const G4String modName[8] = {"HEModule", "HVQF" , "HBModule", "MBAT",
				      "MBBT"    , "MBBTC", "MBBT_R1P", "MBBT_R1M"};
  G4ThreeVector local;
  bool found=false;
  double depth=-2000;
  int idx = 4;
  for (int n=0; n<touch->GetHistoryDepth(); ++n) {
    G4String name = touch->GetVolume(n)->GetName();
#ifdef EDM_ML_DEBUG
    LogDebug("HcalSim") << "plotProfile Depth " << n << " Name " << name;
#endif
    for (unsigned int ii=0; ii<8; ++ii) {
      if (name == modName[ii]) {
	found = true;
	int dn = touch->GetHistoryDepth() - n;
	local = touch->GetHistory()->GetTransform(dn).TransformPoint(global);
	if      (ii == 0) {depth = local.z() - 4006.5; idx = 1;}
	else if (ii == 1) {depth = local.z() + 825.0;  idx = 3;}
	else if (ii == 2) {depth = local.x() - 1775.;  idx = 0;}
	else              {depth = local.y() + 15.;    idx = 2;}
	break;
      }
    }
    if (found) break;
  }
  if (!found) depth = std::abs(global.z()) - 11500;
#ifdef EDM_ML_DEBUG
  LogDebug("HcalSim") << "plotProfile Found " << found << " Global " << global
                      << " Local " << local << " depth " << depth << " ID " 
		      << id << " EDEP " << edep << " Time " << time;
#endif
  if (hit_[idx]  != 0) hit_[idx]->Fill(edep);
  if (time_[idx] != 0) time_[idx]->Fill(time,edep);
  if (dist_[idx] != 0) dist_[idx]->Fill(depth,edep);
  int jd = 2*idx + id - 7;
  if (jd >= 0 && jd < 4) {
    jd += 5;
    if (hit_[jd]  != 0) hit_[jd]->Fill(edep);
    if (time_[jd] != 0) time_[jd]->Fill(time,edep);
    if (dist_[jd] != 0) dist_[jd]->Fill(depth,edep);
  }
}

void HCalSD::plotHF(G4ThreeVector& hitPoint, bool emType) {
  double zv  = std::abs(hitPoint.z()) - gpar[4];
  if (emType) {
    if (hzvem  != 0) hzvem->Fill(zv);
  } else {
    if (hzvhad != 0) hzvhad->Fill(zv);
  }
}

void HCalSD::modifyDepth(HcalNumberingFromDDD::HcalID& id) {
  if (id.subdet == 4) {
    int ieta = (id.zside == 0) ? -id.etaR : id.etaR;
    if (hcalConstants->maxHFDepth(ieta,id.phis) > 2) {
      if (id.depth <= 2) {
	if (G4UniformRand() > 0.5) id.depth += 2;
      }
    }
  } else if ((id.subdet == 1 || id.subdet ==2) && testNumber) {
    if (depth_ == 0) id.depth = 1;
    else             id.depth = 2;
  }
}
