#include "SimG4Core/GFlash/interface/CaloModel.h"
#include "SimG4Core/GFlash/interface/GflashEMShowerModel.h"
#include "SimG4Core/GFlash/interface/GflashHadronShowerModel.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4Electron.hh"
#include "G4FastSimulationManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "GFlashHomoShowerParameterisation.hh"

#include "SimG4Core/G4gflash/src/GFlashHitMaker.hh"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "GFlashParticleBounds.hh"

CaloModel::CaloModel(G4LogicalVolumeToDDLogicalPartMap& map,
		     const edm::ParameterSet & p) : map_(map), m_pCaloModel(p) 
{ 
  build();
}

CaloModel::~CaloModel()
{
  if(theHadronShowerModel!=0) delete theHadronShowerModel;
}

void CaloModel::build()
{
  // material declarations - a=mass of a mole, z=mean number of protons
                                                                                
  G4double a;
  G4double z;
  G4String name;
  G4String symbol;
  G4double density;
  G4double fractionmass;
  G4int ncomponents;
                                                                                
  density = 11.35*g/cm3;
  a = 207.19*g/mole;
  G4Element* Pb = new G4Element(name="Lead" , symbol="Pb"   , z=82., a );
                                                                                
  density = 19.3*g/cm3;
  a = 183.85*g/mole;
  G4Element* W = new G4Element(name="Tungsten",symbol="W" , z= 74., a );
                                                                                
  density = 1.43*mg/cm3;
  a = 15.999*g/mole;
  G4Element* O  = new G4Element(name="Oxygen"  ,symbol="O" , z= 8., a);
                                                                                
  density = 8.28*g/cm3;
  G4Material* PbWO4 = new G4Material("PbWO4"  , density, ncomponents=3);
                                                                                
  PbWO4->AddElement(Pb, fractionmass=0.45532661);
  PbWO4->AddElement(O, fractionmass=0.14063942);
  PbWO4->AddElement(W, fractionmass=0.40403397);

  //add 90-10 Brass (Cupper+Zn)
                                                                                
  density = 8.960*g/cm3;
  a = 63.55*g/mole;
  G4Element* Cu  = new G4Element(name="Copper"  ,symbol="Cu" , z= 29., a);
                                                                                
  density = 7.14*g/cm3;
  a = 65.409*g/mole;
  G4Element* Zn  = new G4Element(name="Zinc"  ,symbol="Zn" , z= 30., a);
                                                                                
  density = 8.53*g/cm3;
  G4Material* Brass = new G4Material("Brass"  ,density, ncomponents=2);
  Brass->AddElement(Cu,fractionmass=0.9);
  Brass->AddElement(Zn,fractionmass=0.1);

  //Logical volumes for envelopes

  G4LogicalVolume * ecal_barrel_log = 0;
  G4LogicalVolume * ecal_ecap_log   = 0;
                                                                                
  G4LogicalVolume * hcal_barrel_log = 0;
  G4LogicalVolume * hcal_ecap_log   = 0;

  //Finding correct G4LogicalVolume for parameterisation
  G4LogicalVolumeToDDLogicalPartMap::Vector vec = map_.all("volumes");

  for (G4LogicalVolumeToDDLogicalPartMap::Vector::iterator tit = vec.begin(); 
       tit != vec.end(); tit++){
    if (((*tit).first)->GetName()=="ESPM"){  
      ecal_barrel_log = (*tit).first;
      ecal_barrel_log->SetMaterial(PbWO4);
    }
    if (((*tit).first)->GetName()=="ENCA"){
      ecal_ecap_log = (*tit).first;
      ecal_ecap_log->SetMaterial(PbWO4);
    }
    if (((*tit).first)->GetName()=="HB"){
      hcal_barrel_log = (*tit).first;
      hcal_barrel_log->SetMaterial(Brass);
    }
    if (((*tit).first)->GetName()=="HE"){
      hcal_ecap_log = (*tit).first;
      hcal_ecap_log->SetMaterial(Brass);
    }
  }
  
  if ((ecal_barrel_log) && (ecal_ecap_log) && (hcal_barrel_log) && (hcal_ecap_log)) {

    // Parameterisaiton components
    theParticleBounds  = new GFlashParticleBounds();
    theHMaker          = new GFlashHitMaker();
    
    //    G4Region* aRegion = new G4Region("GflashEnvelopes");
    G4Region* aRegion = new G4Region("GflashEnvelopes");
    
    ecal_ecap_log->SetRegion(aRegion);
    ecal_barrel_log->SetRegion(aRegion);
    hcal_barrel_log->SetRegion(aRegion);
    hcal_ecap_log->SetRegion(aRegion);
    
    aRegion->AddRootLogicalVolume(ecal_ecap_log);
    aRegion->AddRootLogicalVolume(ecal_barrel_log);
    aRegion->AddRootLogicalVolume(hcal_barrel_log);
    aRegion->AddRootLogicalVolume(hcal_ecap_log);

    //Electromagnetic Shower Model - Joanna Weng
    
    theParameterisation = new GFlashHomoShowerParameterisation(PbWO4);
    theShowerModel  = new GflashEMShowerModel("endcap",aRegion);
    theShowerModel->SetParameterisation(*theParameterisation);
    theShowerModel->SetParticleBounds(*theParticleBounds) ;
    theShowerModel->SetHitMaker(*theHMaker);
    
    double pEmin = m_pCaloModel.getParameter<double>("GFlashEmin");
    double pEmax = m_pCaloModel.getParameter<double>("GFlashEmax");
    double pToKill = m_pCaloModel.getParameter<double>("GFlashEToKill");
    theParticleBounds->SetMaxEneToParametrise(*G4Electron::ElectronDefinition(), pEmax);
    theParticleBounds->SetMinEneToParametrise(*G4Electron::ElectronDefinition(), pEmin);
    theParticleBounds->SetEneToKill(*G4Electron::ElectronDefinition(), pToKill);
    std::cout <<" CaloModel: GFlash:Emin "<<pEmin  << std::endl;
    std::cout <<" CaloModel: GFlash:Emax "<<pEmax  << std::endl;
    std::cout <<" CaloModel: GFlash:EToKill "<<pToKill << std::endl;
    
    //Hadronic Shower Model - added by Soon Yung Jun
    
    theHadronShowerModel = new GflashHadronShowerModel("GflashHadronShower",aRegion);
  }
  else {
    std::cout <<" !! GFlash: No Parameterisation Volumes found -> GFlash NOT ACTIVE !"<< std::endl;
  }
} 
