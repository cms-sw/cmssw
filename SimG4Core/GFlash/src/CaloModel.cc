#include "SimG4Core/GFlash/interface/CaloModel.h"
#include "SimG4Core/Geometry/interface/G4LogicalVolumeToDDLogicalPartMapper.h"

#include "DetectorDescription/Core/interface/DDName.h"
#include "DetectorDescription/Core/interface/DDMaterial.h"
#include "DetectorDescription/Core/interface/DDCurrentNamespace.h"
#include "G4Material.hh"
#include "G4Element.hh"
#include "G4Electron.hh"
#include "G4FastSimulationManager.hh"
#include "G4LogicalVolumeStore.hh"
#include "GFlashHomoShowerParamterisation.hh"

#include "SimG4Core/G4gflash/src/GFlashHitMaker.hh"
#include "SimG4Core/G4gflash/src/GFlashShowerModel.hh"

#include "FWCore/ServiceRegistry/interface/Service.h"
 
//#include "GFlashHitMaker.hh"
#include "GFlashParticleBounds.hh"

CaloModel::CaloModel(const edm::ParameterSet & p) : 
  m_pCaloModel(p.getParameter<edm::ParameterSet>("GFlashCaloModel")) 
{ 
  build();
  std::cout <<" CaloModel built !!!  "<< std::endl;
}

void CaloModel::build()
{
	G4LogicalVolume * barrel_log = 0;
	G4LogicalVolume * ecap_log = 0;
	//Finding correct G4LogicalVolume for parameterisation
	ConcreteG4LogicalVolumeToDDLogicalPartMapper::Vector vec =
        G4LogicalVolumeToDDLogicalPartMapper::instance()->all("volumes");
       	std::cout <<" CaloModel !!!  "<< std::endl;

	for (ConcreteG4LogicalVolumeToDDLogicalPartMapper::Vector::iterator
        tit = vec.begin(); tit != vec.end(); tit++){
		if (((*tit).first)->GetName()=="ESPM"){  
			std::cout <<" GFlash added  to voulme  "<< ((*tit).first)->GetName() << std::endl;
			barrel_log = (*tit).first;
		}
		if (((*tit).first)->GetName()=="ENCA"){
			std::cout <<" GFlash added  to voulme  "<< ((*tit).first)->GetName() << std::endl;
			ecap_log = (*tit).first;
		}
	} 

	// Parameterisaiton components	
	theParticleBounds  = new GFlashParticleBounds();
	theHMaker          = new GFlashHitMaker();  
	// ************Test declaration ****************************************
	G4String name, symbol;                        //a=mass of a mole;
	G4double a, z, density;                       //z=mean number of protons;	
	G4int ncomponents;
	G4double  fractionmass;
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
	// ***************************************************************** 
	theParametrisation = new GFlashHomoShowerParamterisation(PbWO4);                         	
	if  (ecap_log) { 
		theShowerModel  = new GFlashShowerModel("endcap",ecap_log);
		theShowerModel->SetParametrisation(*theParametrisation);
		theShowerModel->SetParticleBounds(*theParticleBounds) ;  
		theShowerModel->SetHitMaker(*theHMaker);
	}
	if  (barrel_log) {
		new G4FastSimulationManager(barrel_log);
		barrel_log->GetFastSimulationManager()->AddFastSimulationModel(theShowerModel);
	}
	
	double pEmin = m_pCaloModel.getParameter<double>("Emin");
	double pEmax = m_pCaloModel.getParameter<double>("Emax");
	double pToKill = m_pCaloModel.getParameter<double>("EToKill");
	theParticleBounds->SetMaxEneToParametrise(*G4Electron::ElectronDefinition(), pEmax);
	theParticleBounds->SetMinEneToParametrise(*G4Electron::ElectronDefinition(), pEmin); 
	theParticleBounds->SetEneToKill(*G4Electron::ElectronDefinition(), pToKill);	
	std::cout <<" CaloModel: GFlash:Emin "<<pEmin  << std::endl;
	std::cout <<" CaloModel: GFlash:Emax "<<pEmax  << std::endl;
	std::cout <<" CaloModel: GFlash:EToKill "<<pToKill << std::endl;		
	// barrel_log->GetFastSimulationManager()->ListModels();
	//  ecap_log->GetFastSimulationManager()->ListModels();	
	
} 


CaloModel::~CaloModel()
{}
