#include "Validation/Geometry/interface/MaterialBudgetData.h"

#include "G4Step.hh"
#include "G4Material.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

MaterialBudgetData::MaterialBudgetData() 
{
  //instantiate categorizer to assign an ID to volumes and materials
  myMaterialBudgetCategorizer = nullptr;
  allStepsToTree = false;
  isHGCal = false;
  densityConvertionFactor = 6.24E18;
}

MaterialBudgetData::~MaterialBudgetData() {

}

void MaterialBudgetData::SetAllStepsToTree()
{
  allStepsToTree = true;
}


void MaterialBudgetData::dataStartTrack( const G4Track* aTrack )
{

  const G4ThreeVector& dir = aTrack->GetMomentum() ;

  if( myMaterialBudgetCategorizer == nullptr){
    if(isHGCal){
      myMaterialBudgetCategorizer = std::make_unique<MaterialBudgetCategorizer>("HGCal"); 
    } else {
      myMaterialBudgetCategorizer = std::make_unique<MaterialBudgetCategorizer>("Tracker"); 
    }
  }
  
  theStepN=0;
  theTotalMB=0;
  theTotalIL=0;
  theEta=0;
  thePhi=0;
  theID=0;
  thePt=0;
  theEnergy=0;
  theMass=0;
 
  theSupportMB     = 0.;
  theSensitiveMB   = 0.;
  theCoolingMB     = 0.;
  theElectronicsMB = 0.;
  theOtherMB       = 0.;

  //HGCal
  theAirMB              = 0.;
  theCablesMB           = 0.;
  theCopperMB           = 0.;
  theH_ScintillatorMB   = 0.;
  theLeadMB             = 0.;
  theM_NEMA_FR4_plateMB = 0.;
  theSiliconMB          = 0.;
  theStainlessSteelMB   = 0.;
  theWCuMB              = 0.;

  theSupportIL     = 0.;
  theSensitiveIL   = 0.;
  theCoolingIL     = 0.;
  theElectronicsIL = 0.;
  theOtherIL       = 0.;

  //HGCal
  theAirIL              = 0.;
  theCablesIL           = 0.;
  theCopperIL           = 0.;
  theH_ScintillatorIL   = 0.;
  theLeadIL             = 0.;
  theM_NEMA_FR4_plateIL = 0.;
  theSiliconIL          = 0.;
  theStainlessSteelIL   = 0.;
  theWCuIL              = 0.;
  
  theSupportFractionMB     = 0.;
  theSensitiveFractionMB   = 0.;
  theCoolingFractionMB     = 0.;
  theElectronicsFractionMB = 0.;
  theOtherFractionMB       = 0.;
  //HGCal
  theAirFractionMB               = 0.;         
  theCablesFractionMB            = 0.;            
  theCopperFractionMB            = 0.;               
  theH_ScintillatorFractionMB    = 0.;          
  theLeadFractionMB              = 0.;              
  theM_NEMA_FR4_plateFractionMB  = 0.;         
  theSiliconFractionMB           = 0.;               
  theStainlessSteelFractionMB    = 0.;          
  theWCuFractionMB               = 0.;                   

  theSupportFractionIL     = 0.;
  theSensitiveFractionIL   = 0.;
  theCoolingFractionIL     = 0.;
  theElectronicsFractionIL = 0.;
  theOtherFractionIL       = 0.;
  //HGCal
  theAirFractionIL               = 0.;         
  theCablesFractionIL            = 0.;            
  theCopperFractionIL            = 0.;               
  theH_ScintillatorFractionIL    = 0.;          
  theLeadFractionIL              = 0.;              
  theM_NEMA_FR4_plateFractionIL  = 0.;         
  theSiliconFractionIL           = 0.;               
  theStainlessSteelFractionIL    = 0.;          
  theWCuFractionIL               = 0.;                   
  
  theID = (int)(aTrack->GetDefinition()->GetPDGEncoding());
  thePt = dir.perp();
  if( dir.theta() != 0 ) {
    theEta = dir.eta(); 
  } else {
    theEta = -99;
  }
  thePhi = dir.phi();
  theEnergy = aTrack->GetTotalEnergy();
  theMass = aTrack->GetDefinition()->GetPDGMass();  
}


void MaterialBudgetData::dataEndTrack( const G4Track* aTrack )
{
  LogDebug("MaterialBudget") << "MaterialBudgetData: [OVAL] MaterialBudget " 
			     << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() 
			     << " Eta:" << theEta << " Phi:" << thePhi << " TotalMB" << theTotalMB;
  
  LogDebug("MaterialBudget") << "MaterialBudgetData:" << theStepN << "Recorded steps ";

  if (!isHGCal){
    
    LogDebug("Material Budget") <<"MaterialBudgetData: Radiation Length " 
				<< G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() 
				<< " Eta" << theEta << " Phi" << thePhi 
				<< " TotalMB" << theTotalMB 
				<< " SUP " << theSupportMB << " SEN " << theSensitiveMB 
				<< " CAB " << theCablesMB << " COL " << theCoolingMB 
				<< " ELE " << theElectronicsMB << " other " << theOtherMB 
				<< " Air " << theAirMB;

    LogDebug("Material Budget") << "MaterialBudgetData: Interaction Length " 
				<< G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() 
				<< " Eta " << theEta << " Phi " << thePhi 
				<< " TotalIL " << theTotalIL
				<< " SUP " << theSupportIL << " SEN " << theSensitiveIL 
				<< " CAB " << theCablesIL << " COL " << theCoolingIL 
				<< " ELE " << theElectronicsIL << " other " << theOtherIL 
				<< " Air " << theAirIL << std::endl;

  } else {

    LogDebug("MaterialBudget") << "MaterialBudgetData: HGCal Material Budget: Radiation Length " 
			       << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() 
			       << " Eta " << theEta << " Phi " << thePhi 
			       << " TotaLMB" << theTotalMB 
			       << " theCopperMB " << theCopperMB << " theH_ScintillatorMB " << theH_ScintillatorMB 
			       << " CAB " << theCablesMB << " theLeadMB " << theLeadMB << " theM_NEMA_FR4_plateMB " 
			       << theM_NEMA_FR4_plateMB << " theSiliconMB " << theSiliconMB 
			       << " theAirMB " << theAirMB << " theStainlessSteelMB " << theStainlessSteelMB 
			       << " theWCuMB " << theWCuMB;
    
    LogDebug("MaterialBudget") << "MaterialBudgetData: HGCal Material Budget: Interaction Length " 
			       << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() 
			       << " Eta " << theEta << " Phi " << thePhi 
			       << " TotalIL " << theTotalIL << " theCopperIL " << theCopperIL 
			       << " theH_ScintillatorIL " << theH_ScintillatorIL 
			       << " CAB " << theCablesIL << " theLeadIL " << theLeadIL 
			       << " theM_NEMA_FR4_plateIL " << theM_NEMA_FR4_plateIL 
			       << " theSiliconIL " << theSiliconIL << " Air " 
			       << theAirIL << " theStainlessSteelIL " << theStainlessSteelIL 
			       << " theWCuIL " << theWCuIL << std::endl;
  }
}

void MaterialBudgetData::dataPerStep( const G4Step* aStep )
{
  assert(aStep);
  G4StepPoint* prePoint  = aStep->GetPreStepPoint();
  G4StepPoint* postPoint = aStep->GetPostStepPoint();
  assert(prePoint);
  assert(postPoint);
  G4Material * theMaterialPre = prePoint->GetMaterial();
  assert(theMaterialPre);
  
  CLHEP::Hep3Vector prePos  = prePoint->GetPosition();
  CLHEP::Hep3Vector postPos = postPoint->GetPosition();

  G4double steplen = aStep->GetStepLength();

  G4double radlen = theMaterialPre->GetRadlen();
  G4double intlen = theMaterialPre->GetNuclearInterLength();
  G4double density = theMaterialPre->GetDensity() / densityConvertionFactor; // always g/cm3
  
  G4String materialName = theMaterialPre->GetName();
  
  LogDebug("MaterialBudget") << "MaterialBudgetData: Material " << materialName
			     << " steplen " << steplen
			     << " radlen " << radlen 
			     << " mb " << steplen/radlen;

  G4String volumeName = aStep->GetPreStepPoint()->GetTouchable()->GetVolume(0)->GetLogicalVolume()->GetName();

  LogDebug("MaterialBudget") << "MaterialBudgetData: Volume " << volumeName
			     << " Material " << materialName;

  // instantiate the categorizer
  assert(myMaterialBudgetCategorizer);
  int volumeID   = myMaterialBudgetCategorizer->volume( volumeName );
  int materialID = myMaterialBudgetCategorizer->material( materialName );

  LogDebug("MaterialBudget") << "MaterialBudgetData: Volume ID " << volumeID 
			     << " Material ID " << materialID;

  // FIXME: Both volume ID and material ID are zeros, so this part is not executed leaving all
  // values as zeros. 

  if (!isHGCal){

    bool isCtgOk = !myMaterialBudgetCategorizer->x0fraction(materialName).empty()
      && !myMaterialBudgetCategorizer->l0fraction(materialName).empty()
      && (myMaterialBudgetCategorizer->x0fraction(materialName).size() == 7) /*7 Categories*/
      && (myMaterialBudgetCategorizer->l0fraction(materialName).size() == 7);
   
    if(!isCtgOk) 
      {
	if(materialName.compare("Air") == 0){
	  theAirFractionMB = 1;
	  theAirFractionIL = 1;
	} else {
	  theOtherFractionMB = 1;
	  theOtherFractionIL = 1;
	  edm::LogWarning("MaterialBudget")
	    << "MaterialBudgetData: Material forced to 'Other': " << materialName 
	    << " in volume " << volumeName << ". Check Categorization.";
	}
      }
    else 
      {
	theSupportFractionMB     = myMaterialBudgetCategorizer->x0fraction(materialName)[0];
	theSensitiveFractionMB   = myMaterialBudgetCategorizer->x0fraction(materialName)[1];
	theCablesFractionMB      = myMaterialBudgetCategorizer->x0fraction(materialName)[2];
	theCoolingFractionMB     = myMaterialBudgetCategorizer->x0fraction(materialName)[3];
	theElectronicsFractionMB = myMaterialBudgetCategorizer->x0fraction(materialName)[4];
	theOtherFractionMB       = myMaterialBudgetCategorizer->x0fraction(materialName)[5];
	theAirFractionMB         = myMaterialBudgetCategorizer->x0fraction(materialName)[6];
      
	if(theOtherFractionMB!=0) 
	  edm::LogWarning("MaterialBudget") << "MaterialBudgetData: Material found with no category: " << materialName 
					    << " in volume " << volumeName;

	theSupportFractionIL     = myMaterialBudgetCategorizer->l0fraction(materialName)[0];
	theSensitiveFractionIL   = myMaterialBudgetCategorizer->l0fraction(materialName)[1];
	theCablesFractionIL      = myMaterialBudgetCategorizer->l0fraction(materialName)[2];
	theCoolingFractionIL     = myMaterialBudgetCategorizer->l0fraction(materialName)[3];
	theElectronicsFractionIL = myMaterialBudgetCategorizer->l0fraction(materialName)[4];
	theOtherFractionIL       = myMaterialBudgetCategorizer->l0fraction(materialName)[5];
	theAirFractionIL         = myMaterialBudgetCategorizer->l0fraction(materialName)[6];

	if(theOtherFractionIL!=0) 
	  edm::LogWarning("MaterialBudget") << "MaterialBudgetData: Material found with no category: " << materialName 
					    << " in volume " << volumeName;
      }
  }  else { // isHGCal

    bool isHGCalx0fractionEmpty = myMaterialBudgetCategorizer->HGCalx0fraction(materialName).empty();
    bool isHGCall0fractionEmpty = myMaterialBudgetCategorizer->HGCall0fraction(materialName).empty();
    
    if( isHGCalx0fractionEmpty && isHGCall0fractionEmpty ) {
      theOtherFractionMB = 1;
      theOtherFractionIL = 1;

      edm::LogWarning("MaterialBudget") << "MaterialBudgetData: Material forced to 'Other': " << materialName 
					<< " in volume " << volumeName;
    } else{
    
      theAirFractionMB              = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[0];
      theCablesFractionMB           = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[1];
      theCopperFractionMB           = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[2];
      theH_ScintillatorFractionMB   = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[3];
      theLeadFractionMB             = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[4];
      theM_NEMA_FR4_plateFractionMB = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[5];
      theSiliconFractionMB          = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[6];
      theStainlessSteelFractionMB   = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[7];
      theWCuFractionMB              = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[8];
      theOtherFractionMB            = myMaterialBudgetCategorizer->HGCalx0fraction(materialName)[9];

    
      if(theOtherFractionMB!=0) 
	edm::LogWarning("MaterialBudget") << "MaterialBudgetData: Material found with no category: " << materialName 
					  << " in volume " << volumeName << std::endl;

      theAirFractionIL              = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[0];
      theCablesFractionIL           = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[1];
      theCopperFractionIL           = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[2];
      theH_ScintillatorFractionIL   = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[3];
      theLeadFractionIL             = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[4];
      theM_NEMA_FR4_plateFractionIL = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[5];
      theSiliconFractionIL          = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[6];
      theStainlessSteelFractionIL   = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[7];
      theWCuFractionIL              = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[8];
      theOtherFractionIL            = myMaterialBudgetCategorizer->HGCall0fraction(materialName)[9];


      if(theOtherFractionIL!=0) 
	edm::LogWarning("MaterialBudget") << "MaterialBudgetData: Material found with no category " << materialName 
					  << " in volume " << volumeName << std::endl;
    }
  }
  
  float dmb = steplen/radlen;
  float dil = steplen/intlen;
  
  G4VPhysicalVolume*       pv                = aStep->GetPreStepPoint()->GetPhysicalVolume();
  const G4VTouchable*      t                 = aStep->GetPreStepPoint()->GetTouchable();
  const G4ThreeVector&            objectTranslation = t->GetTranslation();
  const G4RotationMatrix*  objectRotation    = t->GetRotation();
  const G4VProcess*        interactionPre    = prePoint->GetProcessDefinedStep();
  const G4VProcess*        interactionPost   = postPoint->GetProcessDefinedStep();
  
  G4Track* track = aStep->GetTrack();
  if(theStepN==0) 
    LogDebug("MaterialBudget") << "MaterialBudgetData: Simulated Particle " << theID 
			       << "\tMass " << theMass << " MeV/c2"
			       << "\tPt = " << thePt  << " MeV/c" 
			       << "\tEta = " << theEta 
			       << "\tPhi = " << thePhi 
			       << "\tEnergy = " << theEnergy << " MeV";

  //fill data per step
  if( allStepsToTree ){
    assert(theStepN < MAXNUMBERSTEPS);
    if( theStepN > MAXNUMBERSTEPS ) theStepN = MAXNUMBERSTEPS - 1;
    theDmb[theStepN] = dmb;
    theDil[theStepN] = dil;
    theSupportDmb[theStepN]     = (dmb * theSupportFractionMB);
    theSensitiveDmb[theStepN]   = (dmb * theSensitiveFractionMB);
    theCoolingDmb[theStepN]     = (dmb * theCoolingFractionMB);
    theElectronicsDmb[theStepN] = (dmb * theElectronicsFractionMB);
    theOtherDmb[theStepN]       = (dmb * theOtherFractionMB);
    //HGCal
    theAirDmb[theStepN]                 = (dmb * theAirFractionMB);
    theCablesDmb[theStepN]              = (dmb * theCablesFractionMB);
    theCopperDmb[theStepN]              = (dmb * theCopperFractionMB);                       
    theH_ScintillatorDmb[theStepN]      = (dmb * theH_ScintillatorFractionMB);       
    theLeadDmb[theStepN]                = (dmb * theLeadFractionMB);                           
    theM_NEMA_FR4_plateDmb[theStepN]    = (dmb * theM_NEMA_FR4_plateFractionMB);   
    theSiliconDmb[theStepN]             = (dmb * theSiliconFractionMB);                     
    theStainlessSteelDmb[theStepN]      = (dmb * theStainlessSteelFractionMB);       
    theWCuDmb[theStepN]                 = (dmb * theWCuFractionMB);                             

    theSupportDil[theStepN]     = (dil * theSupportFractionIL);
    theSensitiveDil[theStepN]   = (dil * theSensitiveFractionIL);
    theCoolingDil[theStepN]     = (dil * theCoolingFractionIL);
    theElectronicsDil[theStepN] = (dil * theElectronicsFractionIL);
    theOtherDil[theStepN]       = (dil * theOtherFractionIL);
    //HGCal
    theAirDil[theStepN]                 = (dil * theAirFractionIL);
    theCablesDil[theStepN]              = (dil * theCablesFractionIL);
    theCopperDil[theStepN]              = (dil * theCopperFractionIL);                       
    theH_ScintillatorDil[theStepN]      = (dil * theH_ScintillatorFractionIL);       
    theLeadDil[theStepN]                = (dil * theLeadFractionIL);                           
    theM_NEMA_FR4_plateDil[theStepN]    = (dil * theM_NEMA_FR4_plateFractionIL);   
    theSiliconDil[theStepN]             = (dil * theSiliconFractionIL);                     
    theStainlessSteelDil[theStepN]      = (dil * theStainlessSteelFractionIL);       
    theWCuDil[theStepN]                 = (dil * theWCuFractionIL);                             

    theInitialX[theStepN] = prePos.x();
    theInitialY[theStepN] = prePos.y();
    theInitialZ[theStepN] = prePos.z();
    theFinalX[theStepN]   = postPos.x();
    theFinalY[theStepN]   = postPos.y();
    theFinalZ[theStepN]   = postPos.z();
    theVolumeID[theStepN]   = volumeID;
    theVolumeName[theStepN] = volumeName;
    theVolumeCopy[theStepN] = pv->GetCopyNo();
    theVolumeX[theStepN]    = objectTranslation.x();
    theVolumeY[theStepN]    = objectTranslation.y();
    theVolumeZ[theStepN]    = objectTranslation.z();
    theVolumeXaxis1[theStepN] = objectRotation->xx();
    theVolumeXaxis2[theStepN] = objectRotation->xy();
    theVolumeXaxis3[theStepN] = objectRotation->xz();
    theVolumeYaxis1[theStepN] = objectRotation->yx();
    theVolumeYaxis2[theStepN] = objectRotation->yy();
    theVolumeYaxis3[theStepN] = objectRotation->yz();
    theVolumeZaxis1[theStepN] = objectRotation->zx();
    theVolumeZaxis2[theStepN] = objectRotation->zy();
    theVolumeZaxis3[theStepN] = objectRotation->zz();
    theMaterialID[theStepN]      = materialID;
    theMaterialName[theStepN]    = materialName;
    theMaterialX0[theStepN]      = radlen;
    theMaterialLambda0[theStepN] = intlen;
    theMaterialDensity[theStepN] = density;
    theStepID[theStepN]             = track->GetDefinition()->GetPDGEncoding();
    theStepInitialPt[theStepN]      = prePoint->GetMomentum().perp();
    theStepInitialEta[theStepN]     = prePoint->GetMomentum().eta();
    theStepInitialPhi[theStepN]     = prePoint->GetMomentum().phi();
    theStepInitialEnergy[theStepN]  = prePoint->GetTotalEnergy();
    theStepInitialPx[theStepN]      = prePoint->GetMomentum().x();
    theStepInitialPy[theStepN]      = prePoint->GetMomentum().y();
    theStepInitialPz[theStepN]      = prePoint->GetMomentum().z();
    theStepInitialBeta[theStepN]    = prePoint->GetBeta();
    theStepInitialGamma[theStepN]   = prePoint->GetGamma();
    theStepInitialMass[theStepN]    = prePoint->GetMass();
    theStepFinalPt[theStepN]        = postPoint->GetMomentum().perp();
    theStepFinalEta[theStepN]       = postPoint->GetMomentum().eta();
    theStepFinalPhi[theStepN]       = postPoint->GetMomentum().phi();
    theStepFinalEnergy[theStepN]    = postPoint->GetTotalEnergy();
    theStepFinalPx[theStepN]        = postPoint->GetMomentum().x();
    theStepFinalPy[theStepN]        = postPoint->GetMomentum().y();
    theStepFinalPz[theStepN]        = postPoint->GetMomentum().z();
    theStepFinalBeta[theStepN]      = postPoint->GetBeta();
    theStepFinalGamma[theStepN]     = postPoint->GetGamma();
    theStepFinalMass[theStepN]      = postPoint->GetMass();
    int preProcType  = -99;
    int postProcType = -99;
    if (interactionPre) preProcType = interactionPre->GetProcessType();
    theStepPreProcess[theStepN]     = preProcType;
    if (interactionPost) postProcType = interactionPost->GetProcessType();
    theStepPostProcess[theStepN]    = postProcType;
    
    LogDebug("MaterialBudget") 
      << "MaterialBudgetData: Step " << theStepN
      << "\tDelta MB = " << theDmb[theStepN]
      << std::endl
      << " Support "  << theSupportDmb[theStepN]
      << " Sensitive "   << theSensitiveDmb[theStepN]
      << " Cables "      << theCablesDmb[theStepN]
      << " Cooling "     << theCoolingDmb[theStepN]
      << " Electronics " << theElectronicsDmb[theStepN]
      << " Other "       << theOtherDmb[theStepN]
      << " Air "         << theAirDmb[theStepN]
      << std::endl
      << "\tDelta IL = " << theDil[theStepN]
      << std::endl
      << " Support "  << theSupportDil[theStepN]
      << " Sensitive "   << theSensitiveDil[theStepN]
      << " Cables "      << theCablesDil[theStepN]
      << " Cooling "     << theCoolingDil[theStepN]
      << " Electronics " << theElectronicsDil[theStepN]
      << " Other "       << theOtherDil[theStepN]
      << " Air "         << theAirDil[theStepN];
    
    if (interactionPre)
      LogDebug("MaterialBudget") 
	<< "MaterialBudgetData: Process Pre " << interactionPre->GetProcessName()
	<< " type " << theStepPreProcess[theStepN] 
	<< " name " << interactionPre->GetProcessTypeName(G4ProcessType(theStepPreProcess[theStepN]));
    if (interactionPost)
      LogDebug("MaterialBudget")
	<< "MaterialBudgetData: Process Post " << interactionPost->GetProcessName()
	<< " type " << theStepPostProcess[theStepN] 
	<< " name "<< interactionPost->GetProcessTypeName(G4ProcessType(theStepPostProcess[theStepN]))
	<< " Pre x = " << theInitialX[theStepN]
	<< " y = "     << theInitialY[theStepN]
	<< " z = "     << theInitialZ[theStepN] 
	<< " Polar Radius = " << sqrt(prePos.x()*prePos.x()+prePos.y()*prePos.y())
	<< " Pt = "     << theStepInitialPt[theStepN]
	<< " Energy = " << theStepInitialEnergy[theStepN]
	<< " Final: "
	<< " Post x = " << theFinalX[theStepN]
	<< " y = "      << theFinalY[theStepN]
	<< " z = "      << theFinalZ[theStepN] 
	<< " Polar Radius = " << sqrt(postPos.x()*postPos.x()+postPos.y()*postPos.y())
	<< " Pt = "     << theStepFinalPt[theStepN]
	<< " Energy = " << theStepFinalEnergy[theStepN]
	<< std::endl
	<< " Volume " << volumeID << " name " << theVolumeName[theStepN] 
	<< " copy number " << theVolumeCopy[theStepN]
	<< " material " << theMaterialID[theStepN] << " " << theMaterialName[theStepN]
	<< " Density = " << theMaterialDensity[theStepN] << " g/cm3"
	<< " X0 = " << theMaterialX0[theStepN] << " mm"
	<< " Lambda0 = " << theMaterialLambda0[theStepN] << " mm"
	<< std::endl
	<< " Particle "  << theStepID[theStepN] 
	<< " Initial Pt = " << theStepInitialPt[theStepN]     << " MeV/c"
	<< " eta = "        << theStepInitialEta[theStepN]
	<< " phi = "        << theStepInitialPhi[theStepN]
	<< " Energy = "     << theStepInitialEnergy[theStepN] << " MeV"
	<< " Mass = "       << theStepInitialMass[theStepN]   << " MeV/c2"
	<< " Beta = "       << theStepInitialBeta[theStepN]
	<< " Gamma = "      << theStepInitialGamma[theStepN]
	<< std::endl
	<< " Particle "  << theStepID[theStepN]
	<< " Final Pt = "   << theStepFinalPt[theStepN]       << " MeV/c"
	<< " eta = "        << theStepFinalEta[theStepN]
	<< " phi = "        << theStepFinalPhi[theStepN]
	<< " Energy = "     << theStepFinalEnergy[theStepN]   << " MeV"
	<< " Mass = "       << theStepFinalMass[theStepN]     << " MeV/c2"
	<< " Beta = "       << theStepFinalBeta[theStepN]
	<< " Gamma = "      << theStepFinalGamma[theStepN]
	<< std::endl
	<< " Volume Centre x = " << theVolumeX[theStepN]
	<< " y = "               << theVolumeY[theStepN]
	<< " z = "               << theVolumeZ[theStepN]
	<< "Polar Radius = "    << sqrt( theVolumeX[theStepN]*theVolumeX[theStepN] +
					 theVolumeY[theStepN]*theVolumeY[theStepN] )
	<< std::endl
	<< " x axis = (" 
	<< theVolumeXaxis1[theStepN] << "," 
	<< theVolumeXaxis2[theStepN] << "," 
	<< theVolumeXaxis3[theStepN] << ")"
	<< std::endl
	<< " y axis = (" 
	<< theVolumeYaxis1[theStepN] << "," 
	<< theVolumeYaxis2[theStepN] << "," 
	<< theVolumeYaxis3[theStepN] << ")"
	<< std::endl
	<< " z axis = (" 
	<< theVolumeZaxis1[theStepN] << "," 
	<< theVolumeZaxis2[theStepN] << "," 
	<< theVolumeZaxis3[theStepN] << ")"
	<< std::endl
	<< " Secondaries"
	<< std::endl;
    
    for(G4TrackVector::const_iterator iSec = aStep->GetSecondary()->begin(); iSec!=aStep->GetSecondary()->end(); iSec++) {
      LogDebug("MaterialBudget") 
	<< "MaterialBudgetData: tid " << (*iSec)->GetDefinition()->GetPDGEncoding()
	<< " created through process type " << (*iSec)->GetCreatorProcess()->GetProcessType()
	<< (*iSec)->GetCreatorProcess()->GetProcessTypeName(G4ProcessType((*iSec)->GetCreatorProcess()->GetProcessType()));
    }
  }
  
  theTrkLen = aStep->GetTrack()->GetTrackLength();
  thePVname = pv->GetName();
  thePVcopyNo = pv->GetCopyNo();
  theRadLen = radlen;
  theIntLen = intlen;
  theTotalMB += dmb;
  theTotalIL += dil;
  
  theSupportMB     += (dmb * theSupportFractionMB);
  theSensitiveMB   += (dmb * theSensitiveFractionMB);
  theCoolingMB     += (dmb * theCoolingFractionMB);
  theElectronicsMB += (dmb * theElectronicsFractionMB);
  theOtherMB       += (dmb * theOtherFractionMB);

  //HGCal
  theAirMB                 += (dmb * theAirFractionMB);
  theCablesMB              += (dmb * theCablesFractionMB);
  theCopperMB              += (dmb * theCopperFractionMB);                       
  theH_ScintillatorMB      += (dmb * theH_ScintillatorFractionMB);       
  theLeadMB                += (dmb * theLeadFractionMB);                           
  theM_NEMA_FR4_plateMB    += (dmb * theM_NEMA_FR4_plateFractionMB);   
  theSiliconMB             += (dmb * theSiliconFractionMB);                     
  theStainlessSteelMB      += (dmb * theStainlessSteelFractionMB);       
  theWCuMB                 += (dmb * theWCuFractionMB);                             

  theSupportIL     += (dil * theSupportFractionIL);
  theSensitiveIL   += (dil * theSensitiveFractionIL);
  theCoolingIL     += (dil * theCoolingFractionIL);
  theElectronicsIL += (dil * theElectronicsFractionIL);
  theOtherIL       += (dil * theOtherFractionIL);
  //HGCal
  theAirIL                 += (dil * theAirFractionIL);
  theCablesIL              += (dil * theCablesFractionIL);
  theCopperIL              += (dil * theCopperFractionIL);                       
  theH_ScintillatorIL      += (dil * theH_ScintillatorFractionIL);       
  theLeadIL                += (dil * theLeadFractionIL);                           
  theM_NEMA_FR4_plateIL    += (dil * theM_NEMA_FR4_plateFractionIL);   
  theSiliconIL             += (dil * theSiliconFractionIL);                     
  theStainlessSteelIL      += (dil * theStainlessSteelFractionIL);       
  theWCuIL                 += (dil * theWCuFractionIL);                             
  


  // rr
  
  theStepN++;
  
}


