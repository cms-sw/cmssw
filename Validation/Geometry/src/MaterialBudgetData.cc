//#include "Utilities/UI/interface/SimpleConfigurable.h"

#include "Validation/Geometry/interface/MaterialBudgetData.h"
#include "G4Step.hh"
#include "G4Material.hh"
#include "G4EventManager.hh"
#include "G4Event.hh"

MaterialBudgetData::MaterialBudgetData()
{

  //instantiate categorizer to assing an ID to volumes and materials
  myMaterialBudgetCategorizer = 0;
}

void MaterialBudgetData::SetAllStepsToTree()
{
  allStepsToTree = 1;
  MAXNUMBERSTEPS = 0;
  MAXNUMBERSTEPS = 5000; //!!!WARNING: this number is also hardcoded when booking the tree
  theDmb = new float[MAXNUMBERSTEPS];
  theX = new float[MAXNUMBERSTEPS];
  theY = new float[MAXNUMBERSTEPS];
  theZ = new float[MAXNUMBERSTEPS];
  //  theVoluId = new int[MAXNUMBERSTEPS];
  //  theMateId = new int[MAXNUMBERSTEPS];
  // rr
  theVolumeID     = new int[MAXNUMBERSTEPS];
  theVolumeName   = new std::string[MAXNUMBERSTEPS];
  theVolumeCopy   = new int[MAXNUMBERSTEPS];
  theVolumeX      = new float[MAXNUMBERSTEPS];
  theVolumeY      = new float[MAXNUMBERSTEPS];
  theVolumeZ      = new float[MAXNUMBERSTEPS];
  theVolumeXaxis1 = new float[MAXNUMBERSTEPS];
  theVolumeXaxis2 = new float[MAXNUMBERSTEPS];
  theVolumeXaxis3 = new float[MAXNUMBERSTEPS];
  theVolumeYaxis1 = new float[MAXNUMBERSTEPS];
  theVolumeYaxis2 = new float[MAXNUMBERSTEPS];
  theVolumeYaxis3 = new float[MAXNUMBERSTEPS];
  theVolumeZaxis1 = new float[MAXNUMBERSTEPS];
  theVolumeZaxis2 = new float[MAXNUMBERSTEPS];
  theVolumeZaxis3 = new float[MAXNUMBERSTEPS];
  theMaterialID   = new int[MAXNUMBERSTEPS];
  theMaterialName = new std::string[MAXNUMBERSTEPS];
  theMaterialX0   = new float[MAXNUMBERSTEPS];
  theStepID     = new int[MAXNUMBERSTEPS];
  theStepPt     = new float[MAXNUMBERSTEPS];
  theStepEta    = new float[MAXNUMBERSTEPS];
  theStepPhi    = new float[MAXNUMBERSTEPS];
  theStepEnergy = new float[MAXNUMBERSTEPS];
  // rr
}


void MaterialBudgetData::dataStartTrack( const G4Track* aTrack )
{
  const G4ThreeVector& dir = aTrack->GetMomentum() ;
  
  if( myMaterialBudgetCategorizer == 0) myMaterialBudgetCategorizer = new MaterialBudgetCategorizer;
  
  theStepN=0;
  theTotalMB=0;
  theEta=0;
  thePhi=0;
  
  // rr
  theID=0;
  thePt=0;
  theEnergy=0;
  
  theSupportMB     = 0.;
  theSensitiveMB   = 0.;
  theCablesMB      = 0.;
  theCoolingMB     = 0.;
  theElectronicsMB = 0.;
  theOtherMB       = 0.;
  theAirMB         = 0.;
  theSupportFractionMB     = 0.;
  theSensitiveFractionMB   = 0.;
  theCablesFractionMB      = 0.;
  theCoolingFractionMB     = 0.;
  theElectronicsFractionMB = 0.;
  theOtherFractionMB       = 0.;
  theAirFractionMB         = 0.;
  // rr
  
  theID = (int)(aTrack->GetDefinition()->GetPDGEncoding());
  thePt = dir.perp();
  if( dir.theta() != 0 ) {
    theEta = dir.eta(); 
  } else {
    theEta = -99;
  }
  //  thePhi = dir.phi()/deg; // better not to store in deg
  thePhi = dir.phi();
  theEnergy = aTrack->GetTotalEnergy();
  
}


void MaterialBudgetData::dataEndTrack( const G4Track* aTrack )
{
  //-  std::cout << "[OVAL] MaterialBudget " << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() << " " << theEta << " " << thePhi << " " << theTotalMB << std::endl;
  // rr
  std::cout << "[OVAL] MaterialBudget " << G4EventManager::GetEventManager()->GetConstCurrentEvent()->GetEventID() << " eta " << theEta << " phi " << thePhi << " total MB " << theTotalMB << " SUP " << theSupportMB << " SEN " << theSensitiveMB << " CAB " << theCablesMB << " COL " << theCoolingMB << " ELE " << theElectronicsMB << " other " << theOtherMB << " Air " << theAirMB << std::endl;
  // rr
}


void MaterialBudgetData::dataPerStep( const G4Step* aStep )
{
  G4Material * theMaterialPre = aStep->GetPreStepPoint()->GetMaterial();
  //  G4Material * theMaterialPost = aStep->GetPostStepPoint()->GetMaterial();

  Hep3Vector prePos =  aStep->GetPreStepPoint()->GetPosition();
  Hep3Vector postPos =  aStep->GetPostStepPoint()->GetPosition();

  G4double steplen = aStep->GetStepLength();

  G4double radlen;

  radlen = theMaterialPre->GetRadlen();
  //t    radlen = theMaterialPre->GetNuclearInterLength();

  G4String name = theMaterialPre->GetName();
  //  std::cout << " steplen " << steplen << " radlen " << radlen << " mb " << steplen/radlen << " mate " << theMaterialPre->GetName() << std::endl;
     
  G4LogicalVolume* lv = aStep->GetTrack()->GetVolume()->GetLogicalVolume();

  // instantiate the categorizer
  int volumeID   = myMaterialBudgetCategorizer->volume( lv->GetName() );
  int materialID = myMaterialBudgetCategorizer->material( lv->GetMaterial()->GetName() );
  // rr
  std::string volumeName   = lv->GetName();
  std::string materialName = lv->GetMaterial()->GetName();
  // rr
  
  // rr
  /*
    std::cout << " Volume "   << lv->GetName()                << std::endl;
    std::cout << " Material " << lv->GetMaterial()->GetName() << std::endl;
  */    
  theSupportFractionMB     = myMaterialBudgetCategorizer->x0fraction(lv->GetMaterial()->GetName())[0];
  theSensitiveFractionMB   = myMaterialBudgetCategorizer->x0fraction(lv->GetMaterial()->GetName())[1];
  theCablesFractionMB      = myMaterialBudgetCategorizer->x0fraction(lv->GetMaterial()->GetName())[2];
  theCoolingFractionMB     = myMaterialBudgetCategorizer->x0fraction(lv->GetMaterial()->GetName())[3];
  theElectronicsFractionMB = myMaterialBudgetCategorizer->x0fraction(lv->GetMaterial()->GetName())[4];
  theOtherFractionMB       = myMaterialBudgetCategorizer->x0fraction(lv->GetMaterial()->GetName())[5];
  theAirFractionMB         = myMaterialBudgetCategorizer->x0fraction(lv->GetMaterial()->GetName())[6];
  if(theOtherFractionMB!=0) std::cout << " material found with no category " << lv->GetMaterial()->GetName() 
				 << " in volume " << lv->GetName() << std::endl;
  // rr  
  
  float dmb = steplen/radlen;
  Hep3Vector pos = aStep->GetPreStepPoint()->GetPosition();
  Hep3Vector post = aStep->GetPostStepPoint()->GetPosition();
  
  G4VPhysicalVolume* pv = aStep->GetPreStepPoint()->GetPhysicalVolume();
  const G4VTouchable* t = aStep->GetPreStepPoint()->GetTouchable();
  G4ThreeVector     objectTranslation = t->GetTranslation();
  G4RotationMatrix* objectRotation    = t->GetRotation();
  
  G4Track* track = aStep->GetTrack();
  
  if(theStepN==0) std::cout << " Simulated Particle " << theID
			    << "\tPt = " << thePt  << " MeV/c" << "\tEta = " << theEta << "\tPhi = " << thePhi 
			    << "\tEnergy = " << theEnergy << " MeV"
			    << std::endl;
  
  //fill data per step
  if( allStepsToTree ){
    theDmb[theStepN] = dmb; 
    if( stepN > MAXNUMBERSTEPS ) stepN = MAXNUMBERSTEPS;
    theX[theStepN] = pos.x();
    theY[theStepN] = pos.y();
    theZ[theStepN] = pos.z();
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
    theMaterialID[theStepN]   = materialID;
    theMaterialName[theStepN] = materialName;
    theMaterialX0[theStepN]   = radlen;
    theStepID[theStepN]     = track->GetDefinition()->GetPDGEncoding();
    theStepPt[theStepN]     = track->GetMomentum().perp();
    theStepEta[theStepN]    = track->GetMomentum().eta();
    theStepPhi[theStepN]    = track->GetMomentum().phi();
    theStepEnergy[theStepN] = track->GetTotalEnergy();
    std::cout << " step " << theStepN
	      << "\tDelta MB = " << dmb << std::endl;
    std::cout << "\tPre x = " << pos.x()
	      << "\ty = "     << pos.y()
	      << "\tz = "     << pos.z() 
	      << "\tPolar Radius = " << sqrt(pos.x()*pos.x()+pos.y()*pos.y())
	      << std::endl;
    std::cout  << "\tPost x = " << post.x()
	       << "\ty = "      << post.y()
	       << "\tz = "      << post.z() 
	       << "\tPolar Radius = " << sqrt(post.x()*post.x()+post.y()*post.y())
	       << std::endl;
    std::cout << "\tvolume " << volumeID << " " << theVolumeName[theStepN] 
	      << " copy number " << theVolumeCopy[theStepN]
	      << "\tmaterial " << theMaterialID[theStepN] << " " << theMaterialName[theStepN]
	      << "\tX0 = " << theMaterialX0[theStepN] << " cm"
	      << std::endl;
    std::cout << "\t\tParticle " << theStepID[theStepN] 
	      << " Pt = "        << theStepPt[theStepN]     << " MeV/c"
	      << " eta = "       << theStepEta[theStepN]
	      << " phi = "       << theStepPhi[theStepN]
	      << " Energy = "    << theStepEnergy[theStepN] << " MeV"
	      << std::endl;
    std:: cout << "\tVolume Centre x = " << theVolumeX[theStepN]
	       << "\ty = "               << theVolumeY[theStepN]
	       << "\tz = "               << theVolumeZ[theStepN]
	       << "\tPolar Radius = "    << sqrt( theVolumeX[theStepN]*theVolumeX[theStepN] +
						  theVolumeY[theStepN]*theVolumeY[theStepN] )
	       << std::endl;
    std::cout << "\tx axis = (" 
	      << theVolumeXaxis1[theStepN] << "," 
	      << theVolumeXaxis2[theStepN] << "," 
	      << theVolumeXaxis3[theStepN] << ")"
	      << std::endl;
    std::cout << "\ty axis = (" 
	      << theVolumeYaxis1[theStepN] << "," 
	      << theVolumeYaxis2[theStepN] << "," 
	      << theVolumeYaxis3[theStepN] << ")"
	      << std::endl;
    std::cout << "\tz axis = (" 
	      << theVolumeZaxis1[theStepN] << "," 
	      << theVolumeZaxis2[theStepN] << "," 
	      << theVolumeZaxis3[theStepN] << ")"
	      << std::endl;
  }
  
  theTrkLen = aStep->GetTrack()->GetTrackLength();
  //-  std::cout << " theTrkLen " << theTrkLen << " theTrkLen2 " << theTrkLen2 << " postPos " << postPos.mag() << postPos << std::endl;
  thePVname = pv->GetName();
  thePVcopyNo = pv->GetCopyNo();
  theRadLen = radlen;
  theTotalMB += dmb;
  
  // rr
  theSupportMB     += (dmb * theSupportFractionMB);
  theSensitiveMB   += (dmb * theSensitiveFractionMB);
  theCablesMB      += (dmb * theCablesFractionMB);
  theCoolingMB     += (dmb * theCoolingFractionMB);
  theElectronicsMB += (dmb * theElectronicsFractionMB);
  theOtherMB       += (dmb * theOtherFractionMB);
  theAirMB         += (dmb * theAirFractionMB);
  // rr
  
  theStepN++;
}

