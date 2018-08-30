///////////////////////////////////////////////////////////////////////////////
// File: Totem_RP_SD.cc
// Date: 18.10.2005
// Description: Sensitive Detector class for TOTEM RP Detectors
// Modifications: 
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/PPS/interface/Totem_RP_SD.h"
#include "SimG4CMS/PPS/interface/PPSStripNumberingScheme.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SimG4Core/Notification/interface/TrackInformation.h"
#include "SimG4Core/Notification/interface/G4TrackToParticleID.h"
#include "SimG4Core/Geometry/interface/SensitiveDetectorCatalog.h"
#include "SimG4Core/Physics/interface/G4ProcessTypeEnumerator.h"

#include "SimG4Core/SensitiveDetector/interface/SensitiveTkDetector.h"

#include "SimDataFormats/TrackingHit/interface/UpdatablePSimHit.h"
#include "SimDataFormats/SimHitMaker/interface/TrackingSlaveSD.h"

#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"

#include "G4SDManager.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4VProcess.hh"

#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"

#include <iostream>
#include <vector>
#include <string>


Totem_RP_SD::Totem_RP_SD(const std::string & name, const DDCompactView & cpv, const SensitiveDetectorCatalog & clg,edm::ParameterSet const & p, const SimTrackManager * manager) : 
  SensitiveTkDetector(name, cpv, clg, p), 
  numberingScheme(nullptr),  
  hcID(-1), theHC(nullptr), currentHit(nullptr), theTrack(nullptr), currentPV(nullptr), 
  unitID(0),  preStepPoint(nullptr), postStepPoint(nullptr), eventno(0)
{
  collectionName.insert(name);
  
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("Totem_RP_SD");
  verbosity_ = m_Anal.getParameter<int>("Verbosity");
/*  
  std::cout 
    << "*******************************************************\n"
    << "*                                                     *\n"
    << "* Constructing a Totem_RP_SD  with name " << name << "\n"
    << "*                                                     *\n"
    << "*******************************************************" << std::endl;
*/
  

  slave  = new TrackingSlaveSD(name);
  
  if (name == "TotemHitsRP")
  {
    numberingScheme = dynamic_cast<TotemRPVDetectorOrganization*>(new PPSStripNumberingScheme(3));
  } else {
    edm::LogWarning("TotemRP") << "Totem_RP_SD: ReadoutName not supported\n";
  }
  
  edm::LogInfo("TotemRP") << "Totem_RP_SD: Instantiation completed";
}


Totem_RP_SD::~Totem_RP_SD()
{ 
    delete slave; 
    delete numberingScheme;
}

void Totem_RP_SD::Initialize(G4HCofThisEvent * HCE) {
  LogDebug("TotemRP") << "Totem_RP_SD : Initialize called for " << name;

  theHC = new Totem_RP_G4HitCollection(GetName(), collectionName[0]);
  G4SDManager::GetSDMpointer()->AddNewCollection(name, collectionName[0]);

  if (hcID<0) 
    hcID = G4SDManager::GetSDMpointer()->GetCollectionID(collectionName[0]);
  HCE->AddHitsCollection(hcID, theHC);
}

void Totem_RP_SD::Print_Hit_Info()
{
  LogDebug("TotemRP") << theTrack->GetDefinition()->GetParticleName()
       << " Totem_RP_SD CreateNewHit for"
       << " PV "     << currentPV->GetName()
       << " PVid = " << currentPV->GetCopyNo()
       //<< " MVid = " << currentPV->GetMother()->GetCopyNo()
       << " Unit "   << unitID;
  LogDebug("TotemRP") << " primary "    << primaryID
       << " time slice " << tSliceID 
       << " of energy " << theTrack->GetTotalEnergy()
       << " Eloss " << Eloss
       << " positions ";
       printf("(%10f,%10f,%10f)",preStepPoint->GetPosition().x(),preStepPoint->GetPosition().y(),preStepPoint->GetPosition().z());
       printf("(%10f,%10f,%10f)",postStepPoint->GetPosition().x(),postStepPoint->GetPosition().y(),postStepPoint->GetPosition().z());
  LogDebug("TotemRP") << " positions " << "(" <<postStepPoint->GetPosition().x()<<","<<postStepPoint->GetPosition().y()<<","<<postStepPoint->GetPosition().z()<<")"
       << " For Track  " << theTrack->GetTrackID()
       << " which is a " << theTrack->GetDefinition()->GetParticleName();
     
  if(theTrack->GetTrackID()==1)
  {
    LogDebug("TotemRP") << " primary particle ";
  }
  else
  {
    LogDebug("TotemRP") << " daughter of part. " << theTrack->GetParentID();
  }

  LogDebug("TotemRP")  << " and created by " ;
  
  if(theTrack->GetCreatorProcess()!=nullptr)
    LogDebug("TotemRP") << theTrack->GetCreatorProcess()->GetProcessName() ;
  else 
    LogDebug("TotemRP") << "NO process";
    
  LogDebug("TotemRP") << std::endl;
}


bool Totem_RP_SD::ProcessHits(G4Step * aStep, G4TouchableHistory * )
{
  if (aStep == nullptr)
  {	    
    return true;
  } else {
    GetStepInfo(aStep);
    //Print_Hit_Info();
 
    CreateNewHit();
    return true;
  }
}


void Totem_RP_SD::GetStepInfo(G4Step* aStep)
{
  preStepPoint = aStep->GetPreStepPoint(); 
  postStepPoint = aStep->GetPostStepPoint(); 
  theTrack = aStep->GetTrack();   
  hitPoint = preStepPoint->GetPosition();
  exitPoint = postStepPoint->GetPosition();
  currentPV = preStepPoint->GetPhysicalVolume();
  theLocalEntryPoint = SetToLocal(hitPoint);
  theLocalExitPoint = SetToLocal(exitPoint);


  G4String name = currentPV->GetName();
  name.assign(name,0,4);
  G4String particleType = theTrack->GetDefinition()->GetParticleName();
  tSlice = (postStepPoint->GetGlobalTime() )/nanosecond;
  tSliceID = (int) tSlice;
  unitID = setDetUnitId(aStep);

  if(verbosity_)
    LogDebug("TotemRP") << "UNITa " << unitID <<std::endl;

  primaryID = theTrack->GetTrackID();

  Pabs = (aStep->GetPreStepPoint()->GetMomentum().mag())/GeV;
  p_x = (aStep->GetPreStepPoint()->GetMomentum().x())/GeV;
  p_y = (aStep->GetPreStepPoint()->GetMomentum().y())/GeV;
  p_z = (aStep->GetPreStepPoint()->GetMomentum().z())/GeV;
  
  Tof = aStep->GetPostStepPoint()->GetGlobalTime()/nanosecond;
  Eloss = aStep->GetTotalEnergyDeposit()/GeV;
  ParticleType = theTrack->GetDefinition()->GetPDGEncoding();

  //corrected phi and theta treatment
  G4ThreeVector gmd  = aStep->GetPreStepPoint()->GetMomentumDirection();
  // convert it to local frame
  G4ThreeVector lmd = ((G4TouchableHistory *)(aStep->GetPreStepPoint()->GetTouchable()))->GetHistory()
    ->GetTopTransform().TransformAxis(gmd);
  Local3DPoint lnmd = ConvertToLocal3DPoint(lmd);
  ThetaAtEntry = lnmd.theta();
  PhiAtEntry = lnmd.phi();

//    numberingScheme->GetUnitID(aStep) << std::endl ;
  if(IsPrimary(theTrack))
    ParentId = 0;
  else ParentId = theTrack->GetParentID();
  
  Vx = theTrack->GetVertexPosition().x()/mm;
  Vy = theTrack->GetVertexPosition().y()/mm;
  Vz = theTrack->GetVertexPosition().z()/mm;
}


uint32_t Totem_RP_SD::setDetUnitId(const G4Step * aStep)
{ 
  return (numberingScheme == nullptr ? 0 : numberingScheme->GetUnitID(aStep));
}


void Totem_RP_SD::StoreHit(Totem_RP_G4Hit* hit)
{
  if (hit == nullptr )
  {
    if(verbosity_)
      LogDebug("TotemRP") << "Totem_RP_SD: hit to be stored is NULL !!" <<std::endl;
    return;
  }
  theHC->insert( hit );
}


void Totem_RP_SD::CreateNewHit()
{
// Protect against creating hits in detectors not inserted
  double outrangeX = hitPoint.x();
  double outrangeY = hitPoint.y();
  if (fabs(outrangeX)>40.) return;
  if (fabs(outrangeY)>40.) return;
// end protection

  currentHit = new Totem_RP_G4Hit;
  currentHit->setTrackID(primaryID);
  currentHit->setTimeSlice(tSlice);
  currentHit->setUnitID(unitID);
  currentHit->setIncidentEnergy(incidentEnergy);

  currentHit->setPabs(Pabs);
  currentHit->setTof(Tof);
  currentHit->setEnergyLoss(Eloss);
  currentHit->setParticleType(ParticleType);
  currentHit->setThetaAtEntry(ThetaAtEntry);
  currentHit->setPhiAtEntry(PhiAtEntry);

  currentHit->setEntry(hitPoint);
  currentHit->setExit(exitPoint);
  currentHit->setLocalEntry(theLocalEntryPoint);
  currentHit->setLocalExit(theLocalExitPoint);

  currentHit->setParentId(ParentId);
  currentHit->setVx(Vx);
  currentHit->setVy(Vy);
  currentHit->setVz(Vz);
  
  currentHit->set_p_x(p_x);
  currentHit->set_p_y(p_y);
  currentHit->set_p_z(p_z);

  StoreHit(currentHit);
// LogDebug("TotemRP") << "STORED HIT IN: " << unitID << std::endl;
}	 


G4ThreeVector Totem_RP_SD::SetToLocal(G4ThreeVector global)
{
  G4ThreeVector localPoint;
  const G4VTouchable* touch= preStepPoint->GetTouchable();
  localPoint = touch->GetHistory()->GetTopTransform().TransformPoint(global);
  return localPoint;
}
     

void Totem_RP_SD::EndOfEvent(G4HCofThisEvent* )
{
  // here we loop over transient hits and make them persistent
  for (int j=0; j<theHC->entries() && j<15000; j++)
  {
    Totem_RP_G4Hit* aHit = (*theHC)[j];
    
    Local3DPoint Entrata(aHit->getLocalEntry().x(),
       aHit->getLocalEntry().y(),
       aHit->getLocalEntry().z());
    Local3DPoint Uscita(aHit->getLocalExit().x(),
       aHit->getLocalExit().y(),
       aHit->getLocalExit().z());
    slave->processHits(PSimHit(Entrata,Uscita,
             aHit->getPabs(), aHit->getTof(),
             aHit->getEnergyLoss(), aHit->getParticleType(),
             aHit->getUnitID(), aHit->getTrackID(),
             aHit->getThetaAtEntry(),aHit->getPhiAtEntry()));
  }
  Summarize();
}
     

void Totem_RP_SD::Summarize()
{
}

void Totem_RP_SD::clear()
{
} 

void Totem_RP_SD::DrawAll()
{
} 

void Totem_RP_SD::PrintAll()
{
  LogDebug("TotemRP") << "Totem_RP_SD: Collection " << theHC->GetName() << std::endl;
  theHC->PrintAllHits();
}

void Totem_RP_SD::fillHits(edm::PSimHitContainer& c, const std::string & n) {
  if (slave->name() == n) {c=slave->hits();}
}

void Totem_RP_SD::SetNumberingScheme(TotemRPVDetectorOrganization* scheme)
{
  if (numberingScheme)
    delete numberingScheme;
  numberingScheme = scheme;
}

void Totem_RP_SD::update(const BeginOfEvent * i){
  clearHits();
  eventno = (*i)()->GetEventID();
}

void Totem_RP_SD::update (const ::EndOfEvent*)
{
}

void Totem_RP_SD::clearHits(){
  slave->Initialize();
}

bool Totem_RP_SD::IsPrimary(const G4Track * track)
{
  TrackInformation* info 
    = dynamic_cast<TrackInformation*>( track->GetUserInformation() );
  return info && info->isPrimary();
}

