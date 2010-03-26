#include "SimG4Core/HelpfulWatchers/interface/MonopoleSteppingAction.h"

#include "SimG4Core/Notification/interface/BeginOfJob.h"
#include "SimG4Core/Notification/interface/BeginOfRun.h"
#include "SimG4Core/Notification/interface/BeginOfTrack.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "G4Track.hh"
#include "G4Run.hh"
#include "G4Event.hh"
#include "G4ParticleTable.hh"

#include "CLHEP/Units/GlobalSystemOfUnits.h"
#include "CLHEP/Units/GlobalPhysicalConstants.h"

MonopoleSteppingAction::MonopoleSteppingAction(edm::ParameterSet const & p) :
  actOnTrack (false), bZ(0) {
  mode = p.getUntrackedParameter<bool>("ChangeFromFirstStep",true);
  edm::LogInfo("SimG4CoreWatcher") << "MonopoleSeppingAction set mode for"
				   << " start at first step to " << mode;
}

MonopoleSteppingAction::~MonopoleSteppingAction() {}

void MonopoleSteppingAction::update(const BeginOfJob* job) {

  edm::LogInfo("SimG4CoreWatcher") << "MonopoleSteppingAction::Enter Begin of Job\n";
  const edm::EventSetup* iSetup = (*job)();
  edm::ESHandle<MagneticField> bFieldH;
  iSetup->get<IdealMagneticFieldRecord>().get(bFieldH);
  const MagneticField *bField = bFieldH.product();
  const GlobalPoint p(0,0,0);
  bZ = (bField->inTesla(p)).z();
  edm::LogInfo("SimG4CoreWatcher") << "Magnetic Field (X): " 
				   << (bField->inTesla(p)).x() << " Y: "
				   << (bField->inTesla(p)).y() << " Z: " << bZ;
}

void MonopoleSteppingAction::update(const BeginOfRun* ) {

  G4ParticleTable * partTable = G4ParticleTable::GetParticleTable();
  for (int ii= 0; ii < partTable->size(); ii++) {
    G4ParticleDefinition * particle = partTable->GetParticle(ii);
    std::string particleName = (particle->GetParticleName()).substr(0,8);
    if (strcmp(particleName.c_str(),"monopole") == 0) {
      pdgCode.push_back(particle->GetPDGEncoding());
    }
  }
  edm::LogInfo("SimG4CoreWatcher") << "MonopoleSeppingAction Finds "
				   << pdgCode.size() << " candidates";
  for (unsigned int ii=0; ii<pdgCode.size(); ++ii) {
    edm::LogInfo("SimG4CoreWatcher") << "PDG Code[" << ii << "] = "
				     << pdgCode[ii];
  }
  cMevToJ   = CLHEP::e_SI/CLHEP::eV;
  cMeVToKgMByS = CLHEP::e_SI*CLHEP::meter/(CLHEP::eV*CLHEP::c_light*CLHEP::second);
  cInMByS   = CLHEP::c_light*CLHEP::second/CLHEP::meter;
  magCharge = CLHEP::e_SI/CLHEP::fine_structure_const * 0.5;
  if (bZ < 0.0) bZ  = 3.8; // in Tesla (later take from ES)
  edm::LogInfo("SimG4CoreWatcher") << "MonopoleSeppingAction Constants " 
				   << cMevToJ << ", " << cMeVToKgMByS
				   << ", " << cInMByS << ", " << magCharge;
}

void MonopoleSteppingAction::update(const BeginOfTrack * trk) {

  actOnTrack = false;
  if (pdgCode.size() > 0) {
    const G4Track * aTrack = (*trk)();
    int code = aTrack->GetDefinition()->GetPDGEncoding();
    if (std::count(pdgCode.begin(),pdgCode.end(),code) > 0) {
      actOnTrack = true;
      eStart     = aTrack->GetTotalEnergy();
      pxStart    = aTrack->GetMomentum().x();
      pyStart    = aTrack->GetMomentum().y();
      pzStart    = aTrack->GetMomentum().z();
      dirxStart  = aTrack->GetMomentumDirection().x();
      diryStart  = aTrack->GetMomentumDirection().y();
      dirzStart  = aTrack->GetMomentumDirection().z();
      LogDebug("SimG4CoreWatcher") << "MonopoleSeppingAction Track " 
				   << code << " Flag " << actOnTrack
				   << "(px,py,pz,E) = (" << pxStart/GeV
				   << ", " << pyStart/GeV << ", "
				   << pzStart/GeV << ", " << eStart/GeV << ")";
    }
  }
}

void MonopoleSteppingAction::update(const G4Step* aStep) {

  if (actOnTrack) {
    double eT, pT, pZ, tStep;
    G4Track* aTrack = aStep->GetTrack();
    G4ThreeVector initialPosition(0,0,0);
    if (mode) {
      tStep = aTrack->GetGlobalTime();
      eT    = eStart*std::sqrt(dirxStart*dirxStart+diryStart*diryStart);
      pT    = std::sqrt(pxStart*pxStart+pyStart*pyStart);
      pZ    = pzStart;
    } else {
      G4ThreeVector dirStep = aTrack->GetMomentumDirection();
      double        lStep   = aTrack->GetStepLength();
      double        xStep   = aTrack->GetPosition().x()-lStep*dirStep.x();
      double        yStep   = aTrack->GetPosition().y()-lStep*dirStep.y();
      double        zStep   = aTrack->GetPosition().z()-lStep*dirStep.z();
      double        vStep   = aTrack->GetVelocity();
      initialPosition       = G4ThreeVector(xStep,yStep,zStep);
      tStep                 = lStep/vStep;
      double        eStep   = aTrack->GetTotalEnergy();
      eT                    = eStep*dirStep.perp();
      pT                    = aTrack->GetMomentum().perp();
      pZ                    = aTrack->GetMomentum().z();
    }
    LogDebug("SimG4CoreWatcher") << "MonopoleSeppingAction: tStep " <<tStep
				 << " eT " << eT << " pT " <<pT << " pZ " <<pZ;
    eT = eT*cMevToJ;
    pT = pT*cMeVToKgMByS;
    pZ = pZ*cMeVToKgMByS;
    tStep /= second;
    double fac0 = magCharge*bZ*cInMByS;
    double fac2 = pZ*cInMByS/eT;
    double fac1 = fac0*cInMByS*tStep/eT + fac2;
    G4ThreeVector ez(0,0,1), et(std::sqrt(0.5),std::sqrt(0.5),0);
    G4ThreeVector displacement = 
      (((eT/fac0)*(std::sqrt(1+fac1*fac1)-std::sqrt(1+fac2*fac2)))*ez +
       (pT/(magCharge*bZ)*(asinh(fac1+fac2)-asinh(fac2)))*et)*CLHEP::m;
    
    LogDebug("SimG4CoreWatcher") << "MonopoleSeppingAction: Initial " 
				 << initialPosition << " Displacement "
				 << displacement << " Final "
				 << (initialPosition+displacement);
    aTrack->SetPosition(initialPosition+displacement);
  }
}
