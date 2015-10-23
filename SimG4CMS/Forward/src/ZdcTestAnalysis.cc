
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "DataFormats/Math/interface/Point3D.h"

#include "SimG4CMS/Forward/interface/ZdcTestAnalysis.h"
#include "SimG4CMS/Forward/interface/ZdcNumberingScheme.h"

#include "TFile.h"
#include <cmath>
#include <iostream>
#include <iomanip>

enum ntzdcs_elements {
  ntzdcs_evt, ntzdcs_trackid, ntzdcs_charge, ntzdcs_pdgcode, ntzdcs_x, ntzdcs_y, ntzdcs_z, ntzdcs_stepl,
  ntzdcs_stepe, ntzdcs_eta, ntzdcs_phi, ntzdcs_vpx, ntzdcs_vpy, ntzdcs_vpz, ntzdcs_idx, ntzdcs_idl,
  ntzdcs_pvtype, ntzdcs_ncherphot
};

enum ntzdce_elements {
  ntzdce_evt,ntzdce_ihit,ntzdce_fiberid,ntzdce_zside,ntzdce_subdet,ntzdce_layer,ntzdce_fiber,ntzdce_channel,
  ntzdce_enem,ntzdce_enhad,ntzdce_hitenergy,ntzdce_x,ntzdce_y,ntzdce_z,ntzdce_time,ntzdce_etot
};

ZdcTestAnalysis::ZdcTestAnalysis(const edm::ParameterSet &p){
  //constructor
  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("ZdcTestAnalysis");
  verbosity    = m_Anal.getParameter<int>("Verbosity");
  doNTzdcstep  = m_Anal.getParameter<int>("StepNtupleFlag");
  doNTzdcevent  = m_Anal.getParameter<int>("EventNtupleFlag");
  stepNtFileName = m_Anal.getParameter<std::string>("StepNtupleFileName");
  eventNtFileName = m_Anal.getParameter<std::string>("EventNtupleFileName");
   
   if (verbosity > 0)
   std::cout<<std::endl;
   std::cout<<"============================================================================"<<std::endl;
   std::cout << "ZdcTestAnalysis:: Initialized as observer"<< std::endl;
   if (doNTzdcstep  > 0){
     std::cout <<" Step Ntuple will be created"<< std::endl;
     std::cout <<" Step Ntuple file: "<<stepNtFileName<<std::endl;
   }
   if (doNTzdcevent > 0){
     std::cout <<" Event Ntuple will be created"<< std::endl;
     std::cout <<" Step Ntuple file: "<<stepNtFileName<<std::endl;
   }
   std::cout<<"============================================================================"<<std::endl;
   std::cout<<std::endl;

   if (doNTzdcstep  > 0)
     zdcstepntuple = 
       new TNtuple("NTzdcstep","NTzdcstep",
		   "evt:trackid:charge:pdgcode:x:y:z:stepl:stepe:eta:phi:vpx:vpy:vpz:idx:idl:pvtype:ncherphot");

   if (doNTzdcevent  > 0)
     zdceventntuple = 
       new TNtuple("NTzdcevent","NTzdcevent",
		   "evt:ihit:fiberid:zside:subdet:layer:fiber:channel:enem:enhad:hitenergy:x:y:z:time:etot");

   //theZdcSD = new ZdcSD("ZDCHITSB", new ZdcNumberingScheme());
}
   
ZdcTestAnalysis::~ZdcTestAnalysis() {
  // destructor
  finish();
  if (verbosity > 0) {
    std::cout << std::endl << "ZdcTestAnalysis Dextructor  -------->  End of ZdcTestAnalysis : "
      << std::endl << std::endl; 
  }

  //if (doNTzdcstep  > 0)delete zdcstepntuple;
  //if (doNTzdcevent > 0)delete zdceventntuple;

  std::cout<<"ZdcTestAnalysis: End of process"<<std::endl;
}


void ZdcTestAnalysis::update(const BeginOfJob * job) {
  //job
  std::cout<<"beggining of job"<<std::endl;;
}


//==================================================================== per RUN
void ZdcTestAnalysis::update(const BeginOfRun * run) {
  //run

 std::cout << std::endl << "ZdcTestAnalysis: Begining of Run"<< std::endl; 
  if (doNTzdcstep) { 
    std::cout << "ZDCTestAnalysis: output step file created"<< std::endl;
    TString stepfilename = stepNtFileName;
    zdcOutputStepFile = new TFile(stepfilename,"RECREATE");

  }
  
  if (doNTzdcevent) {
    std::cout << "ZDCTestAnalysis: output event file created"<< std::endl;
    TString stepfilename = eventNtFileName;
    zdcOutputEventFile = new TFile(stepfilename,"RECREATE");
  }

  eventIndex = 0;
}




void ZdcTestAnalysis::update(const BeginOfEvent * evt) {
  //event
  std::cout << "ZdcTest: Processing Event Number: "<<eventIndex<< std::endl;
  eventIndex++;
  stepIndex = 0;
}


void ZdcTestAnalysis::update(const G4Step * aStep) {
  //step;
  stepIndex++;

  if (doNTzdcstep) {
    G4StepPoint * preStepPoint = aStep->GetPreStepPoint();
    // G4StepPoint * postStepPoint= aStep->GetPostStepPoint();
    G4double stepL = aStep->GetStepLength();
    G4double stepE = aStep->GetTotalEnergyDeposit();
    
    if (verbosity >= 2)
      std::cout << "Step " << stepL << "," << stepE <<std::endl;
    
    G4Track * theTrack    = aStep->GetTrack();
    G4int theTrackID      = theTrack->GetTrackID();
    G4double theCharge    = theTrack->GetDynamicParticle()->GetCharge();
    G4String particleType = theTrack->GetDefinition()->GetParticleName();
    G4int pdgcode         = theTrack->GetDefinition()->GetPDGEncoding();
    
    G4ThreeVector vert_mom = theTrack->GetVertexMomentumDirection();
    G4double vpx = vert_mom.x();
    G4double vpy = vert_mom.y();
    G4double vpz = vert_mom.z();
    double eta = 0.5 * log( (1.+vpz) / (1.-vpz) );
    double phi = atan2(vpy,vpx);
    
    G4ThreeVector hitPoint = preStepPoint->GetPosition();
    G4ThreeVector localPoint = theTrack->GetTouchable()->GetHistory()->
    GetTopTransform().TransformPoint(hitPoint);
    
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    int idx = touch->GetReplicaNumber(0);
    int idLayer = -1;
    int thePVtype = -1;
    
    int historyDepth = touch->GetHistoryDepth();

    if (historyDepth > 0) {
      std::vector<int>                theReplicaNumbers(historyDepth);
      std::vector<G4VPhysicalVolume*> thePhysicalVolumes(historyDepth);
      std::vector<G4String>           thePVnames(historyDepth);
      std::vector<G4LogicalVolume*>   theLogicalVolumes(historyDepth);
      std::vector<G4String>           theLVnames(historyDepth);
      std::vector<G4Material*>        theMaterials(historyDepth);
      std::vector<G4String>           theMaterialNames(historyDepth);
      
      for (int jj = 0; jj < historyDepth; jj++) {
	theReplicaNumbers[jj] = touch->GetReplicaNumber(jj);
	thePhysicalVolumes[jj] = touch->GetVolume(jj);
	thePVnames[jj] = thePhysicalVolumes[jj]->GetName();
	theLogicalVolumes[jj] = thePhysicalVolumes[jj]->GetLogicalVolume();
	theLVnames[jj] = theLogicalVolumes[jj]->GetName();
	theMaterials[jj] = theLogicalVolumes[jj]->GetMaterial();
	theMaterialNames[jj] = theMaterials[jj]->GetName();
	if (verbosity >= 2)
	  std::cout << " GHD " << jj << ": " << theReplicaNumbers[jj] << ","
		       << thePVnames[jj] << "," << theLVnames[jj] << ","
		    << theMaterialNames[jj]  << std::endl;
      }

      idLayer = theReplicaNumbers[1];
      if (thePVnames[0] == "ZDC_EMLayer")
	thePVtype = 1;
      else if (thePVnames[0] == "ZDC_EMAbsorber")
	thePVtype = 2;
      else if (thePVnames[0] == "ZDC_EMFiber")
	thePVtype = 3;
      else if (thePVnames[0] == "ZDC_HadLayer")
	thePVtype = 7;
      else if (thePVnames[0] == "ZDC_HadAbsorber")
	thePVtype = 8;
      else if (thePVnames[0] == "ZDC_HadFiber")
	thePVtype = 9;
      else if (thePVnames[0] == "ZDC_PhobosLayer")
	thePVtype = 11;
      else if (thePVnames[0] == "ZDC_PhobosAbsorber")
	thePVtype = 12;
      else if (thePVnames[0] == "ZDC_PhobosFiber")
      thePVtype = 13;
      else {
	thePVtype = 0;
	if (verbosity >= 2)
	  std::cout << " pvtype=0 hd=" << historyDepth << " " << theReplicaNumbers[0] << ","
		    << thePVnames[0] << "," << theLVnames[0] << "," << theMaterialNames[0] << std::endl;
      }
    }    
    else if (historyDepth == 0) { 
      int theReplicaNumber = touch->GetReplicaNumber(0);
      G4VPhysicalVolume* thePhysicalVolume = touch->GetVolume(0);
      G4String thePVname = thePhysicalVolume->GetName();
      G4LogicalVolume * theLogicalVolume = thePhysicalVolume->GetLogicalVolume();
      G4String theLVname = theLogicalVolume->GetName();
      G4Material * theMaterial = theLogicalVolume->GetMaterial();
      G4String theMaterialName = theMaterial->GetName();
      if (verbosity >= 2)
	std::cout << " hd=0 " << theReplicaNumber << "," 
		  << thePVname << "," << theLVname << "," 
		  << theMaterialName << std::endl;
    }
    else {
      std::cout << " hd<0:  hd=" << historyDepth << std::endl;
    }

    
    double NCherPhot = -1;
    zdcsteparray[ntzdcs_evt] = (float)eventIndex;
    zdcsteparray[ntzdcs_trackid] = (float)theTrackID;
    zdcsteparray[ntzdcs_charge] = theCharge;
    zdcsteparray[ntzdcs_pdgcode] = (float)pdgcode;
    zdcsteparray[ntzdcs_x] = hitPoint.x();
    zdcsteparray[ntzdcs_y] = hitPoint.y();
    zdcsteparray[ntzdcs_z] = hitPoint.z();
    zdcsteparray[ntzdcs_stepl] = stepL;
    zdcsteparray[ntzdcs_stepe] = stepE;
    zdcsteparray[ntzdcs_eta] = eta;
    zdcsteparray[ntzdcs_phi] = phi;
    zdcsteparray[ntzdcs_vpx] = vpx;
    zdcsteparray[ntzdcs_vpy] = vpy;
    zdcsteparray[ntzdcs_vpz] = vpz;
    zdcsteparray[ntzdcs_idx] = (float)idx;
    zdcsteparray[ntzdcs_idl] = (float)idLayer;
    zdcsteparray[ntzdcs_pvtype] = thePVtype;
    zdcsteparray[ntzdcs_ncherphot] = NCherPhot;
    zdcstepntuple->Fill(zdcsteparray);

  }
}

void ZdcTestAnalysis::update(const EndOfEvent * evt) {
  //end of event 
   
  // Look for the Hit Collection
  std::cout  << std::endl << "ZdcTest::upDate(const EndOfEvent * evt) - event #" << (*evt)()->GetEventID()
	     << std::endl << "  # of aSteps followed in event = " << stepIndex << std::endl;
  
  // access to the G4 hit collections
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  std::cout << "  accessed all HC";
  
  int theZDCHCid = G4SDManager::GetSDMpointer()->GetCollectionID("ZDCHITS");
  std::cout << " - theZDCHCid = " << theZDCHCid;
  
  CaloG4HitCollection* theZDCHC = (CaloG4HitCollection*) allHC->GetHC(theZDCHCid);
  std::cout << " - theZDCHC = " << theZDCHC << std::endl;
  
  ZdcNumberingScheme * theZdcNumScheme = new ZdcNumberingScheme(1);
  
  float ETot=0., SEnergy=0.;
  int maxTime=0;
  int fiberID=0;
  unsigned int unsignedfiberID=0;
  std::map<int,float,std::less<int> > energyInFibers;
  std::map<int,float,std::less<int> > primaries;
  float totalEnergy = 0;
  int nentries = theZDCHC->entries();
  std::cout << "  theZDCHC has " << nentries << " entries" << std::endl;

  if (doNTzdcevent) {
    if (nentries > 0) {
      for (int ihit = 0; ihit < nentries; ihit++) {
	CaloG4Hit* caloHit = (*theZDCHC)[ihit];
	totalEnergy += caloHit->getEnergyDeposit();
      }

      for (int ihit = 0; ihit < nentries; ihit++) {
	CaloG4Hit* aHit = (*theZDCHC)[ihit];
	    fiberID = aHit->getUnitID();
	    unsignedfiberID = aHit->getUnitID();
	    double enEm = aHit->getEM();
	    double enHad = aHit->getHadr();
	    math::XYZPoint hitPoint = aHit->getPosition();
	    double hitEnergy = aHit->getEnergyDeposit();
	    if (verbosity >= 1)
	      std::cout << " entry #" << ihit << ": fiberID=0x" << std::hex 
			<< fiberID << std::dec << "; enEm=" << enEm 
			<< "; enHad=" << enHad << "; hitEnergy=" << hitEnergy
			<< "z=" << hitPoint.z() << std::endl;
	    energyInFibers[fiberID]+= enEm + enHad;
	    primaries[aHit->getTrackID()]+= enEm + enHad;
	    float time = aHit->getTimeSliceID();
	    if (time > maxTime) maxTime=(int)time;
	    
	    int thesubdet, thelayer, thefiber, thechannel, thez;
	    theZdcNumScheme->unpackZdcIndex(fiberID, thesubdet, thelayer, thefiber, thechannel, thez);
	    int unsignedsubdet, unsignedlayer, unsignedfiber, unsignedchannel, unsignedz;
	    theZdcNumScheme->unpackZdcIndex(unsignedfiberID, unsignedsubdet, 
					    unsignedlayer, unsignedfiber, unsignedchannel, unsignedz);
	    
	    // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);
	    // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);
	    // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);
	    // unsigned int packidx1 = packZdcIndex(thesubdet, thelayer, thefiber, thechannel, thez);

	    zdceventarray[ntzdce_evt] = (float)eventIndex;
	    zdceventarray[ntzdce_ihit] = (float)ihit;
	    zdceventarray[ntzdce_fiberid] = (float)fiberID;
	    zdceventarray[ntzdce_zside] = (float)thez;
	    zdceventarray[ntzdce_subdet] = (float)thesubdet;
	    zdceventarray[ntzdce_layer] = (float)thelayer;
	    zdceventarray[ntzdce_fiber] = (float)thefiber;
	    zdceventarray[ntzdce_channel] = (float)thechannel;
	    zdceventarray[ntzdce_enem] = enEm;
	    zdceventarray[ntzdce_enhad] = enHad;
	    zdceventarray[ntzdce_hitenergy] = hitEnergy;
	    zdceventarray[ntzdce_x] = hitPoint.x();
	    zdceventarray[ntzdce_y] = hitPoint.y();
	    zdceventarray[ntzdce_z] = hitPoint.z();
	    zdceventarray[ntzdce_time] = time;
	    zdceventarray[ntzdce_etot] = totalEnergy;
	    zdceventntuple->Fill(zdceventarray);
      }
      
      for (std::map<int,float,std::less<int> >::iterator is = energyInFibers.begin();
	   is!= energyInFibers.end(); is++)
	{
	  ETot = (*is).second;
	  SEnergy += ETot;
	}

      // Find Primary info:
      int trackID = 0;
      G4PrimaryParticle* thePrim=0;
      G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
      std::cout << "Event has " << nvertex << " vertex" << std::endl;
      if (nvertex==0)
	std::cout << "ZdcTest End Of Event  ERROR: no vertex" << std::endl;
      
      for (int i = 0 ; i<nvertex; i++) {
	
	G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
	if (avertex == 0) {
	  std::cout << "ZdcTest End Of Event ERR: pointer to vertex = 0"
		       << std::endl;
	} else {
	  std::cout << "Vertex number :" <<i << std::endl;
	  int npart = avertex->GetNumberOfParticle();
	  if (npart ==0)
	    std::cout << "ZdcTest End Of Event ERR: no primary!" << std::endl;
	  if (thePrim==0) thePrim=avertex->GetPrimary(trackID);
	}
      }
      
      double px=0.,py=0.,pz=0.;
      double pInit = 0.;
      
      if (thePrim != 0) {
	px = thePrim->GetPx();
	py = thePrim->GetPy();
	pz = thePrim->GetPz();
	pInit = sqrt(pow(px,2.)+pow(py,2.)+pow(pz,2.));
	if (pInit==0) {
	  std::cout << "ZdcTest End Of Event  ERR: primary has p=0 " << std::endl;
	}
      } else {
	std::cout << "ZdcTest End Of Event ERR: could not find primary "
		     << std::endl;
      }
      
    } // nentries > 0
    
 
 } // createNTzdcevent
  
  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10)
    std::cout << " ZdcTest Event " << iEvt << std::endl;
  else if ((iEvt < 100) && (iEvt%10 == 0))
    std::cout << " ZdcTest Event " << iEvt << std::endl;
  else if ((iEvt < 1000) && (iEvt%100 == 0))
    std::cout << " ZdcTest Event " << iEvt << std::endl;
  else if ((iEvt < 10000) && (iEvt%1000 == 0))
    std::cout << " ZdcTest Event " << iEvt << std::endl;
}

void ZdcTestAnalysis::update(const EndOfRun * run) {;}

void ZdcTestAnalysis::finish(){
  if (doNTzdcstep) {
    zdcOutputStepFile->cd();
    zdcstepntuple->Write();
    std::cout << "ZdcTestAnalysis: Ntuple step  written for event: "<<eventIndex<<std::endl;
    zdcOutputStepFile->Close();
    std::cout << "ZdcTestAnalysis: Step file closed" << std::endl;
  }
  

 if (doNTzdcevent) {
   zdcOutputEventFile->cd();
   zdceventntuple->Write("",TObject::kOverwrite);
   std::cout << "ZdcTestAnalysis: Ntuple event written for event: "<<eventIndex<<std::endl;   
   zdcOutputEventFile->Close();
   std::cout << "ZdcTestAnalysis: Event file closed" << std::endl;
 }
 
}
