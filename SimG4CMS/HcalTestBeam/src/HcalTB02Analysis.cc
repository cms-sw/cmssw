// -*- C++ -*-
//
// Package:     HcalTestBeam
// Class  :     HcalTB02Analysis
//
// Implementation:
//     Main analysis class for Hcal Test Beam 2002 Analysis
//
// Original Author:
//         Created:  Sun May 21 10:14:34 CEST 2006
// $Id$
//
  
// system include files
#include <cmath>
#include <iostream>
#include <iomanip>
  
// user include files
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02Analysis.h"

// to retreive hits
#include "SimG4CMS/Calo/interface/CaloG4Hit.h"
#include "SimG4CMS/Calo/interface/CaloG4HitCollection.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02HcalNumberingScheme.h"
#include "SimG4CMS/HcalTestBeam/interface/HcalTB02XtalNumberingScheme.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "G4SDManager.hh"
#include "G4VProcess.hh"
#include "G4HCofThisEvent.hh"
#include "CLHEP/Units/SystemOfUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

//
// constructors and destructor
//

HcalTB02Analysis::HcalTB02Analysis(const edm::ParameterSet &p): 
  tuplesManager(0), tuples(0), histo(0) {

  edm::ParameterSet m_Anal = p.getParameter<edm::ParameterSet>("HcalTB02Analysis");
  hcalOnly      = m_Anal.getUntrackedParameter<bool>("HcalClusterOnly",true);
  fileNameTuple = m_Anal.getUntrackedParameter<std::string>("TupleFileName", "HcalTB02Tuple.root");
  names         = m_Anal.getParameter<std::vector<std::string> >("Names");
  
  edm::LogInfo("HcalTBSim") << "HcalTB02Analysis:: Initialised as observer of "
			    << "BeginOfJob/BeginOfEvent/EndOfEvent with "
			    << "Parameter values:\n \thcalOnly = " << hcalOnly;

  histo  = new HcalTB02Histo(m_Anal);
} 
   
HcalTB02Analysis::~HcalTB02Analysis() {

  finish();

  if (histo)   {
    delete histo;
    histo  = 0;
  }
  edm::LogInfo("HcalTBSim") << "HcalTB02Analysis is deleting";
}

//
// member functions
//

void HcalTB02Analysis::update(const BeginOfJob * ) {

  // Ntuples
  tuplesManager.reset(new HcalTB02HistoManager(fileNameTuple));
}

void HcalTB02Analysis::update(const BeginOfEvent * evt) {
 
  edm::LogInfo("HcalTBSim") << "HcalTB02Analysis: =====> Begin of event = "
			    << (*evt) ()->GetEventID();

  // create tuple object
  tuples = new HcalTB02HistoClass();
}

void HcalTB02Analysis::update(const EndOfEvent * evt) {

  // Look for the Hit Collection
  LogDebug("HcalTBSim") << "HcalTB02Analysis::Fill event " 
			<< (*evt)()->GetEventID();

  // access to the G4 hit collections 
  G4HCofThisEvent* allHC = (*evt)()->GetHCofThisEvent();
  int ihit = 0;    
 
  // HCAL
  std::string sd = names[0];
  int HCHCid = G4SDManager::GetSDMpointer()->GetCollectionID(sd);
  CaloG4HitCollection* theHCHC = (CaloG4HitCollection*) allHC->GetHC(HCHCid);
  HcalTB02HcalNumberingScheme *org = new HcalTB02HcalNumberingScheme();   
  LogDebug("HcalTBSim") << "HcalTB02Analysis :: Hit Collection for " << sd 
			<< " of ID " << HCHCid << " is obtained at " <<theHCHC;

  int nentries = 0;
  nentries = theHCHC->entries();
  if (nentries==0) return;

  int xentries = 0;  
  int XTALSid=0;
  CaloG4HitCollection* theXTHC=0;

  if (!hcalOnly) {
    // XTALS
    sd      = names[1];
    XTALSid = G4SDManager::GetSDMpointer()->GetCollectionID(sd);
    //    assert (XTALSid != 0);
    theXTHC = (CaloG4HitCollection*) allHC->GetHC(XTALSid);
    //    assert (theXTHC != 0);
    //HcalTB02XtalNumberingScheme *xorg = new HcalTB02XtalNumberingScheme();
    LogDebug("HcalTBSim") << "HcalTB02Analysis :: Hit Collection for " << sd
			  << " of ID " << XTALSid << " is obtained at " 
			  << theXTHC;
    xentries = theXTHC->entries();
    if (xentries==0) return;
  }

  LogDebug("HcalTBSim") << "HcalTB02Analysis :: There are " << nentries 
			<< " HCal hits, and" << xentries  << " xtal hits";

  float ETot=0., SEnergy=0., SEnergyN=0.;
  float xETot=0., xSEnergy=0., xSEnergyN=0.;
  float E5x5Matrix=0., E5x5MatrixN=0, E7x7Matrix=0., E7x7MatrixN=0;
  float xE3x3Matrix=0., xE5x5Matrix=0., xE3x3MatrixN=0., xE5x5MatrixN=0.;
  //float maxE = 0.; 
  //int maxI=0, 
  int maxTime=0;
  int scintID=0, xtalID=0;
  std::map<int,float,less<int> > energyInScints, energyInCrystals;
  std::map<int,float,less<int> > primaries;

  // HCAL

  if (HCHCid >= 0 && theHCHC > 0) {
    for ( ihit = 0; ihit < nentries; ihit++) {

      CaloG4Hit* aHit = (*theHCHC)[ihit]; 
      scintID     = aHit->getUnitID();
      int layer   = org->getlayerID(scintID);
      float enEm  = aHit->getEM();
      float enhad = aHit->getHadr();
      
      if (layer==0) {
	enEm =enEm/2.;
	enhad=enhad/2.;
      }

      energyInScints[scintID]+= enEm + enhad;
      primaries[aHit->getTrackID()]+= enEm + enhad;
      float time = aHit->getTimeSliceID();
      if (time > maxTime) maxTime=(int)time;
      histo->fillAllTime(time);	  
      
    }           
 
    //
    // Profile
    //

    float TowerEne[8][18], TowerEneCF[8][18], LayerEne[19], EnRing[100];
    for (int iphi=0 ; iphi<8; iphi++) {
      for (int jeta=0 ; jeta<18; jeta++) {
	TowerEne[iphi][jeta]=0.;
	TowerEneCF[iphi][jeta]=0.;
      }
    }
    
    for (int ilayer=0; ilayer<19; ilayer++) LayerEne[ilayer]=0.;
    for (int iring=0; iring<100; iring++) EnRing[iring]=0.;
    
    for (map<int,float,less<int> >::iterator is = energyInScints.begin();
	 is!= energyInScints.end(); is++) {

      ETot = (*is).second;

      int layer = org->getlayerID((*is).first);

      if ( (layer!=17) && (layer!=18) )  {
	    
	float eta = org->getetaID((*is).first);
	float phi = org->getphiID((*is).first);
	    
	SEnergy += ETot;
	TowerEne[(int)phi][(int)eta] += ETot;

	TowerEneCF[(int)phi][(int)eta] += ETot*(1.+ 0.1*RandGauss::shoot() );
	double dR=0.08727*sqrt( (eta-8.)*(eta-8.) + (phi-3.)*(phi-3.) );
	EnRing[(int)(dR/0.01)] += ETot;
      }

      LayerEne[layer] += ETot;
	
    }
    
    for (int ilayer=0 ; ilayer<19 ; ilayer++) {
      histo->fillProfile(ilayer,LayerEne[ilayer]/GeV);
    }
    
    for (int iring=0; iring<100; iring++)
      histo->fillTransProf(0.01*iring+0.005,EnRing[iring]/GeV);
    
    for (int iphi=0 ; iphi<8; iphi++) {
      for (int jeta=0 ; jeta<18; jeta++) {
	
	//SEnergyN += TowerEneCF[iphi][jeta] + 3.2*RandGauss::shoot(); // LHEP
	SEnergyN += TowerEneCF[iphi][jeta] + 3.*RandGauss::shoot(); // QGSP

	//double dR=0.08727*sqrt( (jeta-8.)*(jeta-8.)+(iphi-3.)*(iphi-3.) );
	//cout.testOut << " phi= " << iphi << " eta= " << jeta 
	//	     << " TowerEne[iphi,jeta]= " << TowerEne[iphi][jeta] 
	//	     << "dR= "  << dR << endl;
	
      	//double Rand = 3.2*RandGauss::shoot(); // LHEP
      	double Rand = 3.*RandGauss::shoot(); // QGSP
	
	if ( (iphi>=0) && (iphi<7) ) {
	  if ( (jeta>=5) && (jeta<12) ) {
		
	    E7x7Matrix += TowerEne[iphi][jeta];
	    E7x7MatrixN += TowerEneCF[iphi][jeta] + Rand;
		
	    if ( (iphi>=1) && (iphi<6) ) {
	      if ( (jeta>=6) && (jeta<11) ) {
			
		E5x5Matrix += TowerEne[iphi][jeta];
		E5x5MatrixN += TowerEneCF[iphi][jeta] + Rand;
			
	      }
	    }
		
	  }
	}
	
      }
    }
    
    //
    // Find Primary info:
    //	
    int trackID = 0;
    int particleType = 0;
    G4PrimaryParticle* thePrim=0;
    G4int nvertex = (*evt)()->GetNumberOfPrimaryVertex();
    LogDebug("HcalTBSim") << "HcalTB02Analysis :: Event has " << nvertex 
			  << " vertex";
    if (nvertex==0)
      edm::LogWarning("HcalTBSim") << "HcalTB02Analysis:: End Of Event  "
				   << "ERROR: no vertex";

    for (int i = 0 ; i<nvertex; i++) {
	
      G4PrimaryVertex* avertex = (*evt)()->GetPrimaryVertex(i);
      if (avertex == 0) {
	edm::LogWarning("HcalTBSim") << "HcalTB02Analysis:: End Of Event "
				     << "ERROR: pointer to vertex = 0";
      } else {
	int npart = avertex->GetNumberOfParticle();
	LogDebug("HcalTBSim") << "HcalTB02Analysis::Vertex number :" << i 
			      << " with " << npart << " particles";
	if (thePrim==0) thePrim=avertex->GetPrimary(trackID);
      }
    }
    
    double px=0.,py=0.,pz=0.;
    double eta = 0., phi = 0., pInit = 0.;
    
    if (thePrim != 0) {
      px = thePrim->GetPx();
      py = thePrim->GetPy();
      pz = thePrim->GetPz();
      pInit = sqrt(pow(px,2.)+pow(py,2.)+pow(pz,2.));
      if (pInit==0) {
	edm::LogWarning("HcalTBSim") << "HcalTB02Analysis:: End Of Event "
				     << " ERROR: primary has p=0 ";
      } else {   
	float costheta = pz/pInit;
	float theta = acos(min(max(costheta,float(-1.)),float(1.)));
	eta = -log(tan(theta/2));
	if (px != 0) phi = atan(py/px);  
      }
      particleType	= thePrim->GetPDGcode();
    } else {
      LogDebug("HcalTBSim") << "HcalTB02Analysis:: End Of Event ERROR: could"
			    << " not find primary ";
    }
    
    //
    // Fill-in ntuple
    //
    CaloG4Hit* firstHit =(*theHCHC)[0];
    tuples->set_Nprim(float(primaries.size()));
    tuples->set_partType(particleType );
    tuples->set_Einit(pInit/GeV);
    tuples->set_eta( eta);
    tuples->set_phi( phi);
    tuples->set_Eentry( firstHit->getIncidentEnergy()/GeV);
    tuples->set_ETot( SEnergy/GeV );
    tuples->set_E7x7( E7x7Matrix/GeV );
    tuples->set_E5x5( E5x5Matrix/GeV );
    tuples->set_ETotN(SEnergyN/GeV);
    tuples->set_E7x7N( E7x7MatrixN/GeV );
    tuples->set_E5x5N( E5x5MatrixN/GeV );
    tuples->set_NUnit(float( energyInScints.size()));
    tuples->set_Ntimesli( float(maxTime));
    
  }// number of Hits > 0

  if (!hcalOnly) {

    // XTALS

    if (XTALSid >= 0 && theXTHC > 0) {
      for (int xihit = 0; xihit < xentries; xihit++) {

	CaloG4Hit* xaHit = (*theXTHC)[xihit]; 

	float xenEm = xaHit->getEM();
	float xenhad = xaHit->getHadr();
	xtalID = xaHit->getUnitID();
	  
	energyInCrystals[xtalID]+= xenEm + xenhad;
      }
      
      float xCrysEne[7][7];
      for (int irow=0 ; irow<7; irow++) {
	for (int jcol=0 ; jcol<7; jcol++) {
	  xCrysEne[irow][jcol]=0.;
	}
      }
	
      for (map<int,float,less<int> >::iterator is = energyInCrystals.begin();
	   is!= energyInCrystals.end(); is++) {
	int xtalID = (*is).first;
	xETot = (*is).second;
	    
	int irow = (int)(xtalID/100.);
	int jcol = (int)(xtalID-100.*irow);
	    
	xSEnergy += xETot;
	xCrysEne[irow][jcol] = xETot;
	    
	float dR=sqrt( 0.01619*0.01619*(jcol-3)*(jcol-3) + 
		       0.01606*0.01606*(irow-3)*(irow-3) );
	histo->fillTransProf(dR,xETot*1.05);
	    
	if ( (irow>0) && (irow<6) ) {
	  if ( (jcol>0) && (jcol<6) ) {
		    
	    xE5x5Matrix += xCrysEne[irow][jcol];
	    xE5x5MatrixN += xCrysEne[irow][jcol] + 108.5*RandGauss::shoot();
		    
	    if ( (irow>1) && (irow<5) ) {
	      if ( (jcol>1) && (jcol<5) ) {		
		xE3x3Matrix += xCrysEne[irow][jcol];
		xE3x3MatrixN += xCrysEne[irow][jcol] +108.5*RandGauss::shoot();
	      }    
	    }
	  }
	}	  
	
      }      

      //
      // Fill-in ntuple
      //


      if (!hcalOnly) {
	//	assert(theXTHC);
	if ( theXTHC != 0 ) {
	  CaloG4Hit* xfirstHit =(*theXTHC)[0];
	  tuples->set_xEentry( xfirstHit->getIncidentEnergy()/GeV);
	  tuples->set_xNUnit(float( energyInCrystals.size()));
	  tuples->set_xETot(xSEnergy/GeV);
	  tuples->set_xETotN(xSEnergyN/GeV);
	  tuples->set_xE5x5(xE5x5Matrix/GeV);
	  tuples->set_xE3x3(xE3x3Matrix/GeV);
	  tuples->set_xE5x5N(xE5x5MatrixN/GeV);
	  tuples->set_xE3x3N(xE3x3MatrixN/GeV);
	}
      }
	
    }// number of Hits > 0

  }
  
  
  // Writing the data to the Tree
  tuplesManager->fillTree(tuples); // (no need to delete it...)
  tuples = 0; // but avoid to reuse it...
  LogDebug("HcalTBSim") << "HcalTB02Analysis:: --- after fillTree";

  int iEvt = (*evt)()->GetEventID();
  if (iEvt < 10) 
    std::cout << " Event " << iEvt << std::endl;
  else if ((iEvt < 100) && (iEvt%10 == 0)) 
    std::cout << " Event " << iEvt << std::endl;
  else if ((iEvt < 1000) && (iEvt%100 == 0)) 
    std::cout << " Event " << iEvt << std::endl;
  else if ((iEvt < 10000) && (iEvt%1000 == 0)) 
    std::cout << " Event " << iEvt << std::endl;
}

void HcalTB02Analysis::finish() {

  /*
  //Profile 
  std::ofstream   oFile;
  oFile.open("profile.dat");
  float st[19] = {0.8,
                  0.4,  0.4,  0.4,  0.4,  0.4,   
                  0.4,  0.4,  0.4,  0.4,  0.4,
                  0.4,  0.4,  0.4,  0.4,  0.4, 
                  0.8,  1.0,  1.0};
                 
  //cm of material (brass) in front of scintillator layer i:
 
  float w[19] = {7.45,                         //iron !
		 6.00, 6.00, 6.00, 6.00, 6.00, //brass
		 6.00, 6.00, 6.00, 6.00, 6.60, //brass
		 6.60, 6.60, 6.60, 6.60, 6.60, //brass
		 8.90, 20.65, 19.5};            //brass,iron !

  for (int ilayer = 0; ilayer<19; ilayer++) {

    // Histogram mean and sigma calculated from the ROOT histos
    edm::LogInfo("HcalTBSim") << "Layer number: " << ilayer << " Mean = " 
			      << histo->getMean(ilayer) << " sigma = "   
			      << histo->getRMS(ilayer) << " LThick= "   
			      << w[ilayer] << " SThick= "   << st[ilayer];
      
    oFile << ilayer << " "  << histo->getMean(ilayer) << " " 
	  << histo->getRMS(ilayer)  << " " << w[ilayer] << " " << st[ilayer] 
	  << std::endl;

  } 
  oFile.close(); 
  */
}
