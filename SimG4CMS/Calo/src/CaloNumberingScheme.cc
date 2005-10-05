///////////////////////////////////////////////////////////////////////////////
// File: CaloNumberingScheme.cc
// Description: Base class for numbering scheme of calorimeters
///////////////////////////////////////////////////////////////////////////////
#include "SimG4CMS/Calo/interface/CaloNumberingScheme.h"
#include "globals.hh"

//#define debug

//UserVerbosity CaloNumberingScheme::cout("CaloNumberingScheme","silent","CaloNumberingScheme");

int CaloNumberingScheme::detectorLevel(const G4Step* aStep) const {

  //Find number of levels
  const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
  int level = 0;
  if (touch) level = ((touch->GetHistoryDepth())+1);
  return level;
}

void CaloNumberingScheme::detectorLevel(const G4Step* aStep, int& level,
					int* copyno, G4String* name) const {

  //Get name and copy numbers
  if (level > 0) {
    const G4VTouchable* touch = aStep->GetPreStepPoint()->GetTouchable();
    for (int ii = 0; ii < level; ii++) {
      int i      = level - ii - 1;
      name[ii]   = touch->GetVolume(i)->GetName();
      copyno[ii] = touch->GetReplicaNumber(i);
    }
  }
}

// int CaloNumberingScheme::getUnitWithMaxEnergy(map<int,float,less<int> >& themap){

//   //look for max
//   int UnitWithMaxEnergy = 0;
//   float maxEnergy = 0.;
	
//   for(map<int,float,less<int> >::iterator iter = themap.begin();
//       iter != themap.end(); iter++){
	    
//     if(	maxEnergy < (*iter).second) {
//       maxEnergy = (*iter).second;	
//       UnitWithMaxEnergy = (*iter).first;
//     }				
//   }
// // #ifdef debug
// //   cout.testOut << "CaloNumberingScheme: *** max energy of " << maxEnergy 
// // 	       << " MeV was found in Unit id " << UnitWithMaxEnergy;
// //   int det,z,eta,phi;
// //   myPacker.unpackEcalIndex(UnitWithMaxEnergy, det, z, eta, phi);
// //   cout.testOut << " corresponding to z= " << z << " eta= " << eta << " phi = " 
// // 	       << phi << endl;
// // #endif
//   return UnitWithMaxEnergy;

// }

// float CaloNumberingScheme::energyInMatrix(int nCellInEta, int nCellInPhi,
// 					  int crystalWithMaxEnergy, 
// 					  map<int,float,less<int> >& themap){

//   int det,z,eta,phi;
//   myPacker.unpackEcalIndex(crystalWithMaxEnergy, det, z, eta, phi);
//   int ncristals=0;
	
//   int goBackInEta = nCellInEta/2;
//   int goBackInPhi = nCellInPhi/2;
//   int startEta = eta-goBackInEta;
//   int startPhi = phi-goBackInPhi;

//   float totalEnergy = 0.;
  
//   for (int ieta=startEta; ieta<startEta+nCellInEta; ieta++) {
//     for (int iphi=startPhi; iphi<startPhi+nCellInPhi; iphi++) {
      
//       int index = myPacker.packEcalIndex(det,z,ieta,iphi);
//       totalEnergy += themap[index];
//       ncristals+=1;
// #ifdef debug
//       cout.debugOut << "CaloNumberingScheme: ieta - iphi - E = " << ieta 
// 		    << "  " << iphi << " "  << themap[index] << endl;
// #endif
//     }
//   }
	
// #ifdef debug   	
//   cout.testOut << "CaloNumberingScheme: energy in " << nCellInEta 
// 	       << " cells in eta times " << nCellInPhi 
// 	       << " cells in phi matrix = " << totalEnergy
// 	       << " for " << ncristals << " crystals" << endl;
// #endif
//   return totalEnergy;

// }   
