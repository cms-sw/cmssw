#define MuProp_cxx
#include "MuProp.h"
#include <TStyle.h>
#include <TCanvas.h>

#include "TVector3.h"
#include "TLorentzVector.h"
#include <iostream>

Double_t MUON_MASS = 0.1056584;

void MuProp::Loop(Double_t maxEloss)
{
   if (fChain == 0) return;

   Long64_t nentries = fChain->GetEntriesFast();

   Long64_t nbytes = 0, nb = 0;
   for (Long64_t jentry=0; jentry<nentries;jentry++) {
      Long64_t ientry = LoadTree(jentry);
      if (ientry < 0) break;
      nb = fChain->GetEntry(jentry);   nbytes += nb;

      //      Float_t p0 = sqrt(p3R[0][0]**2+p3R[0][1]**2+p3R[0][2]**2);
      TVector3 p3_0; p3_0.SetXYZ(p3R[0][0], p3R[0][1], p3R[0][2]);

      for (Int_t iPoint = 1; iPoint < nPoints; iPoint++){
	TVector3 r3Exp; r3Exp.SetXYZ(r3R[iPoint][0], r3R[iPoint][1], r3R[iPoint][2]);
	TVector3 r3Sim; r3Sim.SetXYZ( r3[iPoint][0],  r3[iPoint][1],  r3[iPoint][2]);

	TLorentzVector p3Exp; p3Exp.SetXYZM(p3R[iPoint][0], p3R[iPoint][1], p3R[iPoint][2], MUON_MASS);
	TLorentzVector p3Sim; p3Sim.SetXYZM( p3[iPoint][0],  p3[iPoint][1],  p3[iPoint][2], MUON_MASS);
	if (fabs(p3Exp.E() - p3Sim.E())>maxEloss) continue;

	Float_t rExp = r3Exp.Perp();
	Float_t dX = rExp*r3Exp.DeltaPhi(r3Sim);

	Float_t ddXdX = (r3Exp.X()*dX + rExp*r3Exp.Y())/rExp/rExp;
	Float_t ddXdY = (r3Exp.Y()*dX - rExp*r3Exp.X())/rExp/rExp;
	Float_t dXSigma = ddXdX*ddXdX*covFlat[iPoint][0]
	  + ddXdY*ddXdY*covFlat[iPoint][2]
	  + 2.*ddXdX*ddXdY*covFlat[iPoint][1];
	dXSigma = sqrt(dXSigma);

	Float_t dXPull = dX/dXSigma;

	Int_t idF[2];
	Int_t dSubDMask = (0xF<<28) | (0x7<<25);
	//wheel*station
	Int_t dtWSMask = dSubDMask | (0x3F<<19);
	//endcap*station*ring
	Int_t cscESRMask = dSubDMask | (0xFF<<10);
	idF[0] = id[iPoint] & dSubDMask;
	Int_t det =  (idF[0]>>28) & 0xF;
	Int_t subDet =  (idF[0]>>25) & 0x7;
	//muon detector only
	if ( det != 2) continue;
	//DT and CSC only
	if ( subDet != 1 && subDet != 2 ) continue;
	
	if ( subDet == 1 ){
	  idF[1] = id[iPoint] & dtWSMask;
	} else if ( subDet == 2){
	  idF[1] = id[iPoint] & cscESRMask;
	}

	for (Int_t iF = 0; iF < 2; iF++){
	  if (dX_mh1[idF[iF]] == 0){	    
	    std::string hName = Form("dX_%X", idF[iF]);	    
	    dX_mh1[idF[iF]] = new TH1F(hName.c_str(), hName.c_str(), 100, -50, 50);
	  }
	  dX_mh1[idF[iF]]->Fill(dX);
	  

	  if (dXPull_mh1[idF[iF]] == 0){
	    std::string hName = Form("dXPull_%X", idF[iF]);
	    dXPull_mh1[idF[iF]] = new TH1F(hName.c_str(), hName.c_str(), 100, -5, 5);
	  }
	  dXPull_mh1[idF[iF]]->Fill(dXPull);

	}
      }

   }
}
