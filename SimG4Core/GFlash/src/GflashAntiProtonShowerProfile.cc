#include "SimG4Core/GFlash/interface/GflashAntiProtonShowerProfile.h"

void GflashAntiProtonShowerProfile::loadParameters(const G4FastTrack& fastTrack)
{
  setShowerType(fastTrack);

  G4double einc = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  G4int showerType = getShowerType();

  // energy scale

  double energyMeanHcal = 0.0;
  double energySigmaHcal = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5) {

    G4double r1 = 0.0;
    G4double r2 = 0.0;

    //energy dependent energyRho based on tuning with testbeam data

    const double correl_hadem[4] = { -7.8255e-01,  1.7976e-01, -8.8001e-01,  2.3474e+00 };
    G4double energyRho =  fTanh(einc,correl_hadem); 

    r1 = theRandGauss->fire();
    if(einc<25) {
      energyScale[Gflash::kESPM] = einc*(1.27*fTanh(einc,Gflash::pbar_emscale[0]) + fTanh(einc,Gflash::pbar_emscale[1])*r1);
    }
    else {
      energyScale[Gflash::kESPM] = einc*(1.0*fTanh(einc,Gflash::pbar_emscale[0]) + 1.4*fTanh(einc,Gflash::pbar_emscale[1])*r1);
    }

    r2 = theRandGauss->fire();
    if(einc<25) {
      energyMeanHcal  = (fTanh(einc,Gflash::pbar_hadscale[0]) +
			 fTanh(einc,Gflash::pbar_hadscale[1])*depthScale(position.getRho(),129.,22.));
      energySigmaHcal = (fTanh(einc,Gflash::pbar_hadscale[2]) +
			 fTanh(einc,Gflash::pbar_hadscale[3])*depthScale(position.getRho(),129.,22.));
    }
    else {
      energyMeanHcal  = (fTanh(einc,Gflash::pbar_hadscale[0]) +
       			 1.5*fTanh(einc,Gflash::pbar_hadscale[1])*depthScale(position.getRho(),129.,22.));
      energySigmaHcal = (fTanh(einc,Gflash::pbar_hadscale[2]) +
       			 fTanh(einc,Gflash::pbar_hadscale[3])*depthScale(position.getRho(),129.,22.));
    }
    energyScale[Gflash::kHB] = 
      exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 ));
  }
  else if(showerType == 2 || showerType == 6 || showerType == 3 || showerType == 7) { 
    //Hcal response for mip-like pions (mip)
    double gap_corr = 1.0;
    
    energyMeanHcal  = fTanh(einc,Gflash::pbar_hadscale[4]);
    energySigmaHcal = fTanh(einc,Gflash::pbar_hadscale[5]);
    gap_corr = fTanh(einc,Gflash::pbar_hadscale[6]);
         
    if(showerType == 2 || showerType == 6) {
      energyScale[Gflash::kHB] = 
	exp(energyMeanHcal+energySigmaHcal*theRandGauss->fire())*(1.0- gap_corr*depthScale(position.getRho(),179.,28.));
    }
    else {
      energyScale[Gflash::kHB] = 
	exp(energyMeanHcal+energySigmaHcal*theRandGauss->fire());
    }
  }

  energyScale[Gflash::kENCA] = energyScale[Gflash::kESPM];
  energyScale[Gflash::kHE] = energyScale[Gflash::kHB];

  // parameters for the longitudinal profiles

  G4double *rhoHcal = new G4double [2*Gflash::NPar];
  G4double *correlationVectorHcal = new G4double [Gflash::NPar*(Gflash::NPar+1)/2];

  //@@@until we have a separate parameterization for Endcap 
  bool isEndcap = false;
  if(showerType>3) {
    showerType -= 4;
    isEndcap = true;
  }
  //no separate parameterization before crystal
  if(showerType==0) showerType = 1; 

  //Hcal parameters are always needed regardless of showerType

  for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) {
    rhoHcal[i] = fTanh(einc,Gflash::pbar_rho[i + showerType*2*Gflash::NPar]);
  }

  correlationVectorHcal = getFluctuationVector(rhoHcal);

  G4double normalZ[Gflash::NPar];
  for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = theRandGauss->fire();
  
  for(int i = 0 ; i < Gflash::NPar ; i++) {
    double correlationSum = 0.0;
    for(int j = 0 ; j < i+1 ; j++) {
      correlationSum += correlationVectorHcal[i*(i+1)/2+j]*normalZ[j];
    }
    longHcal[i] = fTanh(einc,Gflash::pbar_par[i+showerType*Gflash::NPar]) +
                  fTanh(einc,Gflash::pbar_par[i+(4+showerType)*Gflash::NPar])*correlationSum;
  }

  delete [] rhoHcal;
  delete [] correlationVectorHcal;

  // lateral parameters for Hcal

  for (G4int i = 0 ; i < Gflash::Nrpar ; i++) {
    lateralPar[Gflash::kHB][i] = fLnE1(einc,Gflash::pbar_rpar[i+showerType*Gflash::Nrpar]);
    lateralPar[Gflash::kHE][i] = lateralPar[Gflash::kHB][i];
  }

  //Ecal parameters are needed if and only if the shower starts inside the crystal

  if(showerType == 1) {
    //A depth dependent correction for the core term of R in Hcal is the linear in 
    //the shower start point while for the spread term is nearly constant

    if(!isEndcap) lateralPar[Gflash::kHB][0] -= 2.3562e-01*(position.getRho()-131.0); 
    else  lateralPar[Gflash::kHE][0] -= 2.3562e-01*(position.getZ()-332.0);

    G4double *rhoEcal = new G4double [2*Gflash::NPar];
    G4double *correlationVectorEcal = new G4double [2*Gflash::NPar];
    for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) rhoEcal[i] = fTanh(einc,Gflash::pbar_rho[i]);

    correlationVectorEcal = getFluctuationVector(rhoEcal);

    for(int i = 0 ; i < Gflash::NPar ; i++) normalZ[i] = theRandGauss->fire();
    for(int i = 0 ; i < Gflash::NPar ; i++) {
      double correlationSum = 0.0;
      for(int j = 0 ; j < i+1 ; j++) {
	correlationSum += correlationVectorEcal[i*(i+1)/2+j]*normalZ[j];
      }
      longEcal[i] = fTanh(einc,Gflash::pbar_par[i]) +
	fTanh(einc,Gflash::pbar_par[i+4*Gflash::NPar])*correlationSum;
    }

    delete [] rhoEcal;
    delete [] correlationVectorEcal;

  }
}
