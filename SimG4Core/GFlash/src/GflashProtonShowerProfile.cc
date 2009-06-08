#include "SimG4Core/GFlash/interface/GflashProtonShowerProfile.h"

void GflashProtonShowerProfile::loadParameters(const G4FastTrack& fastTrack)
{
  setShowerType(fastTrack);

  // energy scale
  //@@@ need additional parameterization for forward detectors

  G4double einc = fastTrack.GetPrimaryTrack()->GetKineticEnergy()/GeV;
  G4ThreeVector position = fastTrack.GetPrimaryTrack()->GetPosition()/cm;
  G4int showerType = getShowerType();

  double energyMeanHcal = 0.0;
  double energySigmaHcal = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5) {

    G4double r1 = 0.0;
    G4double r2 = 0.0;

    //@@@ need energy dependent parameterization and put relevant parameters into GflashNameSpace
    //@@@ put energy dependent energyRho based on tuning with testbeam data

    const double correl_hadem[4] = { -7.8255e-01,  1.7976e-01, -8.8001e-01,  2.3474e+00 };
    G4double energyRho =  fTanh(einc,correl_hadem); 

    //      do {
    r1 = theRandGauss->fire();
    G4double pscale = 5.0463e-01-8.1210e-02*std::tanh(1.8231*(std::log(einc)-2.7472));
    G4double tscale = 0.035+0.045*std::tanh(1.5*(std::log(einc)-2.5));
    energyScale[Gflash::kESPM] = einc*(pscale + (0.4/einc)*depthScale(position.getRho(),151.,22.)
				       +(fTanh(einc,Gflash::emscale[2]) + tscale*depthScale(position.getRho(),151.,22.) )*r1);
    //      }
    //      while (energyScale[Gflash::kESPM] < 0.0);
    
    //@@@extend depthScale for HE
    energyMeanHcal  = (fTanh(einc,Gflash::hadscale[0]) +
		       (0.8297+0.2359*tanh(-0.8*(log(einc)-4.0)))*depthScale(position.getRho(),129.,22.));
    energySigmaHcal = (fTanh(einc,Gflash::hadscale[2]) +
		       fTanh(einc,Gflash::hadscale[3])*depthScale(position.getRho(),129.,22.));
    //Hcal energy dependent scale
    //energyMeanHcal *= 1.+(-0.02+0.02*tanh(6.0*(energyScale[Gflash::kESPM]/einc-0.1)))*(0.5+0.5*tanh(0.025*(einc-150.0)));
    energyMeanHcal *= 1.+(-0.015+0.015*tanh(6.0*(energyScale[Gflash::kESPM]/einc-0.1)))*(0.5+0.5*tanh(0.025*(einc-150.0)));
    //      energySigmaHcal *= (1.1-0.2*tanh(0.015*(einc-50.0)));
    //      energySigmaHcal *= (1.09-0.3*tanh(0.010*(einc-50.0)));
    energySigmaHcal *= (1.05-0.4*tanh(0.010*(einc-80.0)));
    
    //      do {
    r2 = theRandGauss->fire();
    energyScale[Gflash::kHB] =
      exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 ))-0.05*einc;
    //      }
    //      while (energyScale[Gflash::kHB] < 0.0);
  }
  else if(showerType == 2 || showerType == 6 || showerType == 3 || showerType == 7) { 
    //Hcal response for mip-like pions (mip)
    //@@@ test based on test beam scale
    double gap_corr = 1.0;
    
    energyMeanHcal  = fTanh(einc,Gflash::protonscale[0]);
    energySigmaHcal = fTanh(einc,Gflash::protonscale[1]);
    gap_corr = fTanh(einc,Gflash::protonscale[2]);
         
    if(showerType == 2 || showerType == 6) {
      //      do {
      energyScale[Gflash::kHB] = 
	exp(energyMeanHcal+1.15*energySigmaHcal*theRandGauss->fire())-2.0
	- gap_corr*einc*depthScale(position.getRho(),179.,28.);
      //      }
      //      while (energyScale[Gflash::kHB] < 0.0 );
    }
    else {
      //      do {
      energyScale[Gflash::kHB] = 
	exp(energyMeanHcal+energySigmaHcal*theRandGauss->fire())-2.0;
      //      }
      //      while (energyScale[Gflash::kHB] < 0.0 );
    }
  }

  energyScale[Gflash::kENCA] = energyScale[Gflash::kESPM];
  energyScale[Gflash::kHE] = energyScale[Gflash::kHB];

  // parameters for the longitudinal profiles
  //@@@check longitudinal profiles of endcaps for possible varitations
  //correlation and fluctuation matrix of longitudinal parameters

  G4double *rhoHcal = new G4double [2*Gflash::NPar];
  G4double *correlationVectorHcal = new G4double [Gflash::NPar*(Gflash::NPar+1)/2];

  //for now, until we have a separate parameterization for Endcap 
  bool isEndcap = false;
  if(showerType>3) {
    showerType -= 4;
    isEndcap = true;
  }
  if(showerType==0) showerType = 1; //no separate parameterization before crystal

  //Hcal parameters are always needed regardless of showerType

  for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) {
    rhoHcal[i] = fTanh(einc,Gflash::rho[i + showerType*2*Gflash::NPar]);
  }

  correlationVectorHcal = getFluctuationVector(rhoHcal);

  G4double normalZ[Gflash::NPar];
  for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = theRandGauss->fire();
  
  for(int i = 0 ; i < Gflash::NPar ; i++) {
    double correlationSum = 0.0;
    for(int j = 0 ; j < i+1 ; j++) {
      correlationSum += correlationVectorHcal[i*(i+1)/2+j]*normalZ[j];
    }
    longHcal[i] = fTanh(einc,Gflash::par[i+showerType*Gflash::NPar]) +
                  fTanh(einc,Gflash::par[i+(4+showerType)*Gflash::NPar])*correlationSum;
  }

  delete [] rhoHcal;
  delete [] correlationVectorHcal;

  // lateral parameters for Hcal

  for (G4int i = 0 ; i < Gflash::Nrpar ; i++) {
    lateralPar[Gflash::kHB][i] = fLnE1(einc,Gflash::rpar[i+showerType*Gflash::Nrpar]);

    //begin---tuning for pure hadronic response: +10%
    if(showerType==3 && i == 0) lateralPar[Gflash::kHB][i] *= 1.1;
    //endof---tuning for pure hadronic response

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
    for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) rhoEcal[i] = fTanh(einc,Gflash::rho[i]);

    correlationVectorEcal = getFluctuationVector(rhoEcal);

    for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = theRandGauss->fire();
    for(int i = 0 ; i < Gflash::NPar ; i++) {
      double correlationSum = 0.0;
      for(int j = 0 ; j < i+1 ; j++) {
	correlationSum += correlationVectorEcal[i*(i+1)/2+j]*normalZ[j];
      }
      longEcal[i] = fTanh(einc,Gflash::par[i]) +
	fTanh(einc,Gflash::par[i+4*Gflash::NPar])*correlationSum;

    }

    delete [] rhoEcal;
    delete [] correlationVectorEcal;

    // lateral parameters for Ecal

    for (G4int i = 0 ; i < Gflash::Nrpar ; i++) {
      lateralPar[Gflash::kESPM][i] = fLnE1(einc,Gflash::rpar[i]);
      lateralPar[Gflash::kENCA][i] = lateralPar[Gflash::kESPM][i];
    }
  }

  // parameters for the sampling fluctuation

   for(G4int i = 0 ; i < Gflash::kNumberCalorimeter ; i++) {
    averageSpotEnergy[i] = std::pow(Gflash::SAMHAD[0][i],2) // resolution 
      + std::pow(Gflash::SAMHAD[1][i],2)/einc               // noisy
      + std::pow(Gflash::SAMHAD[2][i],2)*einc;              // constant 
  }

}
