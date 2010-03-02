#include "SimGeneral/GFlash/interface/GflashProtonShowerProfile.h"
#include <CLHEP/Random/RandGaussQ.h>

void GflashProtonShowerProfile::loadParameters()
{
  double einc = theShowino->getEnergy();
  Gflash3Vector position = theShowino->getPositionAtShower();
  int showerType = theShowino->getShowerType();

  // energy scale
  double energyMeanHcal = 0.0;
  double energySigmaHcal = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5) {

    double r1 = 0.0;
    double r2 = 0.0;

    const double correl_hadem[4] = { -7.8255e-01,  1.7976e-01, -8.8001e-01,  2.3474e+00 };
    double energyRho =  fTanh(einc,correl_hadem); 

    r1 = CLHEP::RandGaussQ::shoot();
    double pscale = 5.0463e-01-8.1210e-02*std::tanh(1.8231*(std::log(einc)-2.7472));
    double tscale = 0.035+0.045*std::tanh(1.5*(std::log(einc)-2.5));

    if(showerType == 0 || showerType == 1) {
      energyScale[Gflash::kESPM] = einc*(pscale + (0.4/einc)*depthScale(position.getRho(),151.,22.)
					 +(fTanh(einc,Gflash::emscale[2]) + tscale*depthScale(position.getRho(),151.,22.) )*r1);
      energyMeanHcal  = (fTanh(einc,Gflash::hadscale[0]) +
			 (0.8297+0.2359*tanh(-0.8*(log(einc)-4.0)))*depthScale(position.getRho(),Gflash::RFrontCrystalEB,Gflash::LengthCrystalEB));
      energySigmaHcal = (fTanh(einc,Gflash::hadscale[2]) +
			 fTanh(einc,Gflash::hadscale[3])*depthScale(position.getRho(),Gflash::RFrontCrystalEB,Gflash::LengthCrystalEB));
      //Hcal energy dependent scale
      energyMeanHcal *= 1.+(-0.015+0.015*tanh(6.0*(energyScale[Gflash::kESPM]/einc-0.1)))*(0.5+0.5*tanh(0.025*(einc-150.0)));
      energySigmaHcal *= (1.05-0.4*tanh(0.010*(einc-80.0)));
      
      r2 = CLHEP::RandGaussQ::shoot();
      energyScale[Gflash::kHB] =
	exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 ))-0.05*einc;
    }
    else {
      energyScale[Gflash::kENCA] = einc*(pscale + (0.4/einc)*depthScale(std::fabs(position.getZ()),338.0,21.)
					 +(fTanh(einc,Gflash::emscale[2]) + tscale*depthScale(std::fabs(position.getZ()),338.0,21.) )*r1);
      //@@@extend depthScale for HE
      energyMeanHcal  = (fTanh(einc,Gflash::hadscale[0]) +
			 (0.8297+0.2359*tanh(-0.8*(log(einc)-4.0)))*depthScale(std::fabs(position.getZ()),Gflash::ZFrontCrystalEE,Gflash::LengthCrystalEE));
      energySigmaHcal = (fTanh(einc,Gflash::hadscale[2]) +
			 fTanh(einc,Gflash::hadscale[3])*depthScale(std::fabs(position.getZ()),Gflash::ZFrontCrystalEE,Gflash::LengthCrystalEE));
      r2 = CLHEP::RandGaussQ::shoot();
      energyScale[Gflash::kHE] =
	exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 ))-0.05*einc;
    }
  }
  else if(showerType == 2 || showerType == 6 || showerType == 3 || showerType == 7) { 
    //Hcal response for mip-like pions (mip)
    double gap_corr = 1.0;
    
    energyMeanHcal  = fTanh(einc,Gflash::protonscale[0]);
    energySigmaHcal = fTanh(einc,Gflash::protonscale[1]);
    gap_corr = fTanh(einc,Gflash::protonscale[2]);
         
    if(showerType == 2 ) {
      energyScale[Gflash::kHB] = 
	exp(energyMeanHcal+1.15*energySigmaHcal*CLHEP::RandGaussQ::shoot())-2.0
	- gap_corr*einc*depthScale(position.getRho(),Gflash::Rmin[Gflash::kHB],28.);
    }
    else if(showerType == 6) {
      energyScale[Gflash::kHE] = 
	exp(energyMeanHcal+1.15*energySigmaHcal*CLHEP::RandGaussQ::shoot())-2.0
	- gap_corr*einc*depthScale(std::fabs(position.getZ()),Gflash::Zmin[Gflash::kHB],60.);
    }
    else {
      energyScale[Gflash::kHB] = 
	exp(energyMeanHcal+energySigmaHcal*CLHEP::RandGaussQ::shoot())-2.0;
      energyScale[Gflash::kHE] = energyScale[Gflash::kHB];
    }
  }

  // parameters for the longitudinal profiles
  //@@@check longitudinal profiles of endcaps for possible variations

  double *rhoHcal = new double [2*Gflash::NPar];
  double *correlationVectorHcal = new double [Gflash::NPar*(Gflash::NPar+1)/2];

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
    rhoHcal[i] = fTanh(einc,Gflash::rho[i + showerType*2*Gflash::NPar]);
  }

  getFluctuationVector(rhoHcal,correlationVectorHcal);

  double normalZ[Gflash::NPar];
  for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = CLHEP::RandGaussQ::shoot();
  
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

  for (int i = 0 ; i < Gflash::Nrpar ; i++) {
    lateralPar[Gflash::kHB][i] = fLnE1(einc,Gflash::rpar[i+showerType*Gflash::Nrpar]);

    if(showerType==3 && i == 0) lateralPar[Gflash::kHB][i] *= 1.1;
    lateralPar[Gflash::kHE][i] = lateralPar[Gflash::kHB][i];

  }

  //Ecal parameters are needed if and only if the shower starts inside the crystal

  if(showerType == 1) {
    //A depth dependent correction for the core term of R in Hcal is the linear in 
    //the shower start point while for the spread term is nearly constant

    if(!isEndcap) lateralPar[Gflash::kHB][0] -= 2.3562e-01*(position.getRho()-131.0); 
    else  lateralPar[Gflash::kHE][0] -= 2.3562e-01*(std::abs(position.getZ())-332.0);

    double *rhoEcal = new double [2*Gflash::NPar];
    double *correlationVectorEcal = new double [Gflash::NPar*(Gflash::NPar+1)/2];
    for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) rhoEcal[i] = fTanh(einc,Gflash::rho[i]);

    getFluctuationVector(rhoEcal,correlationVectorEcal);

    for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = CLHEP::RandGaussQ::shoot();
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

    for (int i = 0 ; i < Gflash::Nrpar ; i++) {
      lateralPar[Gflash::kESPM][i] = fLnE1(einc,Gflash::rpar[i]);
      lateralPar[Gflash::kENCA][i] = lateralPar[Gflash::kESPM][i];
    }
  }

}
