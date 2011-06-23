#include "SimGeneral/GFlash/interface/GflashAntiProtonShowerProfile.h"
#include <CLHEP/Random/RandGaussQ.h>

void GflashAntiProtonShowerProfile::loadParameters()
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

    //energy dependent energyRho based on tuning with testbeam data
    double energyRho =  fTanh(einc,Gflash::correl_hadem);

    r1 = CLHEP::RandGaussQ::shoot();

    if(showerType == 0 || showerType == 1) {
      energyScale[Gflash::kESPM] = einc*(fTanh(einc,Gflash::pbar_emscale[0]) + fTanh(einc,Gflash::emcorr_pid[6]) +
					 fTanh(einc,Gflash::pbar_emscale[1])*r1);
      
      r2 = CLHEP::RandGaussQ::shoot();
      energyMeanHcal  = (fTanh(einc,Gflash::pbar_hadscale[0]) +
			 fTanh(einc,Gflash::pbar_hadscale[1])*depthScale(position.getRho(),Gflash::RFrontCrystalEB,Gflash::LengthCrystalEB));
      energySigmaHcal = (fTanh(einc,Gflash::pbar_hadscale[2]) +
			 fTanh(einc,Gflash::pbar_hadscale[3])*depthScale(position.getRho(),Gflash::RFrontCrystalEB,Gflash::LengthCrystalEB));
      energyScale[Gflash::kHB] = std::max(0.0,
	einc*fTanh(einc,Gflash::hadcorr_pid[6]) +
	exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 )));
    }
    else {
      energyScale[Gflash::kENCA] = einc*(fTanh(einc,Gflash::pbar_emscale[0]) + fTanh(einc,Gflash::emcorr_pid[7]) +
					 fTanh(einc,Gflash::pbar_emscale[1])*r1);
      
      r2 = CLHEP::RandGaussQ::shoot();
      energyMeanHcal  = (fTanh(einc,Gflash::pbar_hadscale[0]) +
			 fTanh(einc,Gflash::pbar_hadscale[1])*depthScale(std::fabs(position.getZ()),Gflash::ZFrontCrystalEE,Gflash::LengthCrystalEE));
      energySigmaHcal = (fTanh(einc,Gflash::pbar_hadscale[2]) +
			 fTanh(einc,Gflash::pbar_hadscale[3])*depthScale(std::fabs(position.getZ()),Gflash::ZFrontCrystalEE,Gflash::LengthCrystalEE));
      
      energyScale[Gflash::kHE] =  std::max(0.0,
	einc*fTanh(einc,Gflash::hadcorr_pid[7]) +
	exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 )));
    }
  }
  else if(showerType == 2 || showerType == 6 || showerType == 3 || showerType == 7) { 
    //Hcal response for mip-like pions (mip)
    
    energyMeanHcal  = fTanh(einc,Gflash::pbar_hadscale[4]);
    energySigmaHcal = fTanh(einc,Gflash::pbar_hadscale[5]);
    double gap_corr = fTanh(einc,Gflash::pbar_hadscale[6]);

    if(showerType == 2 || showerType == 3) {
      energyScale[Gflash::kHB] =  std::max(0.0, einc*fTanh(einc,Gflash::mipcorr_pid[6]) +
					   exp(energyMeanHcal+energySigmaHcal*CLHEP::RandGaussQ::shoot()));
      
      if(showerType == 2) {
	energyScale[Gflash::kHB] = std::max(0.0,
	     energyScale[Gflash::kHB]*(1.0- gap_corr*depthScale(position.getRho(),Gflash::Rmin[Gflash::kHB],28.)));
      }
    }
    else if(showerType == 6 || showerType == 7) {
      energyScale[Gflash::kHE] =  std::max(0.0, einc*(fTanh(einc,Gflash::mipcorr_pid[7])) +
					   exp(energyMeanHcal+energySigmaHcal*CLHEP::RandGaussQ::shoot()));
      if(showerType == 6) {
	energyScale[Gflash::kHE] = std::max(0.0,
	  energyScale[Gflash::kHE]*(1.0- gap_corr*depthScale(std::fabs(position.getZ()),Gflash::Zmin[Gflash::kHE],60.)));
      }
    }
  }

  // parameters for the longitudinal profiles

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
    rhoHcal[i] = fTanh(einc,Gflash::pbar_rho[i + showerType*2*Gflash::NPar]);
  }

  getFluctuationVector(rhoHcal,correlationVectorHcal);

  double normalZ[Gflash::NPar];
  for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = CLHEP::RandGaussQ::shoot();
  
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

  for (int i = 0 ; i < Gflash::Nrpar ; i++) {
    lateralPar[Gflash::kHB][i] = fLnE1(einc,Gflash::pbar_rpar[i+showerType*Gflash::Nrpar]);
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
    for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) rhoEcal[i] = fTanh(einc,Gflash::pbar_rho[i]);

    getFluctuationVector(rhoEcal,correlationVectorEcal);

    for(int i = 0 ; i < Gflash::NPar ; i++) normalZ[i] = CLHEP::RandGaussQ::shoot();
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
