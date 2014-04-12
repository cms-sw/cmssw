#include "SimGeneral/GFlash/interface/GflashPiKShowerProfile.h"
#include <CLHEP/Random/RandGaussQ.h>

void GflashPiKShowerProfile::loadParameters()
{
  double einc = theShowino->getEnergy();
  Gflash3Vector position = theShowino->getPositionAtShower();
  int showerType = theShowino->getShowerType();

  // energy scale
  double energyMeanHcal = 0.0;
  double energySigmaHcal = 0.0;
  
  double r1 = 0.0;
  double r2 = 0.0;

  if(showerType == 0 || showerType == 1 || showerType == 4 || showerType == 5) {

    //@@@ energy dependent energyRho based on tuning with testbeam data
    double energyRho =  fTanh(einc,Gflash::pion_correl_hadem); 

    if(showerType == 0 || showerType == 1) {
      do {
	r1 = CLHEP::RandGaussQ::shoot();

	energyScale[Gflash::kESPM] = einc*(fTanh(einc,Gflash::pion_emscale[0]) + fTanh(einc,Gflash::pion_emscale[1])*r1);

	//LogNormal mean and sigma of Hcal energy
	energyMeanHcal  = (fTanh(einc,Gflash::pion_hadscale[0]) +
			   fTanh(einc,Gflash::pion_hadscale[1])*depthScale(position.getRho(),Gflash::RFrontCrystalEB,Gflash::LengthCrystalEB));
	energySigmaHcal = (fTanh(einc,Gflash::pion_hadscale[2]) +  
			   fTanh(einc,Gflash::pion_hadscale[3])*depthScale(position.getRho(),Gflash::RFrontCrystalEB,Gflash::LengthCrystalEB));
         
	r2 = CLHEP::RandGaussQ::shoot();
	energyScale[Gflash::kHB] = exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 ));
      } while ( energyScale[Gflash::kESPM] < 0 || energyScale[Gflash::kHB] > einc*1.5 ); 
    }
    else {
      do {
	r1 = CLHEP::RandGaussQ::shoot();
	energyScale[Gflash::kENCA] = einc*(fTanh(einc,Gflash::pion_emscale[0]) + fTanh(einc,Gflash::pion_emscale[1])*r1);
	
	//@@@extend depthScale for HE
	energyMeanHcal  = (fTanh(einc,Gflash::pion_hadscale[0]) + 
			   fTanh(einc,Gflash::pion_hadscale[1])*depthScale(std::fabs(position.getZ()),Gflash::ZFrontCrystalEE,Gflash::LengthCrystalEE));
	energySigmaHcal = (fTanh(einc,Gflash::pion_hadscale[2]) +
			   fTanh(einc,Gflash::pion_hadscale[3])*depthScale(std::fabs(position.getZ()),Gflash::ZFrontCrystalEE,Gflash::LengthCrystalEE));
	r2 = CLHEP::RandGaussQ::shoot();
	energyScale[Gflash::kHE] = exp(energyMeanHcal+energySigmaHcal*(energyRho*r1 + sqrt(1.0- energyRho*energyRho)*r2 ));
      } while ( energyScale[Gflash::kENCA] < 0 || energyScale[Gflash::kHE] > einc*1.5 ); 
    }
  }
  else if(showerType == 2 || showerType == 3 || showerType == 6 || showerType == 7) { 
    //Hcal response for mip-like pions (mip)
    
    energyMeanHcal  = fTanh(einc,Gflash::pion_hadscale[4]);
    energySigmaHcal = fTanh(einc,Gflash::pion_hadscale[5]);

    double gap_corr = einc*fTanh(einc,Gflash::pion_hadscale[6]);

    if(showerType == 2 || showerType == 3) {
      energyScale[Gflash::kESPM] = 0.0;

      do {
	r1 = CLHEP::RandGaussQ::shoot();
	energyScale[Gflash::kHB] = exp(energyMeanHcal+energySigmaHcal*r1);
      } while ( energyScale[Gflash::kHB] > einc*1.5 );

      if(showerType == 2) {
        energyScale[Gflash::kHE] = std::max(0.0,energyScale[Gflash::kHB]
                                 - gap_corr*depthScale(position.getRho(),Gflash::Rmin[Gflash::kHB],28.));
      }
    }
    else if(showerType == 6 || showerType == 7 ) {
      energyScale[Gflash::kENCA] = 0.0;

      do {
	r1 = CLHEP::RandGaussQ::shoot();
	energyMeanHcal +=  std::log(1.0-fTanh(einc,Gflash::pion_hadscale[7]));
        energyScale[Gflash::kHE] = exp(energyMeanHcal+energySigmaHcal*r1);
      } while ( energyScale[Gflash::kHE] > einc*1.5 );

      if(showerType == 6) {
        energyScale[Gflash::kHE] = std::max(0.0,energyScale[Gflash::kHE]
                                 - gap_corr*depthScale(std::fabs(position.getZ()),Gflash::Zmin[Gflash::kHE],66.));
      }
    }
  }

  // parameters for the longitudinal profiles
  //@@@check longitudinal profiles of endcaps for possible variations

  double *rhoHcal = new double [2*Gflash::NPar];
  double *correlationVectorHcal = new double [Gflash::NPar*(Gflash::NPar+1)/2];

  //@@@until we have a separate parameterization for Endcap 

  if(showerType>3) {
    showerType -= 4;
  }
  //no separate parameterization before crystal
  if(showerType==0) showerType = 1; 

  //Hcal parameters are always needed regardless of showerType
  for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) {
    rhoHcal[i] = fTanh(einc,Gflash::pion_rho[i + showerType*2*Gflash::NPar]);
  }

  getFluctuationVector(rhoHcal,correlationVectorHcal);

  double normalZ[Gflash::NPar];
  for (int i = 0; i < Gflash::NPar ; i++) normalZ[i] = CLHEP::RandGaussQ::shoot();
  
  for(int i = 0 ; i < Gflash::NPar ; i++) {
    double correlationSum = 0.0;

    for(int j = 0 ; j < i+1 ; j++) {
      correlationSum += correlationVectorHcal[i*(i+1)/2+j]*normalZ[j];
    }
    longHcal[i] = fTanh(einc,Gflash::pion_par[i+showerType*Gflash::NPar]) +
		  fTanh(einc,Gflash::pion_par[i+(4+showerType)*Gflash::NPar])*correlationSum;
  }
  delete [] rhoHcal;
  delete [] correlationVectorHcal;

  // lateral parameters for Hcal

  for (int i = 0 ; i < Gflash::Nrpar ; i++) {
    lateralPar[Gflash::kHB][i] = fTanh(einc,Gflash::pion_rpar[i+showerType*Gflash::Nrpar]);
    lateralPar[Gflash::kHE][i] = lateralPar[Gflash::kHB][i];
  }

  //Ecal parameters are needed if and only if the shower starts inside the crystal

  if(showerType == 1) {

    double *rhoEcal = new double [2*Gflash::NPar];
    double *correlationVectorEcal = new double [Gflash::NPar*(Gflash::NPar+1)/2];
    for(int i = 0 ; i < 2*Gflash::NPar ; i++ ) rhoEcal[i] = fTanh(einc,Gflash::pion_rho[i]);

    getFluctuationVector(rhoEcal,correlationVectorEcal);

    for(int i = 0 ; i < Gflash::NPar ; i++) normalZ[i] = CLHEP::RandGaussQ::shoot();
    for(int i = 0 ; i < Gflash::NPar ; i++) {
      double correlationSum = 0.0;

      for(int j = 0 ; j < i+1 ; j++) {
	correlationSum += correlationVectorEcal[i*(i+1)/2+j]*normalZ[j];
      }
      longEcal[i] = fTanh(einc,Gflash::pion_par[i]) +
   	            0.5*fTanh(einc,Gflash::pion_par[i+4*Gflash::NPar])*correlationSum;
    }

    delete [] rhoEcal;
    delete [] correlationVectorEcal;

    // lateral parameters for Ecal

    for (int i = 0 ; i < Gflash::Nrpar ; i++) {
      lateralPar[Gflash::kESPM][i] = fTanh(einc,Gflash::pion_rpar[i]);
      lateralPar[Gflash::kENCA][i] = lateralPar[Gflash::kESPM][i];
    }
  }

}
