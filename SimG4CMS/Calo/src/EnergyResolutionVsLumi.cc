#include <string>
#include <vector>
#include "SimG4CMS/Calo/interface/EvolutionECAL.h"
#include "SimG4CMS/Calo/interface/EnergyResolutionVsLumi.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


EnergyResolutionVsLumi::EnergyResolutionVsLumi()
{
  m_lumi=0;
  m_instlumi=0;
  
}

EnergyResolutionVsLumi::~EnergyResolutionVsLumi()
{
}


EnergyResolutionVsLumi::DegradationAtEta EnergyResolutionVsLumi::CalculateDegradation(double eta)
{
  DegradationAtEta result;

  result.eta = eta;
  double totLumi = m_lumi;
  double instLumi = m_instlumi;

  EvolutionECAL model;

  // Index of induced absorption due to EM damages in PWO4
  result.muEM = model.InducedAbsorptionEM(instLumi, eta);
  
  // Index of induced absorption due to hadron damages in PWO4
  result.muHD = model.InducedAbsorptionHadronic(totLumi, eta);

  // Average degradation of amplitude due to transparency change
  result.ampDropTransparency = model.DegradationMeanEM50GeV(result.muEM+result.muHD);

  // Average degradation of amplitude due to photo-detector aging
  result.ampDropPhotoDetector = model.AgingVPT(instLumi, totLumi, eta);

  result.ampDropTotal = result.ampDropTransparency*result.ampDropPhotoDetector;

  // Noise increase in ADC counts due to photo-detector and front-end
  result.noiseIncreaseADC =  model.NoiseFactorFE(totLumi, eta);

  // Resolution degradation due to LY non-uniformity caused by transparency loss
  result.resolutitonConstantTerm =  model.ResolutionConstantTermEM50GeV(result.muEM+result.muHD);

  return result;
}


double EnergyResolutionVsLumi::calcmuEM(double eta)
{
  double instLumi = m_instlumi;
  EvolutionECAL model;
  double result = model.InducedAbsorptionEM(instLumi, eta);
  return result;
}

double EnergyResolutionVsLumi::calcmuHD(double eta)
{
  double totLumi = m_lumi;
  EvolutionECAL model;
  double result = model.InducedAbsorptionHadronic(totLumi, eta);
  return result;
}


void  EnergyResolutionVsLumi::calcmuTot(){

  for(int iEta=1; iEta<=EBDetId::MAX_IETA ;++iEta) {
    if(iEta==0) continue;

      double eta=EBDetId::approxEta(EBDetId(iEta,1));
      eta = fabs(eta);
      double r= calcmuTot(eta);
      
      mu_eta[iEta]=r;
      vpt_eta[iEta]=1.0;

  }

  for(int iX=EEDetId::IX_MIN; iX<=EEDetId::IX_MAX ;++iX) {
    for(int iY=EEDetId::IY_MIN; iY<=EEDetId::IY_MAX; ++iY) {
      if (EEDetId::validDetId(iX,iY,1))
	{
	  EEDetId eedetidpos(iX,iY,1);
	  double eta= -log(tan(0.5*atan(sqrt((iX-50.0)*(iX-50.0)+(iY-50.0)*(iY-50.0))*2.98/328.)));
          eta = fabs(eta);
          double r=calcmuTot(eta);
          double v=calcampDropPhotoDetector(eta);

	  mu_eta[EBDetId::MAX_IETA+iX+iY*(EEDetId::IX_MAX)]=r;
	  vpt_eta[EBDetId::MAX_IETA+iX+iY*(EEDetId::IX_MAX)]=v;
	  //std::cout<<"eta/mu/vpt"<<eta<<"/"<<r<<"/"<<v<<std::endl;
	}
    }
  }

}


double EnergyResolutionVsLumi::calcLightCollectionEfficiencyWeighted(DetId id, double z)
{

  double v=1.0;
  double muTot=0;
  if(id.subdetId()==EcalBarrel) {
    EBDetId ebId(id);
    int ieta= fabs(ebId.ieta());
    muTot= mu_eta[ieta];
    
  } else if(id.subdetId()==EcalEndcap){
    EEDetId eeId(id);
    int ix= eeId.ix();
    int iy= eeId.iy();
    
    muTot= mu_eta[EBDetId::MAX_IETA+ix+iy*(EEDetId::IX_MAX)];
    v=vpt_eta[EBDetId::MAX_IETA+ix+iy*(EEDetId::IX_MAX)];
  } else {
    muTot=0;
  }
  double zcor=z;
  EvolutionECAL model;
  if(z<0.02 ) zcor=0.02;
  if(z>0.98) zcor=0.98;

  double result=model.LightCollectionEfficiencyWeighted( zcor , muTot)*v;
  
  
  
  return result; 

}



double EnergyResolutionVsLumi::calcLightCollectionEfficiencyWeighted2(double eta, double z, double mu_ind)
{
  if(mu_ind<0) mu_ind=this->calcmuTot(eta);
  double v= this->calcampDropPhotoDetector(eta);
  EvolutionECAL model;
  double result=model.LightCollectionEfficiencyWeighted( z , mu_ind)*v;
  return result; 
}


double EnergyResolutionVsLumi::calcmuTot(double eta)
{
  double totLumi = m_lumi;
  double instLumi = m_instlumi;
  EvolutionECAL model;
  double muEM = model.InducedAbsorptionEM(instLumi, eta);
  double muH = model.InducedAbsorptionHadronic(totLumi, eta);
  double result=muEM+muH;
  return result;
} 

double EnergyResolutionVsLumi::calcampDropTransparency(double eta)
{
  double muEM=this->calcmuEM(eta);
  double muHD=this->calcmuHD(eta);
  EvolutionECAL model;
  double result = model.DegradationMeanEM50GeV(muEM+muHD);
  return result;
}

double EnergyResolutionVsLumi::calcampDropPhotoDetector(double eta)
{
  double instLumi = m_instlumi;
  double totLumi = m_lumi;
  EvolutionECAL model;
  double result = model.AgingVPT(instLumi, totLumi, eta);
  return result;
}

double EnergyResolutionVsLumi::calcampDropTotal(double eta)
{
  double tra= this->calcampDropTransparency(eta);
  double pho= this->calcampDropPhotoDetector(eta);
  double result = tra*pho;
  return result;
}

double EnergyResolutionVsLumi::calcnoiseIncreaseADC(double eta)
{
  double totLumi = m_lumi;
  EvolutionECAL model;
  double result = model.NoiseFactorFE(totLumi, eta);
  return result;
  // noise increase in ADC
}

double EnergyResolutionVsLumi::calcnoiseADC(double eta)
{
  double totLumi = m_lumi;
  double Nadc = 1.1;
  double result =1.0;
  EvolutionECAL model;
  if(fabs(eta)<1.497){
    Nadc = 1.1;
    result = model.NoiseFactorFE(totLumi, eta)*Nadc;
  }else{
    Nadc = 2.0;
    result=Nadc;
    // endcaps no increase in ADC
  }
  return result;

}

double EnergyResolutionVsLumi::calcresolutitonConstantTerm(double eta)
{
  double muEM=this->calcmuEM(eta);
  double muHD=this->calcmuHD(eta);
  EvolutionECAL model;
  double result = model.ResolutionConstantTermEM50GeV(muEM+muHD);
  return result;
}


double  EnergyResolutionVsLumi::Resolution(double eta, double ene)
{  


  // Initial parameters for ECAL resolution
  double S;
  double Nadc;
  double adc2GeV;
  double C;
  if(eta<1.497){
    S = 0.028;           // CMS note 2006/148 (EB test beam)
    Nadc = 1.1; 
    adc2GeV = 0.039;
    C = 0.003;
  }else{
    S = 0.052;          //  CMS DN 2009/002
    Nadc = 2.0;
    adc2GeV = 0.069;
    C = 0.004;
  }


  DegradationAtEta d = CalculateDegradation(eta);

  // adjust resolution parameters
  S /= sqrt(d.ampDropTotal);
  Nadc *= d.noiseIncreaseADC;
  adc2GeV /= d.ampDropTotal;
  double N = Nadc*adc2GeV*3.0;   // 3x3 noise in GeV 
  C = sqrt(C*C + d.resolutitonConstantTerm*d.resolutitonConstantTerm);

  return sqrt(S*S/ene + N*N/ene/ene + C*C);

}







 void EnergyResolutionVsLumi::Decomposition()
{

  double eta = 2.2;
  m_instlumi = 5.0e+34;
  m_lumi = 3000.0;

  DegradationAtEta d = this->CalculateDegradation(eta);


  // Initial parameters for ECAL resolution
  double S;
  double Nadc;
  double adc2GeV;
  double C;
  if(eta<1.497){
    S = 0.028;           // CMS note 2006/148 (EB test beam)
    Nadc = 1.1; 
    adc2GeV = 0.039;
    C = 0.003;
  }else{
    S = 0.052;          //  CMS DN 2009/002
    Nadc = 2.0;
    adc2GeV = 0.069;
    C = 0.0038;
  }


  // adjust resolution parameters
  S /= sqrt(d.ampDropTotal);
  Nadc *= d.noiseIncreaseADC;
  adc2GeV /= d.ampDropTotal;
  //  double N = Nadc*adc2GeV*3.0;   // 3x3 noise in GeV 
  C = sqrt(C*C + d.resolutitonConstantTerm*d.resolutitonConstantTerm);



  /*  for(double ene=1.0; ene<1e+3; ene*=1.1){


    // this is the resolution

    double res =  sqrt(S*S/ene + N*N/ene/ene + C*C);

    double factor = 1.0;
    factor = sin(2.0*atan(exp(-1.0*eta)));
  

  }

  */


}





