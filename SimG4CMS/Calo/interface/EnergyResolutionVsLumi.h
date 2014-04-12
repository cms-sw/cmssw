#ifndef CalibCalorimetry_EnergyResolutionVsLumi_H
#define CalibCalorimetry_EnergyResolutionVsLumi_H
// system include files

#include <vector>
#include <typeinfo>
#include <string>
#include <map>

#include <time.h>
#include <stdio.h>

#include <math.h>

#include "SimG4CMS/Calo/interface/EvolutionECAL.h"
#include "DataFormats/DetId/interface/DetId.h"

class EnergyResolutionVsLumi {

 public:
  
  EnergyResolutionVsLumi();

  EnergyResolutionVsLumi(double lumi, double instlumi){
    m_lumi=lumi;
    m_instlumi=instlumi;
    calcmuTot();
  };


  virtual ~EnergyResolutionVsLumi();
  
	  
  struct DegradationAtEta{
    double eta;
    double muEM;
    double muHD;
    double ampDropTransparency;
    double ampDropPhotoDetector;
    double ampDropTotal;
    double noiseIncreaseADC;
    double resolutitonConstantTerm;
  };
  

  

  DegradationAtEta CalculateDegradation(double eta);
  double  Resolution(double eta, double ene);
  void Decomposition();

  void setLumi(double x){m_lumi=x;};
  void setInstLumi(double x){m_instlumi=x;};
  void setLumies(double x, double y){m_lumi=x, m_instlumi=y, calcmuTot();};
 
  double calcmuEM(double eta);
  double calcmuHD(double eta);
  double calcampDropTransparency(double eta);
  double calcampDropPhotoDetector(double eta);
  double calcampDropTotal(double eta);
  double calcnoiseIncreaseADC(double eta);
  double calcnoiseADC(double eta);
  double calcresolutitonConstantTerm(double eta);
  
  double calcLightCollectionEfficiencyWeighted(DetId id, double z);

  double calcLightCollectionEfficiencyWeighted2(double eta, double z, double mu_ind=-1.0);
  double calcmuTot(double eta);
  void   calcmuTot();
  double getmuTot(double eta, int ix, int iy);


 private:
  double m_lumi;
  double m_instlumi;
  double mu_eta[10085];
  double vpt_eta[10085];


};

#endif
