#include "SimCalorimetry/HcalSimProducers/src/HcalTestHitGenerator.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

HcalTestHitGenerator::HcalTestHitGenerator(const edm::ParameterSet & ps)
:  theBarrelSampling(ps.getParameter<edm::ParameterSet>("hb").getParameter<std::vector<double> >("samplingFactors")),
   theEndcapSampling(ps.getParameter<edm::ParameterSet>("he").getParameter<std::vector<double> >("samplingFactors"))
{
}


void HcalTestHitGenerator::getNoiseHits(std::vector<PCaloHit> & noiseHits)
{
  // just say about a foot a nanosecond, pluis 10 for showers
  double e = 10.;
  double hbTof = 17.;
  double heTof = 23.;
  double hfTof = 43.;
  double hoTof = 22.;
  for(int i = 1; i <= 16; ++i)
  {
     HcalDetId detId(HcalBarrel, i, 1, 1);
     PCaloHit hit(detId.rawId(), e/theBarrelSampling[i-1], hbTof, 0., 0);
     noiseHits.push_back(hit);
  }

  // ring 16 is special
  HcalDetId detId(HcalEndcap, 16, 1, 3);
  PCaloHit hit(detId.rawId(), e/theEndcapSampling[0], heTof, 0., 0);
  noiseHits.push_back(hit);

  for(int i = 17; i <= 29; ++i)
  {
     HcalDetId detId(HcalEndcap, i, 1, 1);
     PCaloHit hit(detId.rawId(), e/theEndcapSampling[i-16], heTof, 0., 0);
     noiseHits.push_back(hit);
  }

  HcalDetId outerDetId(HcalOuter, 1, 1, 4);
  PCaloHit outerHit(outerDetId.rawId(), 0.45, hoTof, 0., 0);

  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  PCaloHit forwardHit1(forwardDetId1.rawId(), 35., hfTof, 0., 0);

  HcalDetId forwardDetId2(HcalForward, 30, 1, 2);
  PCaloHit forwardHit2(forwardDetId2.rawId(), 48., hfTof, 0., 0);

  HcalZDCDetId zdcDetId(HcalZDCDetId::Section(2),true,1);
  PCaloHit zdcHit(zdcDetId.rawId(), 50.0, 0.);

  noiseHits.push_back(outerHit);
  noiseHits.push_back(forwardHit1);
  noiseHits.push_back(forwardHit2);
  //noiseHits.push_back(zdcHit);

}


