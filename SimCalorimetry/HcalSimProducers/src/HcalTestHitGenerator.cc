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
     noiseHits.emplace_back(detId.rawId(), e/theBarrelSampling[i-1], hbTof, 0., 0);
  }

  // ring 16 is special
  HcalDetId detId(HcalEndcap, 16, 1, 3);
  noiseHits.emplace_back(detId.rawId(), e/theEndcapSampling[0], heTof, 0., 0);

  for(int i = 17; i <= 29; ++i)
  {
     HcalDetId detId(HcalEndcap, i, 1, 1);
     noiseHits.emplace_back(detId.rawId(), e/theEndcapSampling[i-16], heTof, 0., 0);
  }

  HcalDetId outerDetId(HcalOuter, 1, 1, 4);
  noiseHits.emplace_back(outerDetId.rawId(), 0.45, hoTof, 0., 0);

  HcalDetId forwardDetId1(HcalForward, 30, 1, 1);
  noiseHits.emplace_back(forwardDetId1.rawId(), 35., hfTof, 0., 0);

  HcalDetId forwardDetId2(HcalForward, 30, 1, 2);
  noiseHits.emplace_back(forwardDetId2.rawId(), 48., hfTof, 0., 0);

  //HcalZDCDetId zdcDetId(HcalZDCDetId::Section(2),true,1);
  //noiseHits.emplace_back(zdcDetId.rawId(), 50.0, 0.);

}


