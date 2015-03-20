#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseMaker.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"

int main () {

  HPDNoiseMaker maker ("hpdNoiseLibrary.root");

  maker.addHpd ("HPD01");
  maker.addHpd ("HPD02");
  maker.addHpd ("HPD03");

  HcalDetId id;
  float data[10];

  HPDNoiseData event;
  for (size_t i = 0; i < 10; i++) data[i] = i;
  id = HcalDetId (HcalBarrel, 1, 1, 1);
  event.addChannel (id, data);
  id = HcalDetId (HcalBarrel, 1, 2, 1);
  event.addChannel (id, data);
  id = HcalDetId (HcalBarrel, 1, 3, 1);
  event.addChannel (id, data);
  
  maker.newHpdEvent ("HPD01", event);

  event.clear ();
  for (size_t i = 0; i < 10; i++) data[i] = i*10;
  id = HcalDetId (HcalBarrel, 2, 1, 1);
  event.addChannel (id, data);
  id = HcalDetId (HcalBarrel, 2, 2, 1);
  event.addChannel (id, data);
  id = HcalDetId (HcalBarrel, 2, 3, 1);
  event.addChannel (id, data);
  
  maker.newHpdEvent ("HPD02", event);
  maker.newHpdEvent ("HPD02", event);

  event.clear ();
  for (size_t i = 0; i < 10; i++) data[i] = i*100;
  id = HcalDetId (HcalBarrel, 3, 1, 1);
  event.addChannel (id, data);
  id = HcalDetId (HcalBarrel, 3, 2, 1);
  event.addChannel (id, data);
  id = HcalDetId (HcalBarrel, 3, 3, 1);
  event.addChannel (id, data);
  
  maker.newHpdEvent ("HPD03", event);
  maker.newHpdEvent ("HPD03", event);
  maker.newHpdEvent ("HPD03", event);

  maker.setRate ("HPD01", 0.01, 0.01, 0.01, 0.01);
  maker.setRate ("HPD02", 0.02, 0.01, 0.01, 0.01);
  maker.setRate ("HPD03", 0.03, 0.01, 0.01, 0.01);

  std::cout << "HPD01: total entries: " << maker.totalEntries ("HPD01") << std::endl;
  std::cout << "HPD02: total entries: " << maker.totalEntries ("HPD02") << std::endl;
  std::cout << "HPD03: total entries: " << maker.totalEntries ("HPD03") << std::endl;


  return 0;
}
