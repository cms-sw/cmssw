#ifndef HcalTrigPrimFormatter_h
#define HcalTrigPrimFormatter_h


class HcalTrigPrimFormatter {
public:
  HcalTrigPrimFormatter();

  void run(const HBHEDigiCollection & hbheDigis,
           const HODigiCollection & hoDigis,
           const HFDigiCollection & hfDigis,
           auto_ptr<HcalTrigPrimDigiCollection> result);

private:
  HcalCoder theCoder;
};

#endif

