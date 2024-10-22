#ifndef HCALTBBEAMCOUNTERS_H
#define HCALTBBEAMCOUNTERS_H 1

#include <string>
#include <iostream>
#include <vector>

class HcalTBBeamCounters {
public:
  HcalTBBeamCounters();

  // Getter methods

  /// Muon Veto adc
  double VMadc() const { return VMadc_; }
  double V3adc() const { return V3adc_; }
  double V6adc() const { return V6adc_; }
  double VH1adc() const { return VH1adc_; }
  double VH2adc() const { return VH2adc_; }
  double VH3adc() const { return VH3adc_; }
  double VH4adc() const { return VH4adc_; }
  double Ecal7x7() const { return Ecal7x7_; }
  double CK1adc() const { return CK1adc_; }
  double CK2adc() const { return CK2adc_; }
  double CK3adc() const { return CK3adc_; }
  double SciVLEadc() const { return SciVLEadc_; }
  double Sci521adc() const { return Sci521adc_; }
  double Sci528adc() const { return Sci528adc_; }
  double S1adc() const { return S1adc_; }
  double S2adc() const { return S2adc_; }
  double S3adc() const { return S3adc_; }
  double S4adc() const { return S4adc_; }
  double VMFadc() const { return VMFadc_; }
  double VMBadc() const { return VMBadc_; }
  double VM1adc() const { return VM1adc_; }
  double VM2adc() const { return VM2adc_; }
  double VM3adc() const { return VM3adc_; }
  double VM4adc() const { return VM4adc_; }
  double VM5adc() const { return VM5adc_; }
  double VM6adc() const { return VM6adc_; }
  double VM7adc() const { return VM7adc_; }
  double VM8adc() const { return VM8adc_; }
  double TOF1Sadc() const { return TOF1Sadc_; }
  double TOF1Jadc() const { return TOF1Jadc_; }
  double TOF2Sadc() const { return TOF2Sadc_; }
  double TOF2Jadc() const { return TOF2Jadc_; }
  double BH1adc() const { return BH1adc_; }
  double BH2adc() const { return BH2adc_; }
  double BH3adc() const { return BH3adc_; }
  double BH4adc() const { return BH4adc_; }

  // Setter methods
  void setADCs04(double VMadc,
                 double V3adc,
                 double V6adc,
                 double VH1adc,
                 double VH2adc,
                 double VH3adc,
                 double VH4adc,
                 double CK2adc,
                 double CK3adc,
                 double SciVLEadc,
                 double Sci521adc,
                 double Sci528adc,
                 double S1adc,
                 double S2adc,
                 double S3adc,
                 double S4adc,
                 double Ecal7x7);
  void setADCs06(double VMFadc,
                 double VMBadc,
                 double VM1adc,
                 double VM2adc,
                 double VM3adc,
                 double VM4adc,
                 double VM5adc,
                 double VM6adc,
                 double VM7adc,
                 double VM8adc,
                 double CK1adc,
                 double CK2adc,
                 double CK3adc,
                 double S1adc,
                 double S2adc,
                 double S3adc,
                 double S4adc,
                 double TOF1Sadc,
                 double TOF1Jadc,
                 double TOF2Sadc,
                 double TOF2Jadc,
                 double Sci521adc,
                 double Sci528adc,
                 double BH1adc,
                 double BH2adc,
                 double BH3adc,
                 double BH4adc);

private:
  //    TB2004 specific
  double VMadc_;      // behind HO
  double V3adc_;      // behind HB at (eta,phi)=(7,3)
  double V6adc_;      // behind HB at (eta,phi)=(7,6)
  double VH1adc_;     // part of extra muon veto wall - the end of TB04 data taking
  double VH2adc_;     // part of extra muon veto wall - the end of TB04 data taking
  double VH3adc_;     // part of extra muon veto wall - the end of TB04 data taking
  double VH4adc_;     // part of extra muon veto wall - the end of TB04 data taking
  double Ecal7x7_;    // Ecal energy - sum of all 49 crustals
                      //   Common for TB2004 and TB2006
  double Sci521adc_;  // Scintilator at 521m (see beam line drawings)
  double Sci528adc_;  // Scintilator at 522m (see beam line drawings)
  double CK1adc_;     // TB2006
  double CK2adc_;     // Cerenkov 2 : electron id
  double CK3adc_;     // Cerenkov 3 : pi/proton separation
  double SciVLEadc_;  // Scintillator in VLE beam line
  double S1adc_;      // Trigger scintilator 14x14 cm
  double S2adc_;      // Trigger scintilator 4x4 cm
  double S3adc_;      // Trigger scintilator 2x2 cm
  double S4adc_;      // Trigger scintilator 14x14 cm
                      //    TB2006 specific
  double VMFadc_;     // VM front
  double VMBadc_;     // VM back
  double VM1adc_;     // Muon veto wall
  double VM2adc_;     // Muon veto wall
  double VM3adc_;     // Muon veto wall
  double VM4adc_;     // Muon veto wall
  double VM5adc_;     // Muon veto wall
  double VM6adc_;     // Muon veto wall
  double VM7adc_;     // Muon veto wall
  double VM8adc_;     // Muon veto wall
  double TOF1Sadc_;   // TOF1S (Saleve side)
  double TOF1Jadc_;   // TOF1J (Jura side)
  double TOF2Sadc_;   // TOF2S (Saleve side)
  double TOF2Jadc_;   // TOF2J (Jura side)
  double BH1adc_;     // beam halo up
  double BH2adc_;     // beam halo left from particle view
  double BH3adc_;     // beam halo right from particle view
  double BH4adc_;     // beam halo down
};

std::ostream& operator<<(std::ostream& s, const HcalTBBeamCounters& htbcnt);

#endif
