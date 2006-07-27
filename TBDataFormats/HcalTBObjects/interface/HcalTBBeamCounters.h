#ifndef HCALTBBEAMCOUNTERS_H
#define HCALTBBEAMCOUNTERS_H 1

#include <string>
#include <iostream>
#include <vector>
#include "boost/cstdint.hpp"

  class HcalTBBeamCounters {
  public:
    HcalTBBeamCounters();

    // Getter methods

    /// Muon Veto adc 
    double VMadc()     const { return VMadc_;     }
    double V3adc()     const { return V3adc_;     }
    double V6adc()     const { return V6adc_;     }
    double VH1adc()     const { return VH1adc_;     }
    double VH2adc()     const { return VH2adc_;     }
    double VH3adc()     const { return VH3adc_;     }
    double VH4adc()     const { return VH4adc_;     }
    double CK2adc()     const { return CK2adc_;     }
    double CK3adc()     const { return CK3adc_;     }
    double SciVLEadc()     const { return SciVLEadc_;     }
    double Sci521adc()     const { return Sci521adc_;     }
    double Sci528adc()     const { return Sci528adc_;     }
    double S1adc()     const { return S1adc_;     }
    double S2adc()     const { return S2adc_;     }
    double S3adc()     const { return S3adc_;     }
    double S4adc()     const { return S4adc_;     }
    double VMFadc()     const { return VMFadc_;     }
    double VMBadc()     const { return VMFadc_;     }
    double VM1adc()     const { return VM1adc_;     }
    double VM2adc()     const { return VM2adc_;     }
    double VM3adc()     const { return VM3adc_;     }
    double VM4adc()     const { return VM4adc_;     }
    double VM5adc()     const { return VM5adc_;     }
    double VM6adc()     const { return VM6adc_;     }
    double VM7adc()     const { return VM7adc_;     }
    double VM8adc()     const { return VM8adc_;     }
    double TOF1adc()     const { return TOF1adc_;     }
    double TOF2adc()     const { return TOF2adc_;     }

    // Setter methods
    void   setADCs04 (double VMadc,double V3adc,double V6adc,
                                     double VH1adc ,double VH2adc,double VH3adc,double VH4adc,
                                     double CK2adc,double CK3adc,double SciVLEadc,
                                     double Sci521adc,double Sci528adc,
                                     double S1,double S2,double S3,double S4);
    void   setADCs06 (double VMFadc,double VMBadc,
                                     double VM1adc ,double VM2adc,double VM3adc,double VM4adc,
                                     double VM5adc ,double VM6adc,double VM7adc,double VM8adc,
                                     double CK2adc,double CK3adc,double SciVLEadc,
                                     double S1adc,double S2adc,double S3adc,double S4adc,
				     double TOF1adc,double TOF2adc);

  private:
//    TB2004 specific
	double    VMadc_ ; // behind HO
	double    V3adc_ ; // behind HB at (eta,phi)=(7,3)
	double    V6adc_ ;  // behind HB at (eta,phi)=(7,6)
	double    VH1adc_ ; // part of extra muon veto wall - the end of TB04 data taking
	double    VH2adc_ ; // part of extra muon veto wall - the end of TB04 data taking
	double    VH3adc_ ; // part of extra muon veto wall - the end of TB04 data taking
	double    VH4adc_ ; // part of extra muon veto wall - the end of TB04 data taking
	double    Sci521adc_ ; // Scintilator at 521m (see beam line drawings)
	double    Sci528adc_ ; // Scintilator at 522m (see beam line drawings)
//   Common for TB2004 and TB2006
	double    CK2adc_ ; // Cerenkov 2 : electron id 
	double    CK3adc_ ; // Cerenkov 3 : pi/proton separation
	double    SciVLEadc_ ; // Scintillator in VLE beam line 
	double    S1adc_ ; // Trigger scintilator 14x14 cm
	double    S2adc_ ; // Trigger scintilator 4x4 cm
	double    S3adc_ ; // Trigger scintilator 2x2 cm
	double    S4adc_ ; // Trigger scintilator 14x14 cm
//    TB2006 specific
	double    VMFadc_ ; // VM front
	double    VMBadc_ ; // VM back
	double    VM1adc_ ; // Muon veto wall
	double    VM2adc_ ; // Muon veto wall
	double    VM3adc_ ; // Muon veto wall
	double    VM4adc_ ; // Muon veto wall
	double    VM5adc_ ; // Muon veto wall
	double    VM6adc_ ; // Muon veto wall
	double    VM7adc_ ; // Muon veto wall
	double    VM8adc_ ; // Muon veto wall
	double    TOF1adc_ ; // TOF1
	double    TOF2adc_ ; // TOF2


  };

  std::ostream& operator<<(std::ostream& s, const HcalTBBeamCounters& htbcnt);

#endif
