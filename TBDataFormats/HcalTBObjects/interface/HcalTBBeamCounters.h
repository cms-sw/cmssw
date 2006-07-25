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
    unsigned short VMadc()     const { return VMadc_;     }
    unsigned short V3adc()     const { return V3adc_;     }
    unsigned short V6adc()     const { return V6adc_;     }
    unsigned short VH1adc()     const { return VH1adc_;     }
    unsigned short VH2adc()     const { return VH2adc_;     }
    unsigned short VH3adc()     const { return VH3adc_;     }
    unsigned short VH4adc()     const { return VH4adc_;     }
    unsigned short CK2adc()     const { return CK2adc_;     }
    unsigned short CK3adc()     const { return CK3adc_;     }
    unsigned short SciVLEadc()     const { return SciVLEadc_;     }
    unsigned short Sci521adc()     const { return Sci521adc_;     }
    unsigned short Sci528adc()     const { return Sci528adc_;     }
    unsigned short S1()     const { return S1_;     }
    unsigned short S2()     const { return S2_;     }
    unsigned short S3()     const { return S3_;     }
    unsigned short S4()     const { return S4_;     }

    // Setter methods
    void   setADCs (uint16_t VMadc,uint16_t V3adc,uint16_t V6adc,
                                     uint16_t VH1adc ,uint16_t VH2adc,uint16_t VH3adc,uint16_t VH4adc,
                                     uint16_t CK2adc,uint16_t CK3adc,uint16_t SciVLEadc,
                                     uint16_t Sci521adc,uint16_t Sci528adc,
                                     uint16_t S1,uint16_t S2,uint16_t S3,uint16_t S4);

  private:
	uint16_t    VMadc_ ;
	uint16_t    V3adc_ ;
	uint16_t    V6adc_ ;
	uint16_t    VH1adc_ ;
	uint16_t    VH2adc_ ;
	uint16_t    VH3adc_ ;
	uint16_t    VH4adc_ ;
	uint16_t    CK2adc_ ;
	uint16_t    CK3adc_ ;
	uint16_t    SciVLEadc_ ; 
	uint16_t    Sci521adc_ ;
	uint16_t    Sci528adc_ ;
	uint16_t    S1_ ;
	uint16_t    S2_ ;
	uint16_t    S3_ ;
	uint16_t    S4_ ;

  };

  std::ostream& operator<<(std::ostream& s, const HcalTBBeamCounters& htbcnt);

#endif
