#ifndef TPGFEDataformatEmulation_h
#define TPGFEDataformatEmulation_h

#include "DataFormats/HGCalDigi/interface/HGCalTriggerDefinitions.h"
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include "TMath.h"

namespace TPGFEDataformatEmulation{

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////// Data Formats by Paul Dauncey /////////////////////////////////
  ///provision for tctp is added by Indra (without altering the existing functionalitites)

  struct HalfHgcrocChannelData {
    uint16_t adc;
    uint16_t tot;
    uint8_t tctp;
  };

  class HalfHgcrocData {
  public:

    HalfHgcrocData() : _bx(0xFFFF) {
      setZero();
    }

    void setZero() {
      std::memset((void *)_data,0,sizeof(HalfHgcrocChannelData)*hgcal::hgcroc::NUM_CHANNELS);
    }

    void setBx(uint16_t bx) { _bx = bx;}
    void setSlinkBx(uint16_t bx) { _bxId = bx;}
    uint8_t hasTOT() const { return hasTcTp(3); }
    uint8_t hasTcTp1() const { return hasTcTp(1); }
    uint8_t hasTcTp2() const { return hasTcTp(2); }
    uint8_t hasTcTp3() const { return hasTOT(); }
    uint8_t hasTcTp(uint16_t tctpval) const {
      for(unsigned i(0);i<hgcal::hgcroc::NUM_CHANNELS;i++)
        if(_data[i].tctp==tctpval) {
        // std::cout << "TOT/TOA noted for ich : " << i << std::endl;
        // _data[i].print();
        return true;
      }
      return false;
    }
    void setChAdc(uint16_t i, uint16_t ch_adc) {_data[i].adc = ch_adc;}
    void setChTot(uint16_t i, uint16_t ch_tot) {_data[i].tot = ch_tot;}
    void setChTctp(uint16_t i, uint8_t ch_tctp) {_data[i].tctp = ch_tctp;}

    const uint16_t getChAdc(uint16_t i) {return _data[i].adc;}
    const uint16_t getChTot(uint16_t i, uint16_t ch_tot) {return _data[i].tot;}
    const uint8_t getChTctp(uint16_t i, uint8_t ch_tctp) {return _data[i].tctp;}
    const uint32_t getBx() const {return uint32_t(_bx);}
    const uint32_t getSlinkBx() const {return uint32_t(_bxId);}
    HalfHgcrocChannelData& getChannelData(uint32_t i) {return _data[i];}

    void print() const {
      std::cout << "HalfHgcrocData(" << this << ")::print()" << std::endl;

      for(unsigned i(0);i<hgcal::hgcroc::NUM_CHANNELS;i++) {
        std::cout << " Channel " << std::setw(2) << i << " = 0x"
                  << " ADC "<< _data[i].adc << ","
                  << " Tot "<< _data[i].tot << ","
                  << " bx: " << getBx() << ","
                  << " TcTp: " << uint32_t(_data[i].tctp)
                  << std::endl;
      }
    }

  private:
    uint16_t _bx;//from ECON header
    uint16_t _bxId;//from Slink trailer
    HalfHgcrocChannelData _data[hgcal::hgcroc::NUM_CHANNELS];
  };


  struct HgcrocTcData {
    uint32_t raw;  //raw-uncompressed data
    uint16_t encoded; //compressed 4E+3M ()
    uint8_t isTot;     //isTOT or ADC
    uint8_t istctp1;
    uint8_t istctp2;
    uint8_t istctp3;
  };

  class ModuleTcData {
  public:

    ModuleTcData() {
      setZero();
    }
    void setZero() {
      std::memset((void *)_data,0,sizeof(HgcrocTcData)*hgcal::econt::DEFAULT_MAX_TC);
      NumberOfTCs=0;
    }
    void setTcTp1(uint32_t i, uint8_t value) { _data[i].istctp1 = value; }
    void setTcTp2(uint32_t i, uint8_t value) { _data[i].istctp2 = value; }
    void setTcTp3(uint32_t i, uint8_t value) { _data[i].istctp3 = value; }
    void setTot(uint32_t i, uint8_t value) { _data[i].isTot = value; }
    void setCharge(uint32_t i, uint32_t a) { _data[i].raw = (a<0x1000000)?a:0xffffff; }
    void setCdata(uint32_t i, uint16_t a) {
      if (a<0x80) _data[i].encoded = a;
      else {
        LogDebug("[HGCalTriggerEmulator]") << "TPGFEModuleEmulation::HgcrocTcData:: TC data " << a << "not packed into 7bits, putting to zero" << std::endl;
        _data[i].encoded = 0;
      }
    }
    void setBx(uint16_t bx) { _bx = bx;}
    void setNofTCs(const unsigned nofTCs) {NumberOfTCs = nofTCs;}
    void setTCs(HgcrocTcData* data) {
      for(uint16_t i(0);i<NumberOfTCs;i++)
        _data[i] = data[i];
    }

    const uint32_t getBx() const {return uint32_t(_bx);}
    const uint32_t getNofTCs() const {return uint32_t(NumberOfTCs);}
    const HgcrocTcData* getTCs() const {return _data;}
    const HgcrocTcData& getTC(uint32_t i) const {return _data[i];}
    HgcrocTcData getTC(uint32_t i) {return _data[i];}
    bool isTcTp1() const {
      for(uint16_t i(0);i<NumberOfTCs;i++) if(_data[i].istctp1) return true;
      return false;
    }
    bool isTcTp2() const {
      for(uint16_t i(0);i<NumberOfTCs;i++) if(_data[i].istctp2) return true;
      return false;
    }
    bool isTcTp3() const {
      for(uint16_t i(0);i<NumberOfTCs;i++) if(_data[i].isTot) return true;
      return false;
    }

    void print() const {
      std::cout << "ModuleTriggerCellData(" << this << ")::print() : NumberOfTCs :" << NumberOfTCs << ", bx: " << getBx() << ", isTcTp1 : " << isTcTp1() << ", isTcTp2 : " << isTcTp2() << ", isTcTp3 : " << isTcTp3() << std::endl;

      for(uint16_t i(0);i<NumberOfTCs;i++) {
        std::cout << " TC " << std::setw(2) << i << ": compressed = "
        << std::dec << ::std::setfill(' ')
        << std::setw(10) << _data[i].encoded
        << std::dec << ::std::setfill(' ')
        << ", raw-uncompressed "
        << std::setw(10) << _data[i].raw
        << ", istot : "
        << std::setw(2) << uint32_t(_data[i].isTot)
        << ", istctp1 : "
        << std::setw(2) << uint32_t(_data[i].istctp1)
        << ", istctp2 : "
        << std::setw(2) << uint32_t(_data[i].istctp2)
        << std::endl;
      }
    }

  private:
    HgcrocTcData _data[hgcal::econt::DEFAULT_MAX_TC];
    uint16_t _bx;
    uint16_t NumberOfTCs;
  };

}
#endif
