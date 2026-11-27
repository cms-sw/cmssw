#ifndef TPGFEModuleEmulation_h
#define TPGFEModuleEmulation_h

#include <iostream>
#include <cstring>
#include <vector>
#include <cassert>
#include <cstddef>

#include "EventFilter/HGCalRawToDigi/interface/TPGFEDataformat.hh"
#include "SimCalorimetry/HGCalSimProducers/interface/TPGFEDataformatEmulation.hh"
#include "DataFormats/HGCalDigi/interface/HGCalTriggerDefinitions.h"

#include "CondFormats/HGCalObjects/interface/HGCalTriggerConfiguration.h"

namespace TPGFEModuleEmulation{

  uint16_t CompressHgroc(uint32_t val, uint8_t isldm) { return TPGFEDataformat::TcRawData::EncodeEM<4, 3>(val >> (isldm ? 1 : 3)); }

  //The following emulation function performs 1) pedestal subtraction, 2) linearization, 3) compression
  TPGFEDataformatEmulation::ModuleTcData EmulateHgroc(bool isSim, const std::map<uint32_t,std::vector<uint32_t>>& tcPinMap, const HGCalECONTConfig& econtCfg,  std::map<uint32_t,TPGFEDataformatEmulation::HalfHgcrocData>& rocdata){
    TPGFEDataformatEmulation::ModuleTcData mdata;
    uint16_t bx = 0xFFFF;
    const uint32_t nTCs = tcPinMap.size(); //read it from cfg econt
    uint8_t selTC4 = !(econtCfg.density);
    mdata.setNofTCs(nTCs);

    for (auto it = tcPinMap.begin(); it != tcPinMap.end(); ++it){ // loop over TCs of the module
      auto itc = it->first;
      LogDebug("[HGCalTriggerEmulator]") << "TPGFEModuleEmulation::HGCROCTPGEmulation::Emulate: tclist["<<itc<<"] " << std::endl;
      uint32_t totadc = 0; // accumulated charge of the TC, which will be compressed and stored in the output TcData. For simulation, it is either accumulated ADC or TOT depending on the signal type; for beam-test data, it is the accumulated ADC after pedestal subtraction and threshold cut, or the accumulated TOT after threshold cut and pedestal correction, depending on the signal type.
      uint8_t isTot = 0; // stand for whether the TC is triggered by TOT signal
      uint8_t isTcTp1 = 0; // stand for whether the TC is triggered by a channel with TC/Tp flag = 1
      uint8_t isTcTp2 = 0; // TC means trigger cell, and Tp means pass through. The TC/Tp flag information is provided in the input data for each channel, and is used to keep track of the signal type of the channels corresponding to the TC, which will be stored in the output TcData.
      uint8_t isTcTp3 = 0; // ...

      std::vector<uint32_t> pinlist = it->second;
      for(const auto& tcch : pinlist){ // loop over sensor channels corresponding to the TC

        uint32_t rocpin = tcch%36 ; // the pin number of the channel within the ROC, which is needed for retrieving the channel data and parameters of the corresponding half-ROC.
        uint32_t rocn = TMath::Floor(tcch/72); // the ROC number
        uint32_t half = (int(TMath::Floor(tcch/36))%2==0)?0:1; // the half-ROC number
        uint32_t hrocid = 2*rocn + half; // the half roc id
        HGCalROCTrigConfig hrocCfg = econtCfg.hgcrocs[hrocid];

        if(rocdata.find(hrocid)==rocdata.end()){ // if the half-ROC data is not found in the input data, skip the channel and print warning
          LogDebug("[HGCalTriggerEmulator]") << "TPGFEModuleEmulation::HGCROCTPGEmulation::Emulate: HalfRoc not found in data for Tcch: "<< tcch << ", rocn: " << rocn << ", half: " << half << ", rocpin: " << rocpin << std::endl;
          continue ;
        }

        LogDebug("[HGCalTriggerEmulator]")<<"\t TPGFEModuleEmulation::HGCROCTPGEmulation::Emulate: TC : " << itc << ", tcch: " << tcch <<", rocpin : "<<rocpin<<", hrocid: "<<hrocid<<", rocn: "<<rocn<<", half: "<<half<<std::endl; //print the channel data for the reference event

        TPGFEDataformatEmulation::HalfHgcrocChannelData& chdata = rocdata.at(hrocid).getChannelData(rocpin); //get the channel data corresponding to the channel, which contains ADC, TOT, TC/Tp flag information.
        bx = rocdata.at(hrocid).getBx(); // get the bx information from the half-ROC data
        if(chdata.tctp==1) isTcTp1 = 1;
        if(chdata.tctp==2) isTcTp2 = 1;
        if(chdata.tctp==3) {
          isTcTp3 = 1;
          isTot = 1;
        }

        // chdata.print();
        if(!isSim){ //if not simulation, assumed beamtest data
          uint8_t ped =  hrocCfg.ped[rocpin];  //FIXME

          if(!isTot){ // if the signal is ADC, perform pedestal subtraction and threshold cut, and accumulate the ADC if the channel is not masked and pedestal is valid. If the channel is masked or pedestal is invalid, set the ADC to 0.
            uint8_t thr =  hrocCfg.adc_th;
            uint32_t adc = chdata.adc;

            adc = (adc>(ped+thr) and !(hrocCfg.isChMasked(rocpin)) and (ped<0xFF)) ? adc-ped : 0 ;
            totadc += adc; // accumulate the ADC of the channels corresponding to the TC
            LogDebug("[HGCalTriggerEmulator]")<<"\t TPGFEModuleEmulation::HGCROCTPGEmulation::Emulate: ped: " << ped << ", thr: " << thr <<", adc : "<<adc<<", hrocCfg.isChMasked(rocpin): "<< hrocCfg.isChMasked(rocpin) <<std::endl; //print the pedestal, threshold, ADC after pedestal subtraction and threshold cut, and masking information for the reference event
          }
          else{ // if the signal is TOT, perform threshold cut and pedestal correction.
            uint32_t tot1 = (chdata.tot>=hrocCfg.getTotTH(rocpin)) ? (chdata.tot-hrocCfg.getTotP(rocpin)) : (hrocCfg.getTotTH(rocpin)-hrocCfg.getTotP(rocpin)) ;
            //ideally ped<255 condition should only be restricted for ADC case, however TOT signal triggers in emulation for event 146245 of relay 1695829026 and link 1, which is not that seen by ECONT data
            uint32_t totlin = (!(hrocCfg.isChMasked(rocpin)) and (ped<0xFF))?tot1*hrocCfg.MultFactor:0;  // perform pedestal correction and linearization for TOT signal, and set the TOT to 0 if the channel is masked or pedestal is invalid.
            totadc += totlin;
            //isTot = true;
            LogDebug("[HGCalTriggerEmulator]")<<"\t TPGFEModuleEmulation::HGCROCTPGEmulation::Emulate: tot1: " << tot1 << ", tot: " << chdata.tot <<", thr : "<< hrocCfg.getTotTH(rocpin) <<", ped: "<< hrocCfg.getTotP(rocpin) << ", totlin : "<< totlin <<std::endl;
          }//istot or adc
        }
        else{ // if simulation, directly accumulate ADC or TOT depending on the signal type without pedestal subtraction, threshold cut, and linearization, as the input simulation data is assumed to be already prepared with ideal detector response and perfectly calibrated.
          //check CMSSW for details should be
          if(!isTot)
            totadc += chdata.adc;
          else
            totadc += chdata.tot;
        } //isCMSSW simulation of beam-test analysis
      } //loop of sensor channels

      mdata.setCharge(itc, totadc);
      mdata.setTot(itc, isTot); //stand for tctp==3
      mdata.setTcTp1(itc, isTcTp1);
      mdata.setTcTp2(itc, isTcTp2);
      mdata.setTcTp3(itc, isTcTp3);
      mdata.setCdata(itc, CompressHgroc(totadc,selTC4 ));

    }//loop over TCs
    mdata.setBx(bx);
    //std::cout << "\t TPGFEModuleEmulation::HGCROCTPGEmulation::Emulate: " << std::endl;
    //mdata.print();
    return mdata;
  }


  uint32_t DecompressEcont(uint16_t compressed, uint8_t density){
    //4E+3M with midpoint correction
    //controlled by the density parameter of ECON-T (0/1 stands for LD/HD and represents 1/3 bit shifts)
    uint32_t mant = compressed & 0x7;
    uint32_t expo = (compressed>>3) & 0xF;

    if(expo==0) return (density) ? (mant<<3)+4 : (mant<<1)+1 ; //The the +4/+1 are midpoint corrections for expo==0

    uint32_t shift = expo+2;
              shift += (density) ? 3 : 1;
    uint32_t decomp = 1<<shift;
    uint32_t mpdeco = 1<<(shift-4);
    decomp = decomp | (mant<<(shift-3));
    decomp = decomp | mpdeco;
      return decomp;
  }

  uint16_t CompressEcontBc(uint64_t val, uint32_t dropLSB) { return TPGFEDataformat::TcRawData::EncodeEM<4, 3>(val >> dropLSB); }
  uint16_t CompressEcontModsum(uint64_t val, uint32_t /*dropLSB*/) { return TPGFEDataformat::TcRawData::EncodeEM<5, 3>(val); }

  void batcherOEMSort(std::vector<TPGFEDataformat::TcRawData>& tc) {
    //C++ adaptation of batcher odd-even sorting of https://github.com/dnoonan08/ECONT_Emulator/ASICBlocks/bestchoice.py
    //Requires 49 Tcs for sorting
    TPGFEDataformat::TcRawData dummy(hgcal::econt::Unknown,0x3f, 0, 0, false, false, false);
    tc.push_back(dummy);

    uint32_t N = uint32_t(tc.size());
    uint32_t t = uint32_t(ceil(log(N)/log(2)));
    uint32_t p = uint32_t(pow(2,(t-1)));
    while(p>=1){
      uint32_t q = uint32_t(pow(2,(t-1)));
      uint32_t r = 0;
      uint32_t d = p;
      while (q>=p){
        for(uint32_t i=0; i<(N-d) ; i++){
          if ((i & p) != r) continue;
          if (tc[i] < tc[i+d]) std::swap(tc[i], tc[i+d]);
        }
        d = q - p;
        q = floor(q/2);
        r = p;
      }
      p = floor(p/2);
    }
    tc.pop_back();

  }

  TPGFEDataformat::TcRawDataPacket EmulateEcontSTC(bool isSim, const HGCalECONTConfig& econtCfg , const TPGFEDataformatEmulation::ModuleTcData& mdata) {
    //FIXME to be implemented
    TPGFEDataformat::TcRawDataPacket rdp;
    return rdp;
  }

  TPGFEDataformat::TcRawDataPacket EmulateEcontBc(bool isSim, const HGCalECONTConfig& econtCfg , const TPGFEDataformatEmulation::ModuleTcData& mdata) {
    TPGFEDataformat::TcRawDataPacket rdp;
    uint32_t dropLSB = econtCfg.dropLSB ;

    const hgcal::econt::Type& outputType = econtCfg.getOutType();

    uint32_t nTCs =  mdata.getNofTCs();
    uint16_t bx = (mdata.getBx()==3564) ? 0xF : mdata.getBx() & 0x7 ; //make 8 modulo
    uint64_t decompressedMS = 0;
    std::vector<TPGFEDataformat::TcRawData> tcrawdatalist;
    TPGFEDataformat::TcRawData tcdata;

    for(uint32_t econtc = 0; econtc < nTCs;  econtc++){
      TPGFEDataformatEmulation::HgcrocTcData hrocTc = mdata.getTC(econtc);

      uint64_t decompressed = DecompressEcont(hrocTc.encoded, econtCfg.density); //funny that return needs to defined as 64-bit
      uint64_t decomp64bit = decompressed * econtCfg.getCalibration(econtc); //overflows for 12bit TOT
      LogDebug("[HGCalTriggerEmulator]") << "TPGFEModuleEmulation::EmulateEcontBc (TC="<<econtc<<") decompressed: " << decompressed << ", decompressed*calib: " << decomp64bit << std::endl;
      decompressed =  decomp64bit >> 11;
      decompressedMS += (decompressed >> dropLSB) ;
      LogDebug("[HGCalTriggerEmulator]") << "TPGFEModuleEmulation::EmulateEcontBc (TC="<<econtc<<") decompressed*calib>>11: " << decompressed << ", decompressedMS: " << decompressedMS << std::endl;
      uint16_t compressed_bc = CompressEcontBc(decompressed,dropLSB);
      tcdata.setTriggerCell(outputType, econtc, compressed_bc, decompressed>>dropLSB, hrocTc.istctp1, hrocTc.istctp2, hrocTc.istctp3) ;
      tcrawdatalist.push_back(tcdata);
    }
    batcherOEMSort(tcrawdatalist);
    //for(int itc=0;itc<int(tcrawdatalist.size());itc++) tcrawdatalist[itc].print();

    uint16_t compressed_modsum = CompressEcontModsum(decompressedMS,dropLSB);
    rdp.reset();
    rdp.setTBM(outputType, bx, compressed_modsum, decompressedMS>>dropLSB);
    uint32_t realTcSize = tcrawdatalist.size();
    if (realTcSize > 0 && tcrawdatalist.back().address() == 0x3f) { // check if this condition still make sense.
      realTcSize--;
    }
    uint32_t nofBCTcs = econtCfg.getNofTCs();
    if (nofBCTcs > realTcSize) {
      LogDebug("[HGCalTriggerEmulator]") << "TPGFEModuleEmulation::EmulateBC Only found " << realTcSize << " TCs but expected " << nofBCTcs << std::endl;
      nofBCTcs = realTcSize;
    }
    for(uint32_t itc = 0 ; itc<nofBCTcs ; itc++)
      rdp.setTcData(outputType, tcrawdatalist[itc].address(), tcrawdatalist[itc].energy(), tcrawdatalist[itc].rawE(),tcrawdatalist[itc].isTcTp1(),tcrawdatalist[itc].isTcTp2(),tcrawdatalist[itc].isTcTp3());
    return rdp;
  }//Emulate BC

  //The following emulation function performs 1) decompression, 2) calibration, 3) compression
  TPGFEDataformat::TcRawDataPacket EmulateEcont(bool isSim,  const HGCalECONTConfig& econtCfg, const TPGFEDataformatEmulation::ModuleTcData& mdata){
    const hgcal::econt::Type& outputType = econtCfg.getOutType();

    if(outputType==hgcal::econt::BestC)
      return EmulateEcontBc(isSim, econtCfg, mdata);
    else if(outputType==hgcal::econt::STC4A or outputType==hgcal::econt::STC4B or outputType==hgcal::econt::STC16 or outputType==hgcal::econt::CTC4A or outputType==hgcal::econt::CTC4B)
      return EmulateEcontSTC(isSim, econtCfg, mdata); // FIXME to be implemented
    else return EmulateEcontSTC(isSim, econtCfg, mdata); // FIXME -- here to prevent lack of return in non-void func (after fixing the elseif above) - Jan
  }

}//end of namespace

#endif