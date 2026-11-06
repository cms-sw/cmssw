#ifndef SimCalorimetry_HGCalSimAlgos_interface_alpaka_HGCROCChannelFramer_h
#define SimCalorimetry_HGCalSimAlgos_interface_alpaka_HGCROCChannelFramer_h

// Alpaka imports
#include <alpaka/alpaka.hpp>
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace hgcaldigi {

    namespace ROCChannelFrameMask {
      constexpr uint32_t kFlagMask = 0x1, kFlagsMask = 0x3, kPacketMask = 0x3ff;
    };

    namespace ROCChannelFrameShift {
      constexpr uint32_t kFlag2Shift = 31, kFlag1Shift = 30, kPacket3Shift = 20, kPacket2Shift = 10, kPacket1Shift = 0;
    };

    /** 
      @short packs the 32b for a single ROC channel
    */
    ALPAKA_FN_HOST_ACC
    inline uint32_t rocFrame(
        bool tc, bool tp, uint16_t adcm1, uint16_t adc, uint16_t tot, uint16_t toa, bool charMode = false) {
      auto compressToT = [&](uint16_t totraw) {
        //removes the two LSB of the 12b TOT and sets the MSB to 1 if truncation happens
        if (totraw > 0x1ff)
          return (0x200 | (totraw >> 3));
        return (totraw & 0x1ff);
      };

      auto packageTo = [&](uint32_t word, uint16_t tidbit, uint32_t mask, uint32_t shift) {
        //mask and shift bits
        //clear to 0 bits which will be set by word and set the tidbit
        const uint32_t masked_tidbit = (tidbit & mask) << shift;
        word &= ~(mask << shift);
        word |= (masked_tidbit);
        return word;
      };

      //build the word
      uint32_t word(0);
      word = packageTo(word, tc, ROCChannelFrameMask::kFlagMask, ROCChannelFrameShift::kFlag2Shift);
      word = packageTo(word, tp, ROCChannelFrameMask::kFlagMask, ROCChannelFrameShift::kFlag1Shift);
      word = packageTo(
          word, (charMode ? adc : adcm1), ROCChannelFrameMask::kPacketMask, ROCChannelFrameShift::kPacket3Shift);
      word = packageTo(word,
                       ((charMode || tc) ? compressToT(tot) : adc),
                       ROCChannelFrameMask::kPacketMask,
                       ROCChannelFrameShift::kPacket2Shift);
      word = packageTo(word, toa, ROCChannelFrameMask::kPacketMask, ROCChannelFrameShift::kPacket1Shift);
      return word;
    };

    /** 
      @short unpacks the TOT from the 32b for a single ROC channel
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackTOT(uint32_t word, bool ignoreTC = false) {
      bool tc = ((word >> ROCChannelFrameShift::kFlag2Shift) & ROCChannelFrameMask::kFlagMask);
      if ((!ignoreTC) && (!tc))
        return 0;

      auto decompressToT = [&](uint16_t totraw) {
        //uncompresses 10b to 12b inverting the truncation operation but loosing the two LSBs
        uint16_t totout(totraw & 0x1ff);
        if (totraw & 0x200) {
          totout = ((totraw & 0x1ff) << 3);
          totout += (1 << 2);
        }
        return totout;
      };

      //extract the TOT word
      uint16_t totraw = decompressToT((word >> ROCChannelFrameShift::kPacket2Shift) & ROCChannelFrameMask::kPacketMask);
      return totraw;
    };

    /** 
      @short characterization mode version of unpackTOT which always reads TOT indepedently of the TC bit
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackTOT_charMode(uint32_t word) { return unpackTOT(word, true); }

    /** 
      @short unpacks the ADC from the 32b for a single ROC channel
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackADC(uint32_t word,
                              uint32_t shift = ROCChannelFrameShift::kPacket2Shift,
                              bool ignoreTC = false) {
      bool tc = ((word >> ROCChannelFrameShift::kFlag2Shift) & ROCChannelFrameMask::kFlagMask);
      if ((!ignoreTC) && tc)
        return 0;
      uint32_t mask = ROCChannelFrameMask::kPacketMask;
      uint16_t adc = (word >> shift) & mask;
      return adc;
    };

    /** 
      @short characterization mode version of unpackADC which always reads ADC indepedently of the TC bit
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackADC_charMode(uint32_t word) {
      return unpackADC(word, ROCChannelFrameShift::kPacket3Shift, true);
    };

    /** 
      @short unpacks the ADCm1 from the 32b for a single ROC channel
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackADCm1(uint32_t word) {
      uint16_t adcm1 = (word >> ROCChannelFrameShift::kPacket3Shift) & ROCChannelFrameMask::kPacketMask;
      return adcm1;
    };

    /** 
      @short characterization mode version of unpackADCm1 : never reads ADCm1
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackADCm1_charMode(uint32_t word) { return 0; };

    /** 
      @short unpacks the TOA from the 32b for a single ROC channel
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackTOA(uint32_t word) {
      uint16_t toa = (word >> ROCChannelFrameShift::kPacket1Shift) & ROCChannelFrameMask::kPacketMask;
      return toa;
    };

    /** 
      @short unpacks the TcTp from the 32b for a single ROC channel
    */
    ALPAKA_FN_HOST_ACC
    inline uint16_t unpackTcTp(uint32_t word) {
      uint16_t tctp = (word >> ROCChannelFrameShift::kFlag1Shift) & ROCChannelFrameMask::kFlagsMask;
      return tctp;
    };

  }  // namespace hgcaldigi
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif
