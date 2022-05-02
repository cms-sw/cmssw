#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimCalorimetry/EcalElectronicsEmulation/interface/EcalSimRawData.h"
#include "DataFormats/EcalDigi/interface/EcalMGPASample.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cmath>
#include <fstream>  //used for debugging
#include <iomanip>
#include <iostream>
#include <memory>

const int EcalSimRawData::ttType[nEbTtEta] = {
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1,  // EE-
    0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1   // EE+
};

const int EcalSimRawData::stripCh2Phi[nTtTypes][ttEdge][ttEdge] = {
    // TT type 0:
    /*ch-->*/
    {{4, 3, 2, 1, 0},  /*strip*/
     {0, 1, 2, 3, 4},  /*|*/
     {4, 3, 2, 1, 0},  /*|*/
     {0, 1, 2, 3, 4},  /*|*/
     {4, 3, 2, 1, 0}}, /*V*/
    // TT type 1:
    {{0, 1, 2, 3, 4}, {4, 3, 2, 1, 0}, {0, 1, 2, 3, 4}, {4, 3, 2, 1, 0}, {0, 1, 2, 3, 4}}};

const int EcalSimRawData::strip2Eta[nTtTypes][ttEdge] = {
    {4, 3, 2, 1, 0},  // TT type 0
    {0, 1, 2, 3, 4}   // TT type 1
};

EcalSimRawData::EcalSimRawData(const edm::ParameterSet &params) {
  // sets up parameters:
  digiProducer_ = params.getParameter<std::string>("unsuppressedDigiProducer");
  ebDigiCollection_ = params.getParameter<std::string>("EBdigiCollection");
  eeDigiCollection_ = params.getParameter<std::string>("EEdigiCollection");
  srDigiProducer_ = params.getParameter<std::string>("srProducer");
  ebSrFlagCollection_ = params.getParameter<std::string>("EBSrFlagCollection");
  eeSrFlagCollection_ = params.getParameter<std::string>("EESrFlagCollection");
  tpDigiCollection_ = params.getParameter<std::string>("trigPrimDigiCollection");
  tcpDigiCollection_ = params.getParameter<std::string>("tcpDigiCollection");
  tpProducer_ = params.getParameter<std::string>("trigPrimProducer");
  xtalVerbose_ = params.getUntrackedParameter<bool>("xtalVerbose", false);
  tpVerbose_ = params.getUntrackedParameter<bool>("tpVerbose", false);
  tcc2dcc_ = params.getUntrackedParameter<bool>("tcc2dccData", true);
  srp2dcc_ = params.getUntrackedParameter<bool>("srp2dccData", true);
  fe2dcc_ = params.getUntrackedParameter<bool>("fe2dccData", true);
  fe2tcc_ = params.getUntrackedParameter<bool>("fe2tccData", true);
  dccNum_ = params.getUntrackedParameter<int>("dccNum", -1);
  tccNum_ = params.getUntrackedParameter<int>("tccNum", -1);
  tccInDefaultVal_ = params.getUntrackedParameter<int>("tccInDefaultVal", 0xffff);
  basename_ = params.getUntrackedParameter<std::string>("outputBaseName");

  iEvent = 0;

  std::string writeMode = params.getParameter<std::string>("writeMode");

  if (writeMode == std::string("littleEndian")) {
    writeMode_ = littleEndian;
  } else if (writeMode == std::string("bigEndian")) {
    writeMode_ = bigEndian;
  } else {
    writeMode_ = ascii;
  }

  eeSrFlagToken_ = consumes<EESrFlagCollection>(edm::InputTag(srDigiProducer_, eeSrFlagCollection_));
  ebSrFlagToken_ = consumes<EBSrFlagCollection>(edm::InputTag(srDigiProducer_, ebSrFlagCollection_));
  ebDigisToken_ = consumes<EBDigiCollection>(edm::InputTag(digiProducer_, ebDigiCollection_));
  trigPrimDigisToken_[EcalSimRawData::tcp] =
      consumes<EcalTrigPrimDigiCollection>(edm::InputTag(tpProducer_, tcpDigiCollection_));
  trigPrimDigisToken_[EcalSimRawData::tp] =
      consumes<EcalTrigPrimDigiCollection>(edm::InputTag(tpProducer_, tpDigiCollection_));
}

void EcalSimRawData::analyze(const edm::Event &event, const edm::EventSetup &es) {
  // Event counter:
  ++iEvent;

  if (xtalVerbose_ | tpVerbose_) {
    std::cout << "=================================================================="
                 "====\n"
              << " Event " << iEvent << "\n"
              << "------------------------------------------------------------------"
              << "----\n";
  }

  if (fe2dcc_) {
    std::vector<uint16_t> adc[nEbEta][nEbPhi];
    getEbDigi(event, adc);
    genFeData(basename_, iEvent, adc);
  }

  if (fe2tcc_) {
    int tcp[nTtEta][nTtPhi] = {{0}};
    getTp(event, EcalSimRawData::tcp, tcp);
    genTccIn(basename_, iEvent, tcp);
  }

  if (tcc2dcc_) {
    int tp[nTtEta][nTtPhi] = {{0}};
    getTp(event, EcalSimRawData::tp, tp);
    genTccOut(basename_, iEvent, tp);
  }

  // SR flags:
  int ebSrf[nTtEta][nTtPhi];
  int eeSrf[nEndcaps][nScX][nScY];

  if (srp2dcc_) {
    getSrfs(event, ebSrf, eeSrf);
    genSrData(basename_, iEvent, ebSrf);
  }
}

void EcalSimRawData::elec2GeomNum(int ittEta0, int ittPhi0, int strip1, int ch1, int &iEta0, int &iPhi0) const {
  assert(0 <= ittEta0 && ittEta0 < nEbTtEta);
  assert(0 <= ittPhi0 && ittPhi0 < nTtPhi);
  assert(1 <= strip1 && strip1 <= ttEdge);
  assert(1 <= ch1 && ch1 <= ttEdge);
  const int type = ttType[ittEta0];
  iEta0 = ittEta0 * ttEdge + strip2Eta[type][strip1 - 1];
  iPhi0 = ittPhi0 * ttEdge + stripCh2Phi[type][strip1 - 1][ch1 - 1];
  assert(0 <= iEta0 && iEta0 < nEbEta);
  assert(0 <= iPhi0 && iPhi0 < nEbPhi);
}

void EcalSimRawData::fwrite(std::ofstream &f, uint16_t data, int &iWord, bool hpar) const {
  if (hpar) {
    // set horizontal odd parity bit:
    setHParity(data);
  }

  switch (writeMode_) {
    case littleEndian: {
      char c = data & 0x00FF;
      f.write(&c, sizeof(c));
      c = (data >> 8) & 0x00FF;
      f.write(&c, sizeof(c));
    } break;
    case bigEndian: {
      char c = (data >> 8) & 0x00FF;
      f.write(&c, sizeof(c));
      c = data & 0x00FF;
      f.write(&c, sizeof(c));
    } break;
    case ascii:
      f << ((iWord % 8 == 0 && iWord != 0) ? "\n" : "") << "0x" << std::setfill('0') << std::setw(4) << std::hex << data
        << "\t" << std::dec << std::setfill(' ');
      break;
  }
  ++iWord;
}

std::string EcalSimRawData::getExt() const {
  switch (writeMode_) {
    case littleEndian:
      return ".le";
    case bigEndian:
      return ".be";
    case ascii:
      return ".txt";
    default:
      return ".?";
  }
}

void EcalSimRawData::genFeData(std::string &basename,
                               int iEvent,
                               const std::vector<uint16_t> adcCount[nEbEta][nEbPhi]) const {
  int smf = 0;
  int gmf = 0;
  int nPendingEvt = 0;
  int monitorFlag = 0;
  int chFrameLen = adcCount[0][0].size() + 1;

  int iWord = 0;

  for (int iZ0 = 0; iZ0 < 2; ++iZ0) {
    for (int iDccPhi0 = 0; iDccPhi0 < nDccInPhi; ++iDccPhi0) {
      int iDcc1 = iDccPhi0 + iZ0 * nDccInPhi + nDccEndcap + 1;

      if (dccNum_ != -1 && dccNum_ != iDcc1)
        continue;

      std::stringstream s;
      s.str("");
      const std::string &ext = getExt();
      s << basename << "_fe2dcc" << std::setfill('0') << std::setw(2) << iDcc1 << std::setfill(' ') << ext;
      std::ofstream f(s.str().c_str(), (iEvent == 1 ? std::ios::ate : std::ios::app));

      if (f.fail())
        return;

      if (writeMode_ == ascii) {
        f << (iEvent == 1 ? "" : "\n") << "[Event:" << iEvent << "]\n";
      }

      for (int iTtEtaInSm0 = 0; iTtEtaInSm0 < nTtSmEta; ++iTtEtaInSm0) {
        int iTtEta0 = iZ0 * nTtSmEta + iTtEtaInSm0;
        for (int iTtPhiInSm0 = 0; iTtPhiInSm0 < nTtSmPhi; ++iTtPhiInSm0) {
          // phi=0deg at middle of 1st barrel DCC:
          int iTtPhi0 = -nTtPhisPerEbDcc / 2 + iDccPhi0 * nTtPhisPerEbDcc + iTtPhiInSm0;
          if (iTtPhi0 < 0)
            iTtPhi0 += nTtPhi;
          for (int stripId1 = 1; stripId1 <= ttEdge; ++stripId1) {
            uint16_t stripHeader =
                0xF << 11 | (nPendingEvt & 0x3F) << 5 | (gmf & 0x1) << 4 | (smf & 0x1) << 3 | (stripId1 & 0x7);
            ///	    stripHeader |= parity(stripHeader) << 15;
            fwrite(f, stripHeader, iWord);

            for (int xtalId1 = 1; xtalId1 <= ttEdge; ++xtalId1) {
              uint16_t crystalHeader = 1 << 14 | (chFrameLen & 0xFF) << 4 | (monitorFlag & 0x1) << 3 | (xtalId1 & 0x7);
              //	      crystalHeader |=parity(crystalHeader) << 15;
              fwrite(f, crystalHeader, iWord);

              int iEta0;
              int iPhi0;
              elec2GeomNum(iTtEta0, iTtPhi0, stripId1, xtalId1, iEta0, iPhi0);
              if (xtalVerbose_) {
                std::cout << std::dec << "iDcc1 = " << iDcc1 << "\t"
                          << "iEbTtEta0 = " << iTtEta0 << "\t"
                          << "iEbTtPhi0 = " << iTtPhi0 << "\t"
                          << "stripId1 = " << stripId1 << "\t"
                          << "xtalId1 = " << xtalId1 << "\t"
                          << "iEta0 = " << iEta0 << "\t"
                          << "iPhi0 = " << iPhi0 << "\t"
                          << "adc[5] = 0x" << std::hex << adcCount[iEta0][iPhi0][5] << std::dec << "\n";
              }

              const std::vector<uint16_t> &adc = adcCount[iEta0][iPhi0];
              for (unsigned iSample = 0; iSample < adc.size(); ++iSample) {
                uint16_t data = adc[iSample] & 0x3FFF;
                //		data |= parity(data);
                fwrite(f, data, iWord);
              }  // next time sample
            }    // next crystal in strip
          }      // next strip in TT
        }        // next TT along phi
      }          // next TT along eta
    }            // next DCC
  }              // next half-barrel
}

void EcalSimRawData::genSrData(std::string &basename, int iEvent, int srf[nEbTtEta][nTtPhi]) const {
  for (int iZ0 = 0; iZ0 < 2; ++iZ0) {
    for (int iDccPhi0 = 0; iDccPhi0 < nDccInPhi; ++iDccPhi0) {
      int iDcc1 = iDccPhi0 + iZ0 * nDccInPhi + nDccEndcap + 1;
      if (dccNum_ != -1 && dccNum_ != iDcc1)
        continue;
      std::stringstream s;
      s.str("");
      s << basename << "_ab2dcc" << std::setfill('0') << std::setw(2) << iDcc1 << std::setfill(' ') << getExt();
      std::ofstream f(s.str().c_str(), (iEvent == 1 ? std::ios::ate : std::ios::app));

      if (f.fail())
        throw cms::Exception(std::string("Cannot create/open file ") + s.str() + ".");

      int iWord = 0;

      if (writeMode_ == ascii) {
        f << (iEvent == 1 ? "" : "\n") << "[Event:" << iEvent << "]\n";
      }

      const uint16_t le1 = 0;
      const uint16_t le0 = 0;
      const uint16_t h1 = 1;
      const uint16_t nFlags = 68;
      uint16_t data = (h1 & 0x1) << 14 | (le1 & 0x1) << 12 | (le0 & 0x1) << 11 | (nFlags & 0x7F);

      fwrite(f, data, iWord, true);

      int iFlag = 0;
      data = 0;

      for (int iTtEtaInSm0 = 0; iTtEtaInSm0 < nTtSmEta; ++iTtEtaInSm0) {
        //	int iTtEbEta0 = iZ0*nTtSmEta + iTtEtaInSm0;
        int iTtEta0 = nEeTtEta + iZ0 * nTtSmEta + iTtEtaInSm0;
        for (int iTtPhiInSm0 = 0; iTtPhiInSm0 < nTtSmPhi; ++iTtPhiInSm0) {
          // phi=0deg at middle of 1st barrel DCC:
          int iTtPhi0 = -nTtPhisPerEbDcc / 2 + iDccPhi0 * nTtPhisPerEbDcc + iTtPhiInSm0;
          if (iTtPhi0 < 0)
            iTtPhi0 += nTtPhi;
          // flags are packed by four:
          //|15 |14 |13-12 |11      9|8      6|5      3|2      0|
          //| P | 0 | X  X |  srf i+3| srf i+2| srf i+1| srf i  |
          //|   |   |      | field 3 |field 2 | field 1| field 0|
          const int field = iFlag % 4;
          // std::cout << "TtEta0: " << iTtEta0 << "\tTtPhi0: " << iTtPhi0 << "\n";
          // std::cout << "#" << oct << (int)srf[iTtEta0][iTtPhi0] << "o ****> #" <<
          // oct << (srf[iTtEta0][iTtPhi0] << (field*3)) << "o\n" << std::dec;

          data |= srf[iTtEta0][iTtPhi0] << (field * 3);

          if (field == 3) {
            // std::cout <<  srf[iTtEta0][iTtPhi0] << "----> 0x" << std::hex << data <<
            // "\n";
            fwrite(f, data, iWord, true);
            data = 0;
          }
          ++iFlag;
        }  // next TT along phi
      }    // next TT along eta
    }      // next DCC
  }        // next half-barrel
}

void EcalSimRawData::genTccIn(std::string &basename, int iEvent, const int tcp[nTtEta][nTtPhi]) const {
  for (int iZ0 = 0; iZ0 < 2; ++iZ0) {
    for (int iTccPhi0 = 0; iTccPhi0 < nTccInPhi; ++iTccPhi0) {
      int iTcc1 = iTccPhi0 + iZ0 * nTccInPhi + nTccEndcap + 1;

      if (tccNum_ != -1 && tccNum_ != iTcc1)
        continue;

      std::stringstream s;
      s.str("");
      const char *ext = ".txt";  // only ascii mode supported for TCP

      s << basename << "_tcc" << std::setfill('0') << std::setw(2) << iTcc1 << std::setfill(' ') << ext;
      std::ofstream fe2tcc(s.str().c_str(), (iEvent == 1 ? std::ios::ate : std::ios::app));

      if (fe2tcc.fail())
        throw cms::Exception(std::string("Failed to create file ") + s.str() + ".");

      int memPos = iEvent - 1;
      int iCh1 = 1;
      for (int iTtEtaInSm0 = 0; iTtEtaInSm0 < nTtSmEta; ++iTtEtaInSm0) {
        int iTtEta0 = (iZ0 == 0) ? 27 - iTtEtaInSm0 : 28 + iTtEtaInSm0;
        for (int iTtPhiInSm0 = 0; iTtPhiInSm0 < nTtSmPhi; ++iTtPhiInSm0) {
          // phi=0deg at middle of 1st barrel DCC:
          int iTtPhi0 = -nTtPhisPerEbTcc / 2 + iTccPhi0 * nTtPhisPerEbTcc + iTtPhiInSm0;
          iTtPhi0 += nTtPhisPerEbTcc * iTccPhi0;
          if (iTtPhi0 < 0)
            iTtPhi0 += nTtPhi;
          uint16_t tp_fe2tcc = (tcp[iTtEta0][iTtPhi0] & 0x7ff);  // keep only Et (9:0) and FineGrain (10)

          if (tpVerbose_ && tp_fe2tcc != 0) {
            std::cout << std::dec << "iTcc1 = " << iTcc1 << "\t"
                      << "iTtEta0 = " << iTtEta0 << "\t"
                      << "iTtPhi0 = " << iTtPhi0 << "\t"
                      << "iCh1 = " << iCh1 << "\t"
                      << "memPos = " << memPos << "\t"
                      << "tp = 0x" << std::setfill('0') << std::hex << std::setw(3) << tp_fe2tcc << std::dec
                      << std::setfill(' ') << "\n";
          }
          fe2tcc << iCh1 << "\t" << memPos << "\t" << std::setfill('0') << std::hex << "0x" << std::setw(4) << tp_fe2tcc
                 << "\t"
                 << "0" << std::dec << std::setfill(' ') << "\n";
          ++iCh1;
        }  // next TT along phi
      }    // next TT along eta
      fe2tcc << std::flush;
      fe2tcc.close();
    }  // next TCC
  }    // next half-barrel
}

void EcalSimRawData::genTccOut(std::string &basename, int iEvent, const int tps[nTtEta][nTtPhi]) const {
  int iDccWord = 0;

  for (int iZ0 = 0; iZ0 < 2; ++iZ0) {
    for (int iTccPhi0 = 0; iTccPhi0 < nTccInPhi; ++iTccPhi0) {
      int iTcc1 = iTccPhi0 + iZ0 * nTccInPhi + nTccEndcap + 1;

      if (tccNum_ != -1 && tccNum_ != iTcc1)
        continue;

      std::stringstream s;
      s.str("");
      const char *ext = ".txt";  // only ascii mode supported for TCP

      s << basename << "_tcc" << std::setfill('0') << std::setw(2) << iTcc1 << std::setfill(' ') << ext;

      s.str("");
      s << basename << "_tcc2dcc" << std::setfill('0') << std::setw(2) << iTcc1 << std::setfill(' ') << getExt();
      std::ofstream dccF(s.str().c_str(), (iEvent == 1 ? std::ios::ate : std::ios::app));

      if (dccF.fail()) {
        std::cout << "Warning: failed to create or open file " << s.str() << ".\n";
        return;
      }

      const uint16_t h1 = 1;
      const uint16_t le1 = 0;
      const uint16_t le0 = 0;
      const uint16_t nSamples = 1;
      const uint16_t nTts = 68;
      const uint16_t data =
          (h1 & 0x1) << 14 | (le1 & 0x1) << 12 | (le0 & 0x1) << 11 | (nSamples & 0xF) << 7 | (nTts & 0x7F);
      dccF << (iEvent == 1 ? "" : "\n") << "[Event:" << iEvent << "]\n";
      fwrite(dccF, data, iDccWord, false);

      int memPos = iEvent - 1;
      int iCh1 = 1;
      for (int iTtEtaInSm0 = 0; iTtEtaInSm0 < nTtSmEta; ++iTtEtaInSm0) {
        int iTtEta0 = nEeTtEta + iZ0 * nTtSmEta + iTtEtaInSm0;
        for (int iTtPhiInSm0 = 0; iTtPhiInSm0 < nTtSmPhi; ++iTtPhiInSm0) {
          // phi=0deg at middle of 1st barrel DCC:
          int iTtPhi0 = -nTtPhisPerEbTcc / 2 + iTccPhi0 * nTtPhisPerEbTcc + iTtPhiInSm0;
          if (iTtPhi0 < 0)
            iTtPhi0 += nTtPhi;

          if (tpVerbose_) {
            std::cout << std::dec << "iTcc1 = " << iTcc1 << "\t"
                      << "iTtEta0 = " << iTtEta0 << "\t"
                      << "iTtPhi0 = " << iTtPhi0 << "\t"
                      << "iCh1 = " << iCh1 << "\t"
                      << "memPos = " << memPos << "\t"
                      << "tp = 0x" << std::hex << tps[iTtEta0][iTtPhi0] << std::dec << "\n";
          }
          fwrite(dccF, tps[iTtEta0][iTtPhi0], iDccWord, false);
          ++iCh1;
        }  // next TT along phi
      }    // next TT along eta
    }      // next TCC
  }        // next half-barrel
}

void EcalSimRawData::setHParity(uint16_t &a) const {
  const int odd = 1 << 15;
  const int even = 0;
  // parity bit of numbers from 0x0 to 0xF:
  //                    0   1   2    3   4    5    6   7   8    9    A   B    C
  //                    D   E    F
  const int p[16] = {even, odd, odd, even, odd, even, even, odd, odd, even, even, odd, even, odd, odd, even};
  // inverts parity bit (LSB) of 'a' in case of even parity:
  a ^= p[a & 0xF] ^ p[(a >> 4) & 0xF] ^ p[(a >> 8) & 0xF] ^ p[a >> 12 & 0xF] ^ odd;
}

void EcalSimRawData::getSrfs(const edm::Event &event,
                             int ebSrf[nTtEta][nTtPhi],
                             int eeSrf[nEndcaps][nScX][nScY]) const {
  // EE
  const auto &hEeSrFlags = event.getHandle(eeSrFlagToken_);
  for (size_t i = 0; i < (nEndcaps * nScX * nScY); ((int *)eeSrf)[i++] = -1) {
  };
  if (hEeSrFlags.isValid()) {
    for (EESrFlagCollection::const_iterator it = hEeSrFlags->begin(); it != hEeSrFlags->end(); ++it) {
      const EESrFlag &flag = *it;
      int iZ0 = flag.id().zside() > 0 ? 1 : 0;
      int iX0 = flag.id().ix() - 1;
      int iY0 = flag.id().iy() - 1;
      assert(iZ0 >= 0 && iZ0 < nEndcaps);
      assert(iX0 >= 0 && iX0 < nScX);
      assert(iY0 >= 0 && iY0 < nScY);
      eeSrf[iZ0][iX0][iY0] = flag.value();
    }
  } else {
    edm::LogWarning("EcalSimRawData") << "EE SR flag not found (Product label: " << srDigiProducer_
                                      << "Producet instance: " << eeSrFlagCollection_ << ")";
  }

  // EB
  const auto &hEbSrFlags = event.getHandle(ebSrFlagToken_);
  for (size_t i = 0; i < (nTtEta * nTtPhi); ((int *)ebSrf)[i++] = -1) {
  };
  if (hEbSrFlags.isValid()) {
    for (EBSrFlagCollection::const_iterator it = hEbSrFlags->begin(); it != hEbSrFlags->end(); ++it) {
      const EBSrFlag &flag = *it;
      int iEta = flag.id().ieta();
      int iEta0 = iEta + nTtEta / 2 - (iEta >= 0 ? 1 : 0);  // 0->55 from eta=-3 to eta=3
      int iEbEta0 = iEta0 - nEeTtEta;                       // 0->33 from eta=-1.48 to eta=1.48
      int iPhi0 = flag.id().iphi() - 1;

      assert(iEbEta0 >= 0 && iEbEta0 < nEbTtEta);
      assert(iPhi0 >= 0 && iPhi0 < nTtPhi);

      ebSrf[iEbEta0][iPhi0] = flag.value();
    }
  } else {
    edm::LogWarning("EcalSimRawData") << "EB SR flag not found (Product label: " << srDigiProducer_
                                      << "Producet instance: " << ebSrFlagCollection_ << ")";
  }
}

void EcalSimRawData::getEbDigi(const edm::Event &event, std::vector<uint16_t> adc[nEbEta][nEbPhi]) const {
  const auto &hEbDigis = event.getHandle(ebDigisToken_);

  int nSamples = 0;
  if (hEbDigis.isValid() && !hEbDigis->empty()) {  // there is at least one digi
    nSamples = hEbDigis->begin()->size();          // gets the sample count from 1st digi
  }

  const uint16_t suppressed = 0xFFFF;

  adc[0][0] = std::vector<uint16_t>(nSamples, suppressed);

  for (int iEbEta = 0; iEbEta < nEbEta; ++iEbEta) {
    for (int iEbPhi = 0; iEbPhi < nEbPhi; ++iEbPhi) {
      adc[iEbEta][iEbPhi] = adc[0][0];
    }
  }
  if (hEbDigis.isValid()) {
    if (xtalVerbose_)
      std::cout << std::setfill('0');
    for (EBDigiCollection::const_iterator it = hEbDigis->begin(); it != hEbDigis->end(); ++it) {
      const EBDataFrame &frame = *it;

      int iEta0 = iEta2cIndex((frame.id()).ieta());
      int iPhi0 = iPhi2cIndex((frame.id()).iphi());

      //     std::cout << "xtl indices conv: (" << frame.id().ieta() << ","
      // 	 << frame.id().iphi() << ") -> ("
      // 	 << iEta0 << "," << iPhi0 << ")\n";

      if (iEta0 < 0 || iEta0 >= nEbEta) {
        std::cout << "iEta0 (= " << iEta0 << ") is out of range ("
                  << "[0," << nEbEta - 1 << "])\n";
      }
      if (iPhi0 < 0 || iPhi0 >= nEbPhi) {
        std::cout << "iPhi0 (= " << iPhi0 << ") is out of range ("
                  << "[0," << nEbPhi - 1 << "])\n";
      }

      if (xtalVerbose_) {
        std::cout << iEta0 << "\t" << iPhi0 << ":\t";
        std::cout << std::hex;
      }

      if (nSamples != frame.size()) {
        throw cms::Exception("EcalSimRawData",
                             "Found EB digis with different sample count! This "
                             "is not supported by EcalSimRawData.");
      }

      for (int iSample = 0; iSample < nSamples; ++iSample) {
        const EcalMGPASample &sample = frame.sample(iSample);
        uint16_t encodedAdc = sample.raw();
        adc[iEta0][iPhi0][iSample] = encodedAdc;
        if (xtalVerbose_) {
          std::cout << (iSample > 0 ? " " : "") << "0x" << std::setw(4) << encodedAdc;
        }
      }
      if (xtalVerbose_)
        std::cout << "\n" << std::dec;
    }
    if (xtalVerbose_)
      std::cout << std::setfill(' ');
  }
}

void EcalSimRawData::getTp(const edm::Event &event, EcalSimRawData::tokenType type, int tcp[nTtEta][nTtPhi]) const {
  const auto &hTpDigis = event.getHandle(trigPrimDigisToken_[type]);
  if (hTpDigis.isValid() && !hTpDigis->empty()) {
    const EcalTrigPrimDigiCollection &tpDigis = *hTpDigis.product();

    //    EcalSelectiveReadout::ttFlag_t ttf[nTtEta][nTtPhi];
    for (int iTtEta0 = 0; iTtEta0 < nTtEta; ++iTtEta0) {
      for (int iTtPhi0 = 0; iTtPhi0 < nTtPhi; ++iTtPhi0) {
        tcp[iTtEta0][iTtPhi0] = tccInDefaultVal_;
      }
    }
    if (tpVerbose_) {
      std::cout << std::setfill('0');
    }
    for (EcalTrigPrimDigiCollection::const_iterator it = tpDigis.begin(); it != tpDigis.end(); ++it) {
      const EcalTriggerPrimitiveDigi &tp = *it;
      int iTtEta0 = iTtEta2cIndex(tp.id().ieta());
      int iTtPhi0 = iTtPhi2cIndex(tp.id().iphi());
      if (iTtEta0 < 0 || iTtEta0 >= nTtEta) {
        std::cout << "iTtEta0 (= " << iTtEta0 << ") is out of range ("
                  << "[0," << nEbTtEta - 1 << "])\n";
      }
      if (iTtPhi0 < 0 || iTtPhi0 >= nTtPhi) {
        std::cout << "iTtPhi0 (= " << iTtPhi0 << ") is out of range ("
                  << "[0," << nTtPhi - 1 << "])\n";
      }

      tcp[iTtEta0][iTtPhi0] = tp[tp.sampleOfInterest()].raw();

      if (tpVerbose_) {
        if (tcp[iTtEta0][iTtPhi0] != 0) {  // print non-zero values only
          std::string collName = (type == 0) ? tcpDigiCollection_ : tpDigiCollection_;
          std::cout << collName << (collName.empty() ? "" : " ") << "TP(" << std::setw(2) << iTtEta0 << "," << iTtPhi0
                    << ") = "
                    << "0x" << std::setw(4) << tcp[iTtEta0][iTtPhi0] << "\tcmssw indices: " << tp.id().ieta() << " "
                    << tp.id().iphi() << "\n";
        }
      }
    }  // next TP
    if (tpVerbose_)
      std::cout << std::setfill(' ');
  }
}
