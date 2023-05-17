//emacs settings:-*- mode: c++; c-basic-offset: 2; indent-tabs-mode: nil -*-
/*
 *
 * author: Ph Gras. June, 2010
 */

#include "SimCalorimetry/EcalSelectiveReadoutProducers/interface/namespace_ecalsrcondtools.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <algorithm>

constexpr int tccNum[12][12] = {
    /* EE- */
    {36, 19, 20, 21, 22, 23, 18, 1, 2, 3, 4, 5},       //SRP 1
    {24, 25, 26, 27, 28, 29, 6, 7, 8, 9, 10, 11},      //SRP 2
    {30, 31, 32, 33, 34, 35, 12, 13, 14, 15, 16, 17},  //SRP 3
    /* EB- */
    {54, 37, 38, 39, 40, 41, -1, -1, -1, -1, -1, -1},  //SRP 4
    {42, 43, 44, 45, 46, 47, -1, -1, -1, -1, -1, -1},  //SRP 5
    {48, 49, 50, 51, 52, 53, -1, -1, -1, -1, -1, -1},  //SRP 6
    /* EB+ */
    {72, 55, 56, 57, 58, 59, -1, -1, -1, -1, -1, -1},  //SRP 7
    {60, 61, 62, 63, 64, 65, -1, -1, -1, -1, -1, -1},  //SRP 8
    {66, 67, 68, 69, 70, 71, -1, -1, -1, -1, -1, -1},  //SRP 9
    /* EE+ */
    {90, 73, 74, 75, 76, 77, 108, 91, 92, 93, 94, 95},      //SRP 10
    {78, 79, 80, 81, 82, 83, 96, 97, 98, 99, 100, 101},     //SRP 11
    {84, 85, 86, 87, 88, 89, 102, 103, 104, 105, 106, 107}  //SRP 12
};

constexpr int dccNum[12][12] = {
    {1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1},     //SRP 1
    {4, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1},     //SRP 2
    {7, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1},     //SRP 3
    {10, 11, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1},  //SRP 4
    {16, 17, 18, 19, 20, 21, -1, -1, -1, -1, -1, -1},  //SRP 5
    {22, 23, 24, 25, 26, 27, -1, -1, -1, -1, -1, -1},  //SRP 6
    {28, 29, 30, 31, 32, 33, -1, -1, -1, -1, -1, -1},  //SRP 7
    {34, 35, 36, 37, 38, 39, -1, -1, -1, -1, -1, -1},  //SRP 8
    {40, 41, 42, 43, 44, 45, -1, -1, -1, -1, -1, -1},  //SRP 9
    {46, 47, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1},  //SRP 10
    {49, 50, 51, -1, -1, -1, -1, -1, -1, -1, -1, -1},  //SRP 11
    {52, 53, 54, -1, -1, -1, -1, -1, -1, -1, -1, -1}   //SRP 12
};

using namespace std;

namespace ecalsrcondtools {

  string tokenize(const string& s, const string& delim, int& pos) {
    if (pos < 0)
      return "";
    int pos0 = pos;
    int len = s.size();
    //eats delimeters at beginning of the string
    while (pos0 < len && find(delim.begin(), delim.end(), s[pos0]) != delim.end()) {
      ++pos0;
    }
    if (pos0 == len)
      return "";
    pos = s.find_first_of(delim, pos0);
    return s.substr(pos0, (pos > 0 ? pos : len) - pos0);
  }

  std::string trim(std::string s) {
    std::string::size_type pos0 = s.find_first_not_of(" \t");
    if (pos0 == string::npos) {
      pos0 = 0;
    }
    string::size_type pos1 = s.find_last_not_of(" \t") + 1;
    if (pos1 == string::npos) {
      pos1 = pos0;
    }
    return s.substr(pos0, pos1 - pos0);
  }

  void importParameterSet(EcalSRSettings& sr, const edm::ParameterSet& ps) {
    sr.deltaPhi_.resize(1);
    sr.deltaPhi_[0] = ps.getParameter<int>("deltaPhi");
    sr.deltaEta_.resize(1);
    sr.deltaEta_[0] = ps.getParameter<int>("deltaEta");
    sr.ecalDccZs1stSample_.resize(1);
    sr.ecalDccZs1stSample_[0] = ps.getParameter<int>("ecalDccZs1stSample");
    sr.ebDccAdcToGeV_ = ps.getParameter<double>("ebDccAdcToGeV");
    sr.eeDccAdcToGeV_ = ps.getParameter<double>("eeDccAdcToGeV");
    sr.dccNormalizedWeights_.resize(1);
    const std::vector<double>& w = ps.getParameter<std::vector<double> >("dccNormalizedWeights");
    sr.dccNormalizedWeights_[0].resize(w.size());
    std::copy(w.begin(), w.end(), sr.dccNormalizedWeights_[0].begin());
    sr.symetricZS_.resize(1);
    sr.symetricZS_[0] = ps.getParameter<bool>("symetricZS");
    sr.srpLowInterestChannelZS_.resize(2);
    const int eb = 0;
    const int ee = 1;
    sr.srpLowInterestChannelZS_[eb] = ps.getParameter<double>("srpBarrelLowInterestChannelZS");
    sr.srpLowInterestChannelZS_[ee] = ps.getParameter<double>("srpEndcapLowInterestChannelZS");
    sr.srpHighInterestChannelZS_.resize(2);
    sr.srpHighInterestChannelZS_[eb] = ps.getParameter<double>("srpBarrelHighInterestChannelZS");
    sr.srpHighInterestChannelZS_[ee] = ps.getParameter<double>("srpEndcapHighInterestChannelZS");
    //sr.trigPrimBypass_.resize(1);
    //sr.trigPrimBypass_[0] = ps.getParameter<bool >("trigPrimBypass");
    //sr.trigPrimBypassMode_.resize(1);
    //sr.trigPrimBypassMode_[0] = ps.getParameter<int >("trigPrimBypassMode");
    //sr.trigPrimBypassLTH_.resize(1);
    //sr.trigPrimBypassLTH_[0] = ps.getParameter<double >("trigPrimBypassLTH");
    //sr.trigPrimBypassHTH_.resize(1);
    //sr.trigPrimBypassHTH_[0] = ps.getParameter<double >("trigPrimBypassHTH");
    //sr.trigPrimBypassWithPeakFinder_.resize(1);
    //sr.trigPrimBypassWithPeakFinder_[0] = ps.getParameter<bool >("trigPrimBypassWithPeakFinder");
    //sr.defaultTtf_.resize(1);
    //sr.defaultTtf_[0] = ps.getParameter<int >("defaultTtf");
    sr.actions_ = ps.getParameter<std::vector<int> >("actions");
  }

  void importSrpConfigFile(EcalSRSettings& sr, std::istream& f, bool d) {
    //initialize vectors:
    sr.deltaEta_ = vector<int>(1, 0);
    sr.deltaPhi_ = vector<int>(1, 0);
    sr.actions_ = vector<int>(4, 0);
    sr.tccMasksFromConfig_ = vector<short>(EcalSRSettings::nTccs_, 0);
    sr.srpMasksFromConfig_ = vector<vector<short> >(EcalSRSettings::nSrps_, vector<short>(8, 0));
    sr.dccMasks_ = vector<short>(EcalSRSettings::nDccs_);
    sr.srfMasks_ = vector<short>(EcalSRSettings::nDccs_);
    sr.substitutionSrfs_ = vector<vector<short> >(EcalSRSettings::nSrps_, vector<short>(68, 0));
    sr.testerTccEmuSrpIds_ = vector<int>(EcalSRSettings::nSrps_, 0);
    sr.testerSrpEmuSrpIds_ = vector<int>(EcalSRSettings::nSrps_, 0);
    sr.testerDccTestSrpIds_ = vector<int>(EcalSRSettings::nSrps_, 0);
    sr.testerSrpTestSrpIds_ = vector<int>(EcalSRSettings::nSrps_, 0);
    sr.bxOffsets_ = vector<short>(EcalSRSettings::nSrps_, 0);
    sr.automaticMasks_ = 0;
    sr.automaticSrpSelect_ = 0;

    //string line;
    int iLine = 0;
    int iValueSet = -1;
    const int nValueSets = 6 * EcalSRSettings::nSrps_ + 9;
    string line;
    stringstream sErr("");
    while (!f.eof() && sErr.str().empty()) {
      getline(f, line);
      ++iLine;
      line = trim(line);
      if (line[0] == '#' || line.empty()) {  //comment line and empty line to ignore
        continue;
      } else {
        ++iValueSet;
      }
      if (iValueSet >= nValueSets)
        break;
      uint32_t value;
      string sValue;
      int pos = 0;
      int iCh = 0;
      int nChs[nValueSets] = {
          //TCC masks: 0-11
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          //SRP masks: 12-23
          8,
          8,
          8,
          8,
          8,
          8,
          8,
          8,
          8,
          8,
          8,
          8,
          //DCC masks: 24-35
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          12,
          //SRF Masks: 36-47
          6,
          6,
          6,
          6,
          6,
          6,
          6,
          6,
          6,
          6,
          6,
          6,
          //substitution SRFs: 48-59
          68,
          68,
          68,
          68,
          68,
          68,
          68,
          68,
          68,
          68,
          68,
          68,
          //Tester card to emulate or test: 60-71
          4,
          4,
          4,
          4,
          4,
          4,
          4,
          4,
          4,
          4,
          4,
          4,
          //Bx offsets: 72
          12,
          //algo type: 73
          1,
          //action flags: 74
          4,
          //pattern file directory: 75
          1,
          //VME slots: 76
          12,
          //card types: 77
          12,
          //config Mode
          1,
          //VME Interface card
          1,
          //Spy Mode
          12,
      };

      while (((sValue = tokenize(line, " \t", pos)) != string("")) && (iCh < nChs[iValueSet]) && sErr.str().empty()) {
        value = strtoul(sValue.c_str(), nullptr, 0);
        const int iSrp = iValueSet % EcalSRSettings::nSrps_;
        if (iValueSet < 12) {  //TCC
          assert((unsigned)iSrp < sizeof(tccNum) / sizeof(tccNum[0]));
          assert((unsigned)iCh < sizeof(tccNum[0]) / sizeof(tccNum[0][0]));
          int tcc = tccNum[iSrp][iCh];
          if (tcc >= 0) {
            if (d)
              cout << "tccMasksFromConfig_[" << tcc << "] <- " << value << "\n";
            sr.tccMasksFromConfig_[tcc - 1] = value;
          }
        } else if (iValueSet < 24) {  //SRP-SRP
          if (d)
            cout << "srpMasks_[" << iSrp << "][" << iCh << "] <- " << value << "\n";
          sr.srpMasksFromConfig_[iSrp][iCh] = value;
        } else if (iValueSet < 36) {  //DCC output
          assert((unsigned)iSrp < sizeof(dccNum) / sizeof(dccNum[0]));
          assert((unsigned)iCh < sizeof(dccNum[0]) / sizeof(dccNum[0][0]));
          int dcc = dccNum[iSrp][iCh];
          if (dcc > 0) {
            assert((unsigned)(dcc - 1) < sr.dccMasks_.size());
            if (d)
              cout << "dccMasks_[" << (dcc - 1) << "] <- " << value << "\n";
            sr.dccMasks_[dcc - 1] = value;
          }
        } else if (iValueSet < 48) {  //SRF masks
          assert((unsigned)iSrp < sizeof(dccNum) / sizeof(dccNum[0]));
          assert((unsigned)iCh < sizeof(dccNum[0]) / sizeof(dccNum[0][0]));
          int dcc = dccNum[iSrp][iCh];
          if (dcc > 0) {
            if (d)
              cout << "srfMasks_[" << (dcc - 1) << "] <- " << value << "\n";
            assert((unsigned)(dcc - 1) < sr.srfMasks_.size());
            sr.srfMasks_[dcc - 1] = value;
          }
        } else if (iValueSet < 60) {  //substiution SRFs
          assert((unsigned)iSrp < sr.substitutionSrfs_.size());
          assert((unsigned)iCh < sr.substitutionSrfs_[0].size());
          if (d)
            cout << "substitutionMasks_[" << iSrp << "][" << iCh << "] <- " << value << "\n";
          sr.substitutionSrfs_[iSrp][iCh] = value;
        } else if (iValueSet < 72) {  //Tester card config
          switch (iCh) {
            case 0:
              assert((unsigned)iSrp < sr.testerTccEmuSrpIds_.size());
              if (d)
                cout << "testerTccEmuSrpIds_[" << iSrp << "] <- " << value << "\n";
              sr.testerTccEmuSrpIds_[iSrp] = value;
              break;
            case 1:
              assert((unsigned)iSrp < sr.testerSrpEmuSrpIds_.size());
              if (d)
                cout << "testerSrpEmuSrpIds_[" << iSrp << "] <- " << value << "\n";
              sr.testerSrpEmuSrpIds_[iSrp] = value;
              break;
            case 2:
              assert((unsigned)iSrp < sr.testerDccTestSrpIds_.size());
              if (d)
                cout << "testerDccTestSrpIds_[" << iSrp << "] <- " << value << "\n";
              sr.testerDccTestSrpIds_[iSrp] = value;
              break;
            case 3:
              assert((unsigned)iSrp < sr.testerSrpTestSrpIds_.size());
              if (d)
                cout << "testerSrpTestSrpIds_[" << iSrp << "] <- " << value << "\n";
              sr.testerSrpTestSrpIds_[iSrp] = value;
              break;
            default:
              sErr << "Syntax error in SRP system configuration "
                   << " line " << iLine << ".";
          }
        } else if (iValueSet < 73) {  //bx offsets
          assert((unsigned)iCh < sr.bxOffsets_.size());
          if (d)
            cout << "bxOffset_[" << iCh << "] <- " << value << "\n";
          sr.bxOffsets_[iCh] = value;
        } else if (iValueSet < 74) {  //algo type
          int algo = value;
          switch (algo) {
            case 0:
              sr.deltaEta_[0] = sr.deltaPhi_[0] = 1;
              break;
            case 1:
              sr.deltaEta_[0] = sr.deltaPhi_[0] = 2;
              break;
            default:
              throw cms::Exception("OutOfRange")
                  << "Value of parameter algo ," << algo << ", is invalid. Valid values are 0 and 1.";
          }
          if (d)
            cout << "deltaEta_[0] <- " << sr.deltaEta_[0] << "\t"
                 << "deltaPhi_[0] <- " << sr.deltaPhi_[0] << "\n";
        } else if (iValueSet < 75) {  //action flags
          assert((unsigned)iCh < sr.actions_.size());
          if (d)
            cout << "actions_[" << iCh << "] <- " << value << "\n";
          sr.actions_[iCh] = value;
        } else if (iValueSet < 76) {  //pattern file directory
                                      // 	emuDir_ = sValue;
                                      // 	if(d) cout << "emuDir_ <= "
                                      // 		   << value << "\n";
        } else if (iValueSet < 77) {  //VME slots
          // 	slotIds_[iCh] = value;
          // 	if(d) cout << "slotIds_[" << iCh << "] <= "
          // 		   << value << "\n";
        } else if (iValueSet < 78) {  //card types
          // 	cardTypes_[iCh] = sValue[0];
          // 	if(d) cout << "cardTypes_[" << iCh << "] <= "
          // 		   << value << "\n";
        } else if (iValueSet < 79) {  //config mode
                                      //TODO validity check on value
          // 	configMode_ = (ConfigMode)value;
          // 	if(d) cout << "config mode <= " << value << "\n";
        } else if (iValueSet < 80) {  //VME I/F
                                      //TODO validity check on value
                                      //     vmeInterface_ = (Vme::type_t)value;
                                      //if(d) cout << "Vme Interface code <= " << value << "\n";
        } else if (iValueSet < 81) {  //Spy Mode
                                      //TODO validity check on value
                                      //	spyMode_[iCh] = value & 0x3;
                                      //	if(d) cout << "Spy mode <= " << value << "\n";
        } else {                      //should never be reached!
          assert(false);
        }
        ++iCh;
      }
      if (iCh != nChs[iValueSet]) {  //error
        sErr << "Syntax error in imported SRP system configuration file "
                /*<< filename <<*/ " line "
             << iLine << ".";
      }
    }
    if (sErr.str().empty() && iValueSet != (nValueSets - 1)) {  //error
      sErr << "Syntax Error in imported SRP system configuration file "
              /*<< filename <<*/ " line "
           << iLine << ".";
    }
    if (!sErr.str().empty())
      throw cms::Exception("SyntaxError") << sErr.str();
  }

  double normalizeWeights(int hwWeight) {
    //Fix sign bit in case only the 12 least significant bits of hwWeight were set
    //(hardware reprensentation uses only 12 bits)
    if (hwWeight & (1 << 11))
      hwWeight |= ~0xEFF;
    return hwWeight / 1024.;
  }

}  // namespace ecalsrcondtools
