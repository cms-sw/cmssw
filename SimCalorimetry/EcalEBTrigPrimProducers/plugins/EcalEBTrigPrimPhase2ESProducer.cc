
// user include files
#include "EcalEBTrigPrimPhase2ESProducer.h"

#include <TMath.h>
#include <fstream>
#include <iostream>
#include <sstream>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//
// input stream from a gz file
//

struct GzInputStream {
  gzFile gzf;
  char buffer[256];
  std::istringstream iss;
  bool eof;
  GzInputStream(const char *file) : eof(false) {
    gzf = gzopen(file, "rb");
    edm::LogInfo("EcalEBTrigPrimPhase2ESProducer") << " New weight file " << file;
    if (gzf == Z_NULL) {
      eof = true;
      edm::LogWarning("EcalEBTrigPrimPhase2ESProducer") << "Database file " << file << " not found!!!";
    } else
      readLine();
  }
  void readLine() {
    char *res = gzgets(gzf, buffer, 256);
    eof = (res == Z_NULL);
    if (!eof) {
      iss.clear();
      iss.str(buffer);
    }
  }
  ~GzInputStream() { gzclose(gzf); }
  explicit operator bool() const { return ((eof == true) ? false : !iss.fail()); }
};

template <typename T>
GzInputStream &operator>>(GzInputStream &gis, T &var) {
  while ((bool)gis && !(gis.iss >> var)) {
    gis.readLine();
  }
  return gis;
}

//
// constructors and destructor
//

EcalEBTrigPrimPhase2ESProducer::EcalEBTrigPrimPhase2ESProducer(const edm::ParameterSet &iConfig)
    : dbFilename_(iConfig.getUntrackedParameter<std::string>("DatabaseFile", "")),
      configFilename_(iConfig.getParameter<edm::FileInPath>("WeightTextFile")),
      flagPrint_(iConfig.getParameter<bool>("WriteInFile")) {
  parseWeightsFile();

  // the following lines are needed to tell the framework what
  // data is being produced
  setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceLinearizationConst);
  setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::producePedestals);
  setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceAmpWeight);
  setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceTimeWeight);
  setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceWeightGroup);
  setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceBadX);
  //setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::producePhysicsConst);
  //setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceBadStrip);
  //setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceBadTT);
  //setWhatProduced(this, &EcalEBTrigPrimPhase2ESProducer::produceSpike);
}

EcalEBTrigPrimPhase2ESProducer::~EcalEBTrigPrimPhase2ESProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------

std::unique_ptr<EcalEBPhase2TPGPedestalsMap> EcalEBTrigPrimPhase2ESProducer::producePedestals(
    const EcalEBPhase2TPGPedestalsRcd &iRecord) {
  auto prod = std::make_unique<EcalEBPhase2TPGPedestalsMap>();

  std::map<int, std::vector<unsigned int>>::const_iterator it;
  for (it = mapXtalToLin_.begin(); it != mapXtalToLin_.end(); it++) {
    EBDetId myEBDetId = EBDetId(it->first);
    EcalEBPhase2TPGPedestal ped;

    ped.mean_x10 = (it->second)[0];
    ped.mean_x1 = (it->second)[3];
    prod->insert(std::make_pair(myEBDetId, ped));
  }

  return prod;
}

std::unique_ptr<EcalEBPhase2TPGLinearizationConst> EcalEBTrigPrimPhase2ESProducer::produceLinearizationConst(
    const EcalEBPhase2TPGLinearizationConstRcd &iRecord) {
  auto prod = std::make_unique<EcalEBPhase2TPGLinearizationConst>();

  std::map<int, std::vector<unsigned int>>::const_iterator it;
  for (it = mapXtalToLin_.begin(); it != mapXtalToLin_.end(); it++) {
    EcalEBPhase2TPGLinearizationConstant param;

    param.mult_x10 = (it->second)[1];
    param.mult_x1 = (it->second)[5];
    param.shift_x10 = (it->second)[2];
    param.shift_x1 = (it->second)[6];
    param.i2cSub_x10 = (it->second)[3];
    param.i2cSub_x1 = (it->second)[7];
    prod->setValue(it->first, param);
  }

  return prod;
}

std::unique_ptr<EcalEBPhase2TPGAmplWeightIdMap> EcalEBTrigPrimPhase2ESProducer::produceAmpWeight(
    const EcalEBPhase2TPGAmplWeightIdMapRcd &iRecord) {
  auto prod = std::make_unique<EcalEBPhase2TPGAmplWeightIdMap>();

  EcalEBPhase2TPGAmplWeights weights;
  std::map<uint32_t, std::vector<uint32_t>>::const_iterator it;
  for (it = mapWeight_.begin(); it != mapWeight_.end(); it++) {
    weights.setValues((it->second)[0],
                      (it->second)[1],
                      (it->second)[2],
                      (it->second)[3],
                      (it->second)[4],
                      (it->second)[5],
                      (it->second)[6],
                      (it->second)[7],
                      (it->second)[8],
                      (it->second)[9],
                      (it->second)[10],
                      (it->second)[11]);
    prod->setValue(it->first, weights);
  }

  return prod;
}

std::unique_ptr<EcalEBPhase2TPGTimeWeightIdMap> EcalEBTrigPrimPhase2ESProducer::produceTimeWeight(
    const EcalEBPhase2TPGTimeWeightIdMapRcd &iRecord) {
  auto prod = std::make_unique<EcalEBPhase2TPGTimeWeightIdMap>();

  EcalEBPhase2TPGTimeWeights weights_time;
  std::map<uint32_t, std::vector<uint32_t>>::const_iterator it;
  for (it = mapTimeWeight_.begin(); it != mapTimeWeight_.end(); it++) {
    weights_time.setValues((it->second)[0],
                           (it->second)[1],
                           (it->second)[2],
                           (it->second)[3],
                           (it->second)[4],
                           (it->second)[5],
                           (it->second)[6],
                           (it->second)[7],
                           (it->second)[8],
                           (it->second)[9],
                           (it->second)[10],
                           (it->second)[11]);
    prod->setValue(it->first, weights_time);
  }

  return prod;
}

std::unique_ptr<EcalTPGWeightGroup> EcalEBTrigPrimPhase2ESProducer::produceWeightGroup(
    const EcalTPGWeightGroupRcd &iRecord) {
  auto prod = std::make_unique<EcalTPGWeightGroup>();

  const int NGROUPS = 61200;

  for (int iGroup = 0; iGroup < NGROUPS; iGroup++) {
    std::map<int, std::vector<unsigned int>>::const_iterator it;
    for (it = mapXtalToGroup_.begin(); it != mapXtalToGroup_.end(); it++) {
      prod->setValue(it->first, it->second[0]);
    }
  }

  return prod;
}

std::unique_ptr<EcalTPGPhysicsConst> EcalEBTrigPrimPhase2ESProducer::producePhysicsConst(
    const EcalTPGPhysicsConstRcd &iRecord) {
  auto prod = std::make_unique<EcalTPGPhysicsConst>();
  // EcalEBTrigPrimPhase2ESProducer::producePhysicsConst Needs updating if we want to keep it

  parseTextFile();
  std::map<uint32_t, std::vector<float>>::const_iterator it;
  for (it = mapPhys_.begin(); it != mapPhys_.end(); it++) {
    EcalTPGPhysicsConst::Item item;
    item.EtSat = (it->second)[0];
    item.ttf_threshold_Low = (it->second)[1];
    item.ttf_threshold_High = (it->second)[2];
    item.FG_lowThreshold = (it->second)[3];
    item.FG_highThreshold = (it->second)[4];
    item.FG_lowRatio = (it->second)[5];
    item.FG_highRatio = (it->second)[6];
    prod->setValue(it->first, item);
  }

  return prod;
}

std::unique_ptr<EcalTPGCrystalStatus> EcalEBTrigPrimPhase2ESProducer::produceBadX(
    const EcalTPGCrystalStatusRcd &iRecord) {
  auto prod = std::make_unique<EcalTPGCrystalStatus>();

  parseTextFile();
  std::map<uint32_t, std::vector<uint32_t>>::const_iterator it;
  for (it = mapXtal_.begin(); it != mapXtal_.end(); it++) {
    EcalTPGCrystalStatusCode badXValue;
    badXValue.setStatusCode(0);
    prod->setValue(it->first, badXValue);
  }
  return prod;
}

void EcalEBTrigPrimPhase2ESProducer::parseWeightsFile() {
  uint32_t id;
  std::string dataCard;
  std::vector<unsigned int> param;

  int data;
  std::string filename = configFilename_.fullPath();;
  std::string finalFileName;
  size_t slash = filename.find('/');
  if (slash != 0) {
    edm::FileInPath fileInPath(filename);
    finalFileName = fileInPath.fullPath();
  } else {
    finalFileName = filename;;
    edm::LogWarning("EcalEBTPGESProducer")
        << "Couldnt find database file via fileinpath trying with pathname directly!!";
  }

  GzInputStream gis(finalFileName.c_str());
  while (gis >> dataCard) {
    if (dataCard == "WEIGHTAMP") {
      gis >> std::dec >> id;

      if (flagPrint_) {
        std::cout << dataCard << " " << std::dec << id << std::endl;
      }

      param.clear();

      std::string st6;
      for (int i = 0; i < 12; i++) {
        gis >> std::hex >> data;
        param.push_back(data);
        /// debug

        if (flagPrint_) {
          std::ostringstream oss;
          oss << std::hex << data;
          std::string result4 = oss.str();

          st6.append("0x");
          st6.append(result4);
          st6.append(" ");
        }
      }

      // debug
      if (flagPrint_) {
        std::cout << st6 << std::endl;
        std::cout << std::endl;
      }

      // std::cout << " WEIGHTAMP id " << id << std::endl;
      mapWeight_[id] = param;
    }

    if (dataCard == "WEIGHTTIME") {
      gis >> std::dec >> id;

      if (flagPrint_) {
        std::cout << dataCard << " " << std::dec << id << std::endl;
      }

      param.clear();

      std::string st6;
      for (int i = 0; i < 12; i++) {
        gis >> std::hex >> data;
        //std::cout << " Parse time weight filling data " << data;
        param.push_back(data);
        /// debug

        if (flagPrint_) {
          std::ostringstream oss;
          oss << std::hex << data;
          std::string result4 = oss.str();

          st6.append("0x");
          st6.append(result4);
          st6.append(" ");
        }
      }

      // debug
      if (flagPrint_) {
        std::cout << st6 << std::endl;
        std::cout << std::endl;
      }
      mapTimeWeight_[id] = param;
    }

    if (dataCard == "CRYSTAL") {
      gis >> std::dec >> id;

      if (flagPrint_) {
        std::cout << dataCard << " " << std::dec << id << std::endl;
      }

      param.clear();
      std::string st6;
      gis >> std::dec >> data;
      param.push_back(data);

      if (flagPrint_) {
        std::ostringstream oss;
        oss << std::dec << data;
        std::string result4 = oss.str();
        st6.append(result4);
        st6.append(" ");
        std::cout << st6 << std::endl;
        std::cout << std::endl;
      }
      mapXtalToGroup_[id] = param;
    }

    if (dataCard == "LINCONST") {
      gis >> std::dec >> id;

      if (flagPrint_) {
        std::cout << dataCard << " " << std::dec << id << std::endl;
      }

      param.clear();
      std::string st6;
      std::string st7;

      for (int i = 0; i < 8; i++) {
        gis >> std::hex >> data;
        param.push_back(data);

        if (flagPrint_) {
          if (i < 4) {
            std::ostringstream oss;
            oss << std::hex << data;
            std::string result6 = oss.str();
            st6.append("0x");
            st6.append(result6);
            if (i != 3)
              st6.append(" ");
          } else if (i >= 4 && i < 8) {
            std::ostringstream oss;
            oss << std::hex << data;
            std::string result7 = oss.str();
            st7.append("0x");
            st7.append(result7);
            if (i != 7)
              st7.append(" ");
          }
        }
      }
      if (flagPrint_) {
        std::cout << st6 << std::endl;
        std::cout << st7 << std::endl;
      }
      mapXtalToLin_[id] = param;
    }
  }
}

void EcalEBTrigPrimPhase2ESProducer::parseTextFile() {
  if (!mapXtal_.empty())
    return;  // just parse the file once!

  uint32_t id;
  std::string dataCard;
  std::string line;
  std::ifstream infile;
  std::vector<unsigned int> param;
  std::vector<float> paramF;
  int NBstripparams[2] = {4, 4};
  unsigned int data;

  std::string bufString;
  std::string iString;
  std::string fString;
  std::string filename = "SimCalorimetry/EcalTrigPrimProducers/data/" + dbFilename_;
  std::string finalFileName;
  size_t slash = dbFilename_.find('/');
  if (slash != 0) {
    edm::FileInPath fileInPath(filename);
    finalFileName = fileInPath.fullPath();
  } else {
    finalFileName = dbFilename_;
    edm::LogWarning("EcalTPG") << "Couldnt find database file via fileinpath, "
                                  "trying with pathname directly!!";
  }

  int k = 0;

  GzInputStream gis(finalFileName.c_str());
  while (gis >> dataCard) {
    if (dataCard == "CRYSTAL") {
      gis >> std::dec >> id;
      // std::cout<<dataCard<<" "<<std::dec<<id;
      std::string st3;
      std::string st4;
      std::string st5;

      if (flagPrint_) {
        // Print this comment only one time
        if (k == 0)
          std::cout << "COMMENT ====== barrel crystals ====== " << std::endl;

        if (k == 61200)
          std::cout << "COMMENT ====== endcap crystals ====== " << std::endl;

        k = k + 1;

        std::cout << dataCard << " " << std::dec << id << std::endl;
      }

      param.clear();
      for (int i = 0; i < 9; i++) {
        gis >> std::hex >> data;
        // std::cout<<", "<<std::hex<<data ;
        param.push_back(data);

        if (flagPrint_) {
          if (i < 3) {
            std::ostringstream oss;
            oss << std::hex << data;
            std::string result1 = oss.str();

            st3.append("0x");
            st3.append(result1);
            if (i != 2)
              st3.append(" ");

          } else if (i > 2 && i < 6) {
            std::ostringstream oss;
            oss << std::hex << data;
            std::string result2 = oss.str();

            st4.append("0x");
            st4.append(result2);
            if (i != 5)
              st4.append(" ");
          } else if (i > 5 && i < 9) {
            std::ostringstream oss;
            oss << std::hex << data;
            std::string result3 = oss.str();

            st5.append("0x");
            st5.append(result3);
            if (i != 8)
              st5.append(" ");
          }
        }

      }  // end for

      if (flagPrint_) {
        std::cout << " " << st3 << std::endl;
        std::cout << " " << st4 << std::endl;
        std::cout << " " << st5 << std::endl;
      }

      mapXtal_[id] = param;
    }

    if (dataCard == "STRIP_EB") {
      gis >> std::dec >> id;

      std::string st1;

      if (flagPrint_)
        std::cout << dataCard << " " << std::dec << id << std::endl;

      param.clear();
      for (int i = 0; i < NBstripparams[0]; i++) {
        gis >> std::hex >> data;
        // std::cout << " data = " << data << std::endl;
        param.push_back(data);

        if (flagPrint_) {
          if (i == 0) {
            std::cout << "0x" << std::hex << data << std::endl;
          } else if (i == 1) {
            std::cout << "" << std::hex << data << std::endl;
          } else if (i > 1) {
            std::ostringstream oss;
            if (i == 2) {
              oss << "0x" << std::hex << data;
              std::string result4 = oss.str();
              st1.append(result4);
            } else if (i == 3) {
              std::ostringstream oss;
              oss << " 0x" << std::hex << data;
              std::string result5 = oss.str();

              st1.append(result5);
              std::cout << "" << st1 << std::endl;
            }
          }
        }
      }

      mapStrip_[0][id] = param;
    }

    if (dataCard == "STRIP_EE") {
      gis >> std::dec >> id;

      std::string st6;

      if (flagPrint_) {
        std::cout << dataCard << " " << std::dec << id << std::endl;
      }

      param.clear();
      for (int i = 0; i < NBstripparams[1]; i++) {
        gis >> std::hex >> data;
        param.push_back(data);

        if (flagPrint_) {
          if (i == 0) {
            std::cout << "0x" << std::hex << data << std::endl;
          } else if (i == 1) {
            std::cout << " " << std::hex << data << std::endl;
          } else if (i > 1) {
            std::ostringstream oss;
            if (i == 2) {
              oss << "0x" << std::hex << data;
              std::string result4 = oss.str();
              st6.append(result4);
            } else if (i == 3) {
              std::ostringstream oss;
              oss << " 0x" << std::hex << data;
              std::string result5 = oss.str();

              st6.append(result5);
              std::cout << "" << st6 << std::endl;
            }
          }
        }
      }

      mapStrip_[1][id] = param;
    }

    if (dataCard == "TOWER_EE") {
      gis >> std::dec >> id;

      if (flagPrint_)
        std::cout << dataCard << " " << std::dec << id << std::endl;

      param.clear();
      for (int i = 0; i < 2; i++) {
        gis >> std::hex >> data;
        param.push_back(data);

        if (flagPrint_) {
          if (i == 1) {
            std::cout << "0x" << std::dec << data << std::endl;
          } else {
            std::cout << " " << std::dec << data << std::endl;
          }
        }
      }

      // std::cout<<std::endl ;
      mapTower_[1][id] = param;
    }

    if (dataCard == "TOWER_EB") {
      gis >> std::dec >> id;

      if (flagPrint_)
        std::cout << dataCard << " " << std::dec << id << std::endl;

      param.clear();
      for (int i = 0; i < 3; i++) {
        gis >> std::dec >> data;

        if (flagPrint_) {
          std::cout << " " << std::dec << data << std::endl;
        }

        param.push_back(data);
      }

      // std::cout<<std::endl ;
      mapTower_[0][id] = param;
    }
  }
}

std::vector<int> EcalEBTrigPrimPhase2ESProducer::getRange(
    int subdet, int tccNb, int towerNbInTcc, int stripNbInTower, int xtalNbInStrip) {
  std::vector<int> range;
  if (subdet == 0) {
    // Barrel
    range.push_back(37);  // stccNbMin
    range.push_back(73);  // tccNbMax
    range.push_back(1);   // towerNbMin
    range.push_back(69);  // towerNbMax
    range.push_back(1);   // stripNbMin
    range.push_back(6);   // stripNbMax
    range.push_back(1);   // xtalNbMin
    range.push_back(6);   // xtalNbMax
  } else {
    // Endcap eta >0
    if (subdet > 0) {
      range.push_back(73);   // tccNbMin
      range.push_back(109);  // tccNbMax
    } else {                 // endcap eta <0
      range.push_back(1);    // tccNbMin
      range.push_back(37);   // tccNbMax
    }
    range.push_back(1);   // towerNbMin
    range.push_back(29);  // towerNbMax
    range.push_back(1);   // stripNbMin
    range.push_back(6);   // stripNbMax
    range.push_back(1);   // xtalNbMin
    range.push_back(6);   // xtalNbMax
  }

  if (tccNb > 0) {
    range[0] = tccNb;
    range[1] = tccNb + 1;
  }
  if (towerNbInTcc > 0) {
    range[2] = towerNbInTcc;
    range[3] = towerNbInTcc + 1;
  }
  if (stripNbInTower > 0) {
    range[4] = stripNbInTower;
    range[5] = stripNbInTower + 1;
  }
  if (xtalNbInStrip > 0) {
    range[6] = xtalNbInStrip;
    range[7] = xtalNbInStrip + 1;
  }

  return range;
}

// bool EcalEBTrigPrimPhase2ESProducer::getNextString(gzFile &gzf){
//   size_t blank;
//   if (bufpos_==0) {
//     gzgets(gzf,buf_,80);
//     if (gzeof(gzf)) return true;
//     bufString_=std::string(buf_);
//   }
//   int  pos=0;
//   pos =bufpos_;
//   // look for next non-blank
//   while (pos<bufString_.size()) {
//     if (!bufString_.compare(pos,1," ")) pos++;
//     else break;
//   }
//   blank=bufString_.find(" ",pos);
//   size_t end = blank;
//   if (blank==std::string::npos) end=bufString_.size();
//   sub_=bufString_.substr(pos,end-pos);
//   bufpos_= blank;
//   if (blank==std::string::npos) bufpos_=0;
//   return false;
// }
//
// int EcalEBTrigPrimPhase2ESProducer::converthex() {
//   // converts hex dec string sub to hexa
//   //FIXME:: find something better (istrstream?)!!!!
//
//   std::string chars("0123456789abcdef");
//   int hex=0;
//   for (size_t i=2;i<sub_.length();++i) {
//     size_t f=chars.find(sub_[i]);
//     if (f==std::string::npos) break;  //FIXME: length is 4 for 0x3!!
//     hex=hex*16+chars.find(sub_[i]);
//   }
//   return hex;
// }
