// user include files
#include "DataFormats/EcalDigi/interface/EcalConstants.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
#include "CondFormats/DataRecord/interface/EcalTPGCrystalStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGPhysicsConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGStripStatusRcd.h"
// commented lines are for a reminder that in future we might need to implement something alike
//#include "CondFormats/DataRecord/interface/EcalTPGSpikeRcd.h"
//#include "CondFormats/DataRecord/interface/EcalTPGTowerStatusRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGLinearizationConstRcd.h"
#include "CondFormats/DataRecord/interface/EcalTPGWeightGroupRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGAmplWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGTimeWeightIdMapRcd.h"
#include "CondFormats/DataRecord/interface/EcalEBPhase2TPGPedestalsRcd.h"
#include "CondFormats/EcalObjects/interface/EcalTPGCrystalStatus.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGLinearizationConst.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalTPGPhysicsConst.h"
//#include "CondFormats/EcalObjects/interface/EcalTPGSpike.h"
//#include "CondFormats/EcalObjects/interface/EcalTPGTowerStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGStripStatus.h"
#include "CondFormats/EcalObjects/interface/EcalTPGWeightGroup.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGAmplWeightIdMap.h"
#include "CondFormats/EcalObjects/interface/EcalEBPhase2TPGTimeWeightIdMap.h"

#include "zlib.h"
#include <TMath.h>
#include <fstream>
#include <iostream>
#include <sstream>

//
// class declaration
//

/** \class EcalEBTrigPrimPhase2ESProducer
\author L. Lutton, N. Marinelli - Univ. of Notre Dame
 Description: forPhase II
*/

class EcalEBTrigPrimPhase2ESProducer : public edm::ESProducer {
public:
  EcalEBTrigPrimPhase2ESProducer(const edm::ParameterSet &);
  ~EcalEBTrigPrimPhase2ESProducer() override;

  std::unique_ptr<EcalEBPhase2TPGLinearizationConst> produceLinearizationConst(
      const EcalEBPhase2TPGLinearizationConstRcd &);
  std::unique_ptr<EcalEBPhase2TPGPedestalsMap> producePedestals(const EcalEBPhase2TPGPedestalsRcd &);
  std::unique_ptr<EcalEBPhase2TPGAmplWeightIdMap> produceAmpWeight(const EcalEBPhase2TPGAmplWeightIdMapRcd &);
  std::unique_ptr<EcalEBPhase2TPGTimeWeightIdMap> produceTimeWeight(const EcalEBPhase2TPGTimeWeightIdMapRcd &);
  std::unique_ptr<EcalTPGWeightGroup> produceWeightGroup(const EcalTPGWeightGroupRcd &);
  std::unique_ptr<EcalTPGPhysicsConst> producePhysicsConst(const EcalTPGPhysicsConstRcd &);
  std::unique_ptr<EcalTPGCrystalStatus> produceBadX(const EcalTPGCrystalStatusRcd &);

  // These commented lines are a reminder that in the future we might need to implement something alike
  //std::unique_ptr<EcalTPGLutGroup> produceLutGroup(const EcalTPGLutGroupRcd &);
  //std::uniq//std::unique_ptr<EcalTPGStripStatus> produceBadStrip(const EcalTPGStripStatusRcd &);
  //std::unique_ptr<EcalTPGTowerStatus> produceBadTT(const EcalTPGTowerStatusRcd &);
  //std::unique_ptr<EcalTPGSpike> produceSpike(const EcalTPGSpikeRcd &);

private:
  void parseTextFile();
  std::vector<int> getRange(int subdet, int smNb, int towerNbInSm, int stripNbInTower = 0, int xtalNbInStrip = 0);
  void parseWeightsFile();

  // ----------member data ---------------------------
  std::string dbFilename_;
  //  std::string configFilename_;
  const edm::FileInPath configFilename_;
  bool flagPrint_;
  std::map<uint32_t, std::vector<uint32_t>> mapXtal_;
  std::map<uint32_t, std::vector<uint32_t>> mapStrip_[2];
  std::map<uint32_t, std::vector<uint32_t>> mapTower_[2];
  std::map<uint32_t, std::vector<uint32_t>> mapWeight_;
  std::map<uint32_t, std::vector<uint32_t>> mapTimeWeight_;
  std::map<int, std::vector<unsigned int>> mapXtalToGroup_;
  std::map<int, std::vector<unsigned int>> mapXtalToLin_;
  std::map<uint32_t, std::vector<float>> mapPhys_;
  static const int maxSamplesUsed_;
  static const int nLinConst_;
};

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

const int EcalEBTrigPrimPhase2ESProducer::maxSamplesUsed_ = 12;
const int EcalEBTrigPrimPhase2ESProducer::nLinConst_ = 8;

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
  // the following commented lines as a reminder for items which might need to be implemented for Phase2
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
  std::string filename = configFilename_.fullPath();
  ;
  std::string finalFileName;
  size_t slash = filename.find('/');
  if (slash != 0) {
    edm::FileInPath fileInPath(filename);
    finalFileName = fileInPath.fullPath();
  } else {
    finalFileName = filename;
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
      for (int i = 0; i < maxSamplesUsed_; i++) {
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
      for (int i = 0; i < maxSamplesUsed_; i++) {
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

      for (int i = 0; i < nLinConst_; i++) {
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
          } else if (i < 8) {
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

      mapTower_[0][id] = param;
    }
  }
}

/// This method is not used at all, however is a reminder that something alike will probably be needed once the mapping EB to BCPs will be in place
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

DEFINE_FWK_EVENTSETUP_MODULE(EcalEBTrigPrimPhase2ESProducer);
