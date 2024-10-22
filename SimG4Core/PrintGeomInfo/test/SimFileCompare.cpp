////////////////////////////////////////////////////////////////////////////////
//
//    Compares output files from PrintGeomInfo created using DDD and DD4hep
//    inputs. Usage:
//
//    SimFileCompare infile1 infile2 type files debug
//    infile1  (const char*)   First file name
//    infile2  (const char*)   Second file name
//    type     (int)           Type of file: material (0), solid (1),
//                             LogicalVolume (2), PhysicalVolume (3);
//                             Region (4)
//    files    (int)           Double digits each inidicating the file source
//                             (0 for DDD, 1 for DD4hep). So if first file is
//                             DDD and second is DD4hep, it will be 10
//    debug    (int)           Single digit number (0 minimum printout)
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <cstdint>
#include "SimG4Core/Geometry/interface/DD4hep2DDDName.h"

struct materials {
  int occ;
  double radl, intl;
  materials(int oc = 1, double rd = 0, double in = 0) : occ(oc), radl(rd), intl(in) {}
};

struct solids {
  int occ;
  double volume;
  solids(int oc = 1, double vol = 0) : occ(oc), volume(vol) {}
};

struct lvs {
  int occ;
  double mass;
  lvs(int oc = 1, double m = 0) : occ(oc), mass(m) {}
};

struct pvs {
  int occ;
  double xx, yy, zz;
  pvs(int oc = 1, double x = 0, double y = 0, double z = 0) : occ(oc), xx(x), yy(y), zz(z) {}
};

struct regions {
  int occ;
  double nmat, nvol;
  regions(int oc = 1, double mat = 0, double vol = 0) : occ(oc), nmat(mat), nvol(vol) {}
};

std::string removeExtraName(const std::string& name, int debug) {
  std::string nam(name);
  std::string nam1 = name.substr(0, 2);
  if (((nam1 == "GE") || (nam1 == "GH") || (nam1 == "MB") || (nam1 == "ME") || (nam1 == "RE") || (nam1 == "RR") ||
       (nam1 == "RT")) &&
      (name.size() > 5)) {
    uint32_t loc = name.size() - 5;
    if ((name.substr(0, 15) != "MBCables_Wheels") && (name.substr(loc, 1) == "_")) {
      std::string nam2 = (name.substr(loc + 3, 1) == "0") ? name.substr(loc + 4, 1) : name.substr(loc + 3, 2);
      nam = name.substr(0, loc + 1) + nam2;
    }
  }
  if (debug)
    std::cout << name << " : " << nam1 << " " << nam << std::endl;
  return nam;
}

std::string reducedName(const std::string& name, int debug) {
  std::string nam(name);
  uint32_t first = ((name.find(":") == std::string::npos) ? 0 : (name.find(":") + 1));
  uint32_t last(name.size() + 1);
  uint32_t loc(first);
  while (1) {
    if (name.find("_", loc) == std::string::npos)
      break;
    if (((loc + 5) < name.size()) && (name.substr(loc, 5) == "shape")) {
      last = loc;
      break;
    }
    loc = name.find("_", loc) + 1;
    if (loc > name.size())
      break;
  }
  nam = name.substr(first, last - first - 1);
  if ((last < name.size()) && (name.substr(name.size() - 5, 5) == "_refl"))
    nam += "_refl";
  if (debug > 0)
    std::cout << name << " col " << first << ":" << last << " " << nam << std::endl;
  return nam;
}

std::vector<std::string> splitString(const std::string& fLine) {
  std::vector<std::string> result;
  int start = 0;
  bool empty = true;
  for (unsigned i = 0; i <= fLine.size(); i++) {
    if (fLine[i] == ' ' || i == fLine.size()) {
      if (!empty) {
        std::string item(fLine, start, i - start);
        result.push_back(item);
        empty = true;
      }
      start = i + 1;
    } else {
      if (empty)
        empty = false;
    }
  }
  return result;
}

template <typename T>
void myPrint1(std::map<std::string, T> const& obj) {
  for (auto it : obj) {
    if (it.second.occ > 1)
      std::cout << it.first << " : " << it.second.occ << std::endl;
  }
}

template <typename T>
void myPrint2(std::map<std::string, T> const& obj1, std::map<std::string, T> const& obj2) {
  for (auto it : obj1) {
    if (obj2.find(it.first) == obj2.end())
      std::cout << it.first << " appearing " << it.second.occ << " times" << std::endl;
  }
}

void CompareFiles(const char* fileFile1, const char* fileFile2, int type, int files, int debug) {
  std::map<std::string, materials> matFile1, matFile2;
  std::map<std::string, solids> solidFile1, solidFile2;
  std::map<std::string, lvs> lvFile1, lvFile2;
  std::map<std::string, pvs> pvFile1, pvFile2;
  std::map<std::string, regions> regFile1, regFile2;
  bool typeFile1 = ((files % 10) == 0);
  bool typeFile2 = (((files / 10) % 10) == 0);
  char buffer[100];
  std::string name;
  std::ifstream fInput1(fileFile1);
  unsigned int sizeFile1(0), sizeFile2(0);
  if (!fInput1.good()) {
    std::cout << "Cannot open file " << fileFile1 << std::endl;
  } else {
    while (fInput1.getline(buffer, 100)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      if ((type == 0) || (type == 2))
        name = DD4hep2DDDName::nameMatterLV(items[0], !typeFile1);
      else if (type == 1)
        name = DD4hep2DDDName::nameSolid(items[0], !typeFile1);
      else if (type == 3)
        name = DD4hep2DDDName::namePV(items[0], !typeFile1);
      else
        name = items[0];
      double r1 = (items.size() > 1) ? atof(items[1].c_str()) : 0;
      double r2 = (items.size() > 2) ? atof(items[2].c_str()) : 0;
      double r3 = (items.size() > 3) ? atof(items[3].c_str()) : 0;
      if (type == 0) {
        auto it = matFile1.find(name);
        if (it == matFile1.end())
          matFile1[name] = materials(1, r1, r2);
        else
          ++((it->second).occ);
      } else if (type == 1) {
        auto it = solidFile1.find(name);
        if (it == solidFile1.end())
          solidFile1[name] = solids(1, r1);
        else
          ++((it->second).occ);
      } else if (type == 2) {
        auto it = lvFile1.find(name);
        if (it == lvFile1.end())
          lvFile1[name] = lvs(1, r1);
        else
          ++((it->second).occ);
      } else if (type == 3) {
        auto it = pvFile1.find(name);
        if (it == pvFile1.end())
          pvFile1[name] = pvs(1, r1, r2, r3);
        else
          ++((it->second).occ);
      } else {
        auto it = regFile1.find(name);
        if (it == regFile1.end())
          regFile1[name] = regions(1, r1, r2);
        else
          ++((it->second).occ);
      }
    }
    fInput1.close();
    sizeFile1 = ((type == 0) ? matFile1.size()
                             : ((type == 1) ? solidFile1.size() : ((type == 2) ? lvFile1.size() : pvFile1.size())));
  }
  std::ifstream fInput2(fileFile2);
  if (!fInput2.good()) {
    std::cout << "Cannot open file " << fileFile2 << std::endl;
  } else {
    while (fInput2.getline(buffer, 100)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      if ((type == 0) || (type == 2))
        name = DD4hep2DDDName::nameMatterLV(items[0], !typeFile2);
      else if (type == 1)
        name = DD4hep2DDDName::nameSolid(items[0], !typeFile2);
      else if (type == 3)
        name = DD4hep2DDDName::namePV(items[0], !typeFile2);
      else
        name = items[0];
      double r1 = (items.size() > 1) ? atof(items[1].c_str()) : 0;
      double r2 = (items.size() > 2) ? atof(items[2].c_str()) : 0;
      double r3 = (items.size() > 3) ? atof(items[3].c_str()) : 0;
      if (type == 0) {
        auto it = matFile2.find(name);
        if (it == matFile2.end())
          matFile2[name] = materials(1, r1, r2);
        else
          ++((it->second).occ);
      } else if (type == 1) {
        auto it = solidFile2.find(name);
        if (it == solidFile2.end())
          solidFile2[name] = solids(1, r1);
        else
          ++((it->second).occ);
      } else if (type == 2) {
        auto it = lvFile2.find(name);
        if (it == lvFile2.end())
          lvFile2[name] = lvs(1, r1);
        else
          ++((it->second).occ);
      } else if (type == 3) {
        auto it = pvFile2.find(name);
        if (it == pvFile2.end())
          pvFile2[name] = pvs(1, r1, r2, r3);
        else
          ++((it->second).occ);
      } else {
        auto it = regFile2.find(name);
        if (it == regFile2.end())
          regFile2[name] = regions(1, r1, r2);
        else
          ++((it->second).occ);
      }
    }
    fInput2.close();
    sizeFile2 = ((type == 0) ? matFile2.size()
                             : ((type == 1) ? solidFile2.size() : ((type == 2) ? lvFile2.size() : pvFile2.size())));
  }
  std::cout << "Reads " << sizeFile1 << " names from " << fileFile1 << " and " << sizeFile2 << " names from "
            << fileFile2 << std::endl;

  std::cout << "\nMore than one entry for a given name in " << fileFile1 << std::endl;
  if (type == 0) {
    myPrint1(matFile1);
  } else if (type == 1) {
    myPrint1(solidFile1);
  } else if (type == 2) {
    myPrint1(lvFile1);
  } else if (type == 3) {
    myPrint1(pvFile1);
  } else {
    myPrint1(regFile1);
  }

  std::cout << "\nMore than one entry for a given name in " << fileFile2 << std::endl;
  if (type == 0) {
    myPrint1(matFile2);
  } else if (type == 1) {
    myPrint1(solidFile2);
  } else if (type == 2) {
    myPrint1(lvFile2);
  } else if (type == 3) {
    myPrint1(pvFile2);
  } else {
    myPrint1(regFile2);
  }

  std::cout << "\nEntry in " << fileFile1 << " not in " << fileFile2 << std::endl;
  if (type == 0) {
    myPrint2(matFile1, matFile2);
  } else if (type == 1) {
    myPrint2(solidFile1, solidFile2);
  } else if (type == 2) {
    myPrint2(lvFile1, lvFile2);
  } else if (type == 3) {
    myPrint2(pvFile1, pvFile2);
  } else {
    myPrint2(regFile1, regFile2);
  }

  std::cout << "\nEntry in " << fileFile2 << " not in " << fileFile1 << std::endl;
  if (type == 0) {
    myPrint2(matFile2, matFile1);
  } else if (type == 1) {
    myPrint2(solidFile2, solidFile1);
  } else if (type == 2) {
    myPrint2(lvFile2, lvFile1);
  } else if (type == 2) {
    myPrint2(pvFile2, pvFile1);
  } else {
    myPrint2(regFile2, regFile1);
  }

  //Now type specific changes
  std::cout << "\nEntries in " << fileFile1 << " and " << fileFile2 << " do not match in the content\n";
  const double denmin = 0.0001;
  int kount1(0), kount2(0);
  double difmax1(0), difmax2(0);
  std::string nameMax("");
  if (type == 0) {
    const double tol1 = 0.00001;
    for (auto it1 : matFile1) {
      auto it2 = matFile2.find(it1.first);
      if (it2 != matFile2.end()) {
        ++kount1;
        double rdif =
            0.5 * (it1.second.radl - it2->second.radl) / std::max(denmin, (it1.second.radl + it2->second.radl));
        double idif =
            0.5 * (it1.second.intl - it2->second.intl) / std::max(denmin, (it1.second.intl + it2->second.intl));
        if (std::abs(rdif) > difmax1) {
          difmax1 = std::abs(rdif);
          difmax2 = std::abs(idif);
          nameMax = it1.first;
        }
        if ((std::abs(rdif) > tol1) || (std::abs(idif) > tol1)) {
          ++kount2;
          std::cout << it1.first << " X0 " << it1.second.radl << ":" << it2->second.radl << ":" << rdif << " #L "
                    << it1.second.intl << ":" << it2->second.intl << ":" << idif << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol1
              << " or more; the maximum happens for " << nameMax << " with " << difmax1 << ":" << difmax2 << "\n";
  } else if (type == 1) {
    const double tol2 = 0.0001;
    for (auto it1 : solidFile1) {
      auto it2 = solidFile2.find(it1.first);
      if (it2 != solidFile2.end()) {
        ++kount1;
        double vdif =
            0.5 * (it1.second.volume - it2->second.volume) / std::max(denmin, (it1.second.volume + it2->second.volume));
        if (std::abs(vdif) > difmax1) {
          difmax1 = std::abs(vdif);
          nameMax = it1.first;
        }
        if (std::abs(vdif) > tol2) {
          ++kount2;
          std::cout << it1.first << " Volume " << it1.second.volume << ":" << it2->second.volume << ":" << vdif
                    << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol2
              << " or more; the maximum happens for " << nameMax << " with " << difmax1 << "\n";
  } else if (type == 2) {
    const double tol3 = 0.0001;
    for (auto it1 : lvFile1) {
      auto it2 = lvFile2.find(it1.first);
      if (it2 != lvFile2.end()) {
        ++kount1;
        double vdif =
            0.5 * (it1.second.mass - it2->second.mass) / std::max(denmin, (it1.second.mass + it2->second.mass));
        if (std::abs(vdif) > difmax1) {
          difmax1 = std::abs(vdif);
          nameMax = it1.first;
        }
        if (std::abs(vdif) > tol3) {
          ++kount2;
          std::cout << it1.first << " Mass " << it1.second.mass << ":" << it2->second.mass << ":" << vdif << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol3
              << " or more; the maximum happens for " << nameMax << " with " << difmax1 << "\n";
  } else if (type == 3) {
    const double tol4 = 0.0001;
    for (auto it1 : pvFile1) {
      auto it2 = pvFile2.find(it1.first);
      if (it2 != pvFile2.end()) {
        ++kount1;
        double xdif = (it1.second.xx - it2->second.xx);
        double ydif = (it1.second.yy - it2->second.yy);
        double zdif = (it1.second.zz - it2->second.zz);
        double vdif = std::max(std::abs(xdif), std::abs(ydif));
        vdif = std::max(vdif, std::abs(zdif));
        if (vdif > difmax1) {
          difmax1 = vdif;
          nameMax = it1.first;
        }
        if ((std::abs(xdif) > tol4) || (std::abs(ydif) > tol4) || (std::abs(zdif) > tol4)) {
          ++kount2;
          std::cout << it1.first << " x " << it1.second.xx << ":" << it2->second.xx << ":" << xdif << " y "
                    << it1.second.yy << ":" << it2->second.yy << ":" << ydif << " z " << it1.second.zz << ":"
                    << it2->second.zz << ":" << zdif << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol4
              << " or more; the maximum happens for " << nameMax << " with " << difmax1 << "\n";
  } else {
    const double tol5 = 0.0001;
    for (auto it1 : regFile1) {
      auto it2 = regFile2.find(it1.first);
      if (it2 != regFile2.end()) {
        ++kount1;
        double matdif = (it1.second.nmat - it2->second.nmat);
        double voldif = (it1.second.nvol - it2->second.nvol);
        if (std::abs(matdif) > difmax1) {
          difmax1 = std::abs(matdif);
          nameMax = it1.first;
        }
        if (std::abs(voldif) > difmax2) {
          difmax2 = std::abs(voldif);
          nameMax = it1.first;
        }
        if ((std::abs(matdif) > tol5) || (std::abs(voldif) > tol5)) {
          ++kount2;
          std::cout << it1.first << " Material " << it1.second.nmat << ":" << it2->second.nmat << ":" << matdif
                    << " Volume " << it1.second.nvol << ":" << it2->second.nvol << ":" << voldif << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol5
              << " or more; the maximum happens for " << nameMax << " with " << difmax1 << ":" << difmax2 << "\n";
  }
}

int main(int argc, char* argv[]) {
  if (argc <= 5) {
    std::cout << "Please give a minimum of 2 arguments \n"
              << "name of the first input file\n"
              << "name of the second input file\n"
              << "type (Material:0, Solid:1, LV:2, PV:3, Region:4)\n"
              << "files (10 if first file from DDD and second from DD4hep)\n"
              << "debug flag (0 for minimum printout)\n"
              << std::endl;
    return 0;
  }

  const char* infile1 = argv[1];
  const char* infile2 = argv[2];
  int type = ((argc > 3) ? atoi(argv[3]) : 0);
  if (type < 0 || type > 3)
    type = 0;
  int files = ((argc > 4) ? atoi(argv[4]) : 10);
  int debug = ((argc > 6) ? atoi(argv[6]) : 0);
  CompareFiles(infile1, infile2, type, files, debug);
  return 0;
}
