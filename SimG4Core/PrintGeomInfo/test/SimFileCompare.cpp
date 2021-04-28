////////////////////////////////////////////////////////////////////////////////
//
//    Compares output files from PrintGeomInfo created using DDD and DD4Hep
//    inputs. Usage:
//
//    SimFileCompare infile1 infile2 mode debug
//    infile1  (const char*)   File created using DDD
//    infile2  (const char*)   File created using DD4Hep
//    type     (int)           Type of file: material (0), solid (1),
//                             LogicalVolume (2), PhysicalVolume (3)
//    mode     (int)           Treat (0) or not treat (1) names from DDD
//    deug     (int)           Single digit number (0 minimum printout)
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

void CompareFiles(const char* fileDDD, const char* fileDD4Hep, int type, int mode, int debug) {
  std::map<std::string, materials> matDDD, matDD4Hep;
  std::map<std::string, solids> solidDDD, solidDD4Hep;
  std::map<std::string, lvs> lvDDD, lvDD4Hep;
  std::map<std::string, pvs> pvDDD, pvDD4Hep;
  char buffer[100];
  std::string name;
  std::ifstream fInput1(fileDDD);
  unsigned int sizeDDD(0), sizeDD4Hep(0);
  if (!fInput1.good()) {
    std::cout << "Cannot open file " << fileDDD << std::endl;
  } else {
    while (fInput1.getline(buffer, 100)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      name = ((mode == 1) ? removeExtraName(items[0], debug) : items[0]);
      double r1 = (items.size() > 1) ? atof(items[1].c_str()) : 0;
      double r2 = (items.size() > 2) ? atof(items[2].c_str()) : 0;
      double r3 = (items.size() > 3) ? atof(items[3].c_str()) : 0;
      if (type == 0) {
        auto it = matDDD.find(name);
        if (it == matDDD.end())
          matDDD[name] = materials(1, r1, r2);
        else
          ++((it->second).occ);
      } else if (type == 1) {
        auto it = solidDDD.find(name);
        if (it == solidDDD.end())
          solidDDD[name] = solids(1, r1);
        else
          ++((it->second).occ);
      } else if (type == 2) {
        auto it = lvDDD.find(name);
        if (it == lvDDD.end())
          lvDDD[name] = lvs(1, r1);
        else
          ++((it->second).occ);
      } else {
        auto it = pvDDD.find(name);
        if (it == pvDDD.end())
          pvDDD[name] = pvs(1, r1, r2, r3);
        else
          ++((it->second).occ);
      }
    }
    fInput1.close();
    sizeDDD =
        ((type == 0) ? matDDD.size() : ((type == 1) ? solidDDD.size() : ((type == 2) ? lvDDD.size() : pvDDD.size())));
  }
  std::ifstream fInput2(fileDD4Hep);
  if (!fInput2.good()) {
    std::cout << "Cannot open file " << fileDD4Hep << std::endl;
  } else {
    while (fInput2.getline(buffer, 100)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      name = reducedName(items[0], debug);
      double r1 = (items.size() > 1) ? atof(items[1].c_str()) : 0;
      double r2 = (items.size() > 2) ? atof(items[2].c_str()) : 0;
      double r3 = (items.size() > 3) ? atof(items[3].c_str()) : 0;
      if (type == 0) {
        auto it = matDD4Hep.find(name);
        if (it == matDD4Hep.end())
          matDD4Hep[name] = materials(1, r1, r2);
        else
          ++((it->second).occ);
      } else if (type == 1) {
        auto it = solidDD4Hep.find(name);
        if (it == solidDD4Hep.end())
          solidDD4Hep[name] = solids(1, r1);
        else
          ++((it->second).occ);
      } else if (type == 2) {
        auto it = lvDD4Hep.find(name);
        if (it == lvDD4Hep.end())
          lvDD4Hep[name] = lvs(1, r1);
        else
          ++((it->second).occ);
      } else {
        auto it = pvDD4Hep.find(name);
        if (it == pvDD4Hep.end())
          pvDD4Hep[name] = pvs(1, r1, r2, r3);
        else
          ++((it->second).occ);
      }
    }
    fInput2.close();
    sizeDD4Hep = ((type == 0) ? matDD4Hep.size()
                              : ((type == 1) ? solidDD4Hep.size() : ((type == 2) ? lvDD4Hep.size() : pvDD4Hep.size())));
  }
  std::cout << "Reads " << sizeDDD << " names from " << fileDDD << " and " << sizeDD4Hep << " names from " << fileDD4Hep
            << std::endl;

  std::cout << "\nMore than one entry for a given name in " << fileDDD << std::endl;
  if (type == 0) {
    myPrint1(matDDD);
  } else if (type == 1) {
    myPrint1(solidDDD);
  } else if (type == 2) {
    myPrint1(lvDDD);
  } else {
    myPrint1(pvDDD);
  }

  std::cout << "\nMore than one entry for a given name in " << fileDD4Hep << std::endl;
  if (type == 0) {
    myPrint1(matDD4Hep);
  } else if (type == 1) {
    myPrint1(solidDD4Hep);
  } else if (type == 2) {
    myPrint1(lvDD4Hep);
  } else {
    myPrint1(pvDD4Hep);
  }

  std::cout << "\nEntry in " << fileDDD << " not in " << fileDD4Hep << std::endl;
  if (type == 0) {
    myPrint2(matDDD, matDD4Hep);
  } else if (type == 1) {
    myPrint2(solidDDD, solidDD4Hep);
  } else if (type == 2) {
    myPrint2(lvDDD, lvDD4Hep);
  } else {
    myPrint2(pvDDD, pvDD4Hep);
  }

  std::cout << "\nEntry in " << fileDD4Hep << " not in " << fileDDD << std::endl;
  if (type == 0) {
    myPrint2(matDD4Hep, matDDD);
  } else if (type == 1) {
    myPrint2(solidDD4Hep, solidDDD);
  } else if (type == 2) {
    myPrint2(lvDD4Hep, lvDDD);
  } else {
    myPrint2(pvDD4Hep, pvDDD);
  }

  //Now type specific changes
  std::cout << "\nEntries in " << fileDDD << " and " << fileDD4Hep << " do not match in the content\n";
  const double denmin = 0.0001;
  int kount1(0), kount2(0);
  if (type == 0) {
    const double tol1 = 0.00001;
    for (auto it1 : matDDD) {
      auto it2 = matDD4Hep.find(it1.first);
      if (it2 != matDD4Hep.end()) {
        ++kount1;
        double rdif =
            0.5 * (it1.second.radl - it2->second.radl) / std::max(denmin, (it1.second.radl + it2->second.radl));
        double idif =
            0.5 * (it1.second.intl - it2->second.intl) / std::max(denmin, (it1.second.intl + it2->second.intl));
        if ((std::abs(rdif) > tol1) || (std::abs(idif) > tol1)) {
          ++kount2;
          std::cout << it1.first << " Radiation Length " << it1.second.radl << ":" << it2->second.radl << ":" << rdif
                    << " Interaction Length " << it1.second.intl << ":" << it2->second.intl << ":" << idif << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol1
              << " or more\n";
  } else if (type == 1) {
    const double tol2 = 0.0001;
    for (auto it1 : solidDDD) {
      auto it2 = solidDD4Hep.find(it1.first);
      if (it2 != solidDD4Hep.end()) {
        ++kount1;
        double vdif =
            0.5 * (it1.second.volume - it2->second.volume) / std::max(denmin, (it1.second.volume + it2->second.volume));
        if (std::abs(vdif) > tol2) {
          ++kount2;
          std::cout << it1.first << " Volume " << it1.second.volume << ":" << it2->second.volume << ":" << vdif
                    << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol2
              << " or more\n";
  } else if (type == 2) {
    const double tol3 = 0.0001;
    for (auto it1 : lvDDD) {
      auto it2 = lvDD4Hep.find(it1.first);
      if (it2 != lvDD4Hep.end()) {
        ++kount1;
        double vdif =
            0.5 * (it1.second.mass - it2->second.mass) / std::max(denmin, (it1.second.mass + it2->second.mass));
        if (std::abs(vdif) > tol3) {
          ++kount2;
          std::cout << it1.first << " Mass " << it1.second.mass << ":" << it2->second.mass << ":" << vdif << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol3
              << " or more\n";
  } else {
    const double tol4 = 0.0001;
    for (auto it1 : pvDDD) {
      auto it2 = pvDD4Hep.find(it1.first);
      if (it2 != pvDD4Hep.end()) {
        ++kount1;
        double xdif = (it1.second.xx - it2->second.xx);
        double ydif = (it1.second.yy - it2->second.yy);
        double zdif = (it1.second.zz - it2->second.zz);
        if ((std::abs(xdif) > tol4) || (std::abs(ydif) > tol4) || (std::abs(zdif) > tol4)) {
          ++kount2;
          std::cout << it1.first << " x " << it1.second.xx << ":" << it2->second.xx << ":" << xdif << " y "
                    << it1.second.yy << ":" << it2->second.yy << ":" << ydif << " z " << it1.second.zz << ":"
                    << it2->second.zz << ":" << zdif << std::endl;
        }
      }
    }
    std::cout << "\n " << kount2 << " out of " << kount1 << " entries having discrpancies at the level of " << tol4
              << " or more\n";
  }
}

int main(int argc, char* argv[]) {
  if (argc <= 4) {
    std::cout << "Please give a minimum of 2 arguments \n"
              << "input file name from the DDD run\n"
              << "input file name from the DD4Hep run\n"
              << "type (Material:0, Solid:1, LV:2, PV:3\n"
              << "mode (treat the name for DDD or not == needed for PV)\n"
              << "debug flag (0 for minimum printout)\n"
              << std::endl;
    return 0;
  }

  const char* infile1 = argv[1];
  const char* infile2 = argv[2];
  int type = ((argc > 3) ? atoi(argv[3]) : 0);
  if (type < 0 || type > 3)
    type = 0;
  int mode = ((argc > 4) ? atoi(argv[4]) : 0);
  int debug = ((argc > 5) ? atoi(argv[5]) : 0);
  CompareFiles(infile1, infile2, type, mode, debug);
  return 0;
}
