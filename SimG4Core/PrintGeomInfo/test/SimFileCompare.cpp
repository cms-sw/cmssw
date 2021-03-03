////////////////////////////////////////////////////////////////////////////////
//
//    Compares output files from PrintGeomInfo created using DDD and DD4Hep
//    inputs. Usage:
//
//    SimFileCompare infile1 infile2 mode debug
//    infile1  (const char*)   File created using DDD
//    infile2  (const char*)   File created using DD4Hep
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

void CompareFiles(const char* fileDDD, const char* fileDD4Hep, int mode, int debug) {
  std::map<std::string, int> nameDDD, nameDD4Hep;
  char buffer[100];
  std::string name;
  std::ifstream fInput1(fileDDD);
  if (!fInput1.good()) {
    std::cout << "Cannot open file " << fileDDD << std::endl;
  } else {
    while (fInput1.getline(buffer, 100)) {
      name = ((mode == 1) ? removeExtraName(std::string(buffer), debug) : std::string(buffer));
      auto it = nameDDD.find(name);
      if (it == nameDDD.end())
        nameDDD[name] = 1;
      else
        ++(it->second);
    }
    fInput1.close();
  }
  std::ifstream fInput2(fileDD4Hep);
  if (!fInput2.good()) {
    std::cout << "Cannot open file " << fileDD4Hep << std::endl;
  } else {
    while (fInput2.getline(buffer, 100)) {
      name = reducedName(std::string(buffer), debug);
      auto it = nameDD4Hep.find(name);
      if (it == nameDD4Hep.end())
        nameDD4Hep[name] = 1;
      else
        ++(it->second);
    }
    fInput1.close();
  }
  std::cout << "Reads " << nameDDD.size() << " names from " << fileDDD << " and " << nameDD4Hep.size() << " names from "
            << fileDD4Hep << std::endl;

  std::cout << "\nMore than one entry for a given name in " << fileDDD << std::endl;
  for (auto it : nameDDD) {
    if (it.second > 1)
      std::cout << it.first << " : " << it.second << std::endl;
  }
  std::cout << "\nMore than one entry for a given name in " << fileDD4Hep << std::endl;
  for (auto it : nameDD4Hep) {
    if (it.second > 1)
      std::cout << it.first << " : " << it.second << std::endl;
  }
  std::cout << "\nEntry in " << fileDDD << " not in " << fileDD4Hep << std::endl;
  for (auto it : nameDDD) {
    if (nameDD4Hep.find(it.first) == nameDD4Hep.end())
      std::cout << it.first << " appearing " << it.second << " times" << std::endl;
  }
  std::cout << "\nEntry in " << fileDD4Hep << " not in " << fileDDD << std::endl;
  for (auto it : nameDD4Hep) {
    if (nameDDD.find(it.first) == nameDDD.end())
      std::cout << it.first << " appearing " << it.second << " times" << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc <= 3) {
    std::cout << "Please give a minimum of 2 arguments \n"
              << "input file name from the DDD run\n"
              << "input file name from the DD4Hep run\n"
              << "mode (treat the name for DDD or not == needed for PV)\n"
              << "debug flag (0 for minimum printout)\n"
              << std::endl;
    return 0;
  }

  const char* infile1 = argv[1];
  const char* infile2 = argv[2];
  int mode = ((argc > 3) ? atoi(argv[3]) : 0);
  int debug = ((argc > 4) ? atoi(argv[4]) : 0);
  CompareFiles(infile1, infile2, mode, debug);
  return 0;
}
