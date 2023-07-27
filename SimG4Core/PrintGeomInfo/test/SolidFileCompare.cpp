////////////////////////////////////////////////////////////////////////////////
//
//    Compares output files from PrintG4Solids created using DDD and DD4hep
//    inputs. Usage:
//
//    SolidFileCompare infile1 infile2 debug
//    infile1  (const char*)   First file name (from dd4hep)
//    infile2  (const char*)   Second file name (from ddd)
//    debug    (int)           Single digit number (0 minimum printout)
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdint>

std::pair<std::string, std::string> splitInto2(const std::string& list) {
  std::string list1, list2;
  uint32_t first = (list.find(":") == std::string::npos) ? 0 : (list.find(":") + 1);
  uint32_t last(list.size() + 1);
  if (first > 0) {
    list2 = list.substr(first, last - first - 1);
    list1 = list.substr(0, first - 1);
  } else {
    list1 = list;
    list2 = "";
  }
  return std::make_pair(list1, list2);
}

void CompareFiles(const char* fileFile1, const char* fileFile2, int debug) {
  std::map<std::string, std::string> solidFile1, solidFile2;
  std::map<std::string, int> solidFile1Dup, solidFile2Dup;
  char buffer[1000];
  std::string name;
  std::ifstream fInput1(fileFile1);
  unsigned int sizeFile1(0), sizeFile2(0);
  if (!fInput1.good()) {
    std::cout << "Cannot open file " << fileFile1 << std::endl;
  } else {
    while (fInput1.getline(buffer, 1000)) {
      std::pair<std::string, std::string> items = splitInto2(std::string(buffer));
      if (solidFile1.find(items.first) == solidFile1.end())
        solidFile1[items.first] = items.second;
      else if (solidFile1Dup.find(items.first) == solidFile1Dup.end())
        solidFile1Dup[items.first] = 1;
      else
        ++solidFile1Dup[items.first];
    }
    fInput1.close();
    sizeFile1 = solidFile1.size();
  }
  std::ifstream fInput2(fileFile2);
  if (!fInput2.good()) {
    std::cout << "Cannot open file " << fileFile2 << std::endl;
  } else {
    while (fInput2.getline(buffer, 1000)) {
      std::pair<std::string, std::string> items = splitInto2(std::string(buffer));
      if (solidFile2.find(items.first) == solidFile2.end())
        solidFile2[items.first] = items.second;
      else if (solidFile2Dup.find(items.first) == solidFile2Dup.end())
        solidFile2Dup[items.first] = 1;
      else
        ++solidFile2Dup[items.first];
    }
    fInput2.close();
    sizeFile2 = solidFile2.size();
  }
  std::cout << "Reads " << sizeFile1 << " names from " << fileFile1 << " and " << sizeFile2 << " names from "
            << fileFile2 << std::endl;

  std::cout << "\n" << solidFile1Dup.size() << " more than one entry for a given name in " << fileFile1 << std::endl;
  int i1(0);
  std::map<std::string, int>::iterator itr;
  for (itr = solidFile1Dup.begin(); itr != solidFile1Dup.end(); ++itr, ++i1)
    std::cout << "[" << i1 << "] " << itr->first << " # " << itr->second << std::endl;

  std::cout << "\n" << solidFile1Dup.size() << " more than one entry for a given name in " << fileFile2 << std::endl;
  int i2(0);
  for (itr = solidFile2Dup.begin(); itr != solidFile2Dup.end(); ++itr, ++i2)
    std::cout << "[" << i2 << "] " << itr->first << " # " << itr->second << std::endl;

  std::cout << "\nEntry in " << fileFile1 << " vs entry in " << fileFile2 << std::endl;
  int k3(0);
  std::vector<std::string> v1;
  std::map<std::string, std::string>::iterator ktr;
  for (ktr = solidFile1.begin(); ktr != solidFile1.end(); ++ktr) {
    if (solidFile2.find(ktr->first) == solidFile2.end())
      v1.emplace_back(ktr->first);
    else if (solidFile2[ktr->first] == ktr->second)
      ++k3;
    else
      std::cout << itr->first << " in File1 " << ktr->second << "\n         in File2 " << solidFile2[ktr->first]
                << std::endl;
  }
  std::cout << "\n" << k3 << " entries match between " << fileFile1 << " and " << fileFile2 << std::endl;
  std::cout << v1.size() << " entries in " << fileFile1 << " are not in " << fileFile2 << std::endl;
  for (auto const& it : v1) {
    std::cout << it << std::endl;
  }

  int k4(0);
  std::vector<std::string> v2;
  for (ktr = solidFile2.begin(); ktr != solidFile2.end(); ++ktr) {
    if (solidFile1.find(ktr->first) == solidFile1.end())
      v2.emplace_back(ktr->first);
    else if (solidFile1[ktr->first] == ktr->second)
      ++k4;
  }
  std::cout << "\n" << k4 << " entries match between " << fileFile2 << " and " << fileFile1 << std::endl;
  std::cout << v2.size() << " entries in " << fileFile2 << " are not in " << fileFile1 << std::endl;
  for (auto const& it : v2) {
    std::cout << it << std::endl;
  }
}

int main(int argc, char* argv[]) {
  if (argc <= 3) {
    std::cout << "Please give a minimum of 2 arguments \n"
              << "name of the first input file (dd4hep) \n"
              << "name of the second input file (ddd) \n"
              << "debug flag (0 for minimum printout)\n"
              << std::endl;
    return 0;
  }

  const char* infile1 = argv[1];
  const char* infile2 = argv[2];
  const int debug = std::atoi(argv[3]);
  CompareFiles(infile1, infile2, debug);
  return 0;
}
