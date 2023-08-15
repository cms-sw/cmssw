////////////////////////////////////////////////////////////////////////////////
//
//    Compares iles containing lists of Sensitive Detectors, Touchables obtaied
//    from DDD and DD4hep and uses a list of volumes touched while running
//    MinimumBias events and reaching partial wafers. Usage:
//
//    TouchFileCompare infile1 infile2 debug
//    sdFileDDD       (const char*) SD file name (from DDD)
//    sdFileDD4hep    (const char*) SD file name (from DD4hep)
//    touchFileDDD    (const char*) Touch file name (from DDD)
//    touchFileDD4hep (const char*) Touch file name (from DD4hep)
//    mbFile          (const char*) MinimumBias o/p file name (grep with "Top ")
//    debug           (int)         Debug flag (0 minimum printout)
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

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

std::string noNameSpace(std::string& name) {
  std::size_t found = name.find(":");
  std::string nam = (found == std::string::npos) ? name : name.substr(found + 1, (name.size() - found));
  return nam;
}

void CompareFiles(const char* sdFileDDD,
                  const char* sdFileDD4hep,
                  const char* touchFileDDD,
                  const char* touchFileDD4hep,
                  const char* mbFile,
                  int debug) {
  std::vector<std::string> sdDDD, sdDD4hep, touchDDD, touchDD4hep, mbVol;
  char buffer[1000];
  std::ifstream fInput1(sdFileDDD);
  int all(0), good(0), ee(0), hesil(0), hesci(0);
  if (!fInput1.good()) {
    std::cout << "Cannot open file " << sdFileDDD << std::endl;
  } else {
    while (fInput1.getline(buffer, 1000)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      ++all;
      if (items.size() > 0) {
        sdDDD.emplace_back(items[0]);
        if (((debug / 10) % 10) > 0)
          std::cout << "[" << good << "] " << sdDDD.back() << std::endl;
        ++good;
        if (sdDDD.back().find("EE") != std::string::npos) {
          ++ee;
        } else if (sdDDD.back().find("HESil") != std::string::npos) {
          ++hesil;
        } else if (sdDDD.back().find("HESci") != std::string::npos) {
          ++hesci;
        }
      }
    }
    fInput1.close();
    std::cout << "Reads " << all << ":" << good << " names from " << sdFileDDD << " with " << ee << ":" << hesil << ":"
              << hesci << " for EE:HESilicon:HEScintillator" << std::endl;
  }
  std::ifstream fInput2(sdFileDD4hep);
  all = good = ee = hesil = hesci = 0;
  if (!fInput2.good()) {
    std::cout << "Cannot open file " << sdFileDD4hep << std::endl;
  } else {
    while (fInput2.getline(buffer, 1000)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      ++all;
      if (items.size() > 0) {
        sdDD4hep.emplace_back(noNameSpace(items[0]));
        if (((debug / 10) % 10) > 0)
          std::cout << "[" << good << "] " << sdDD4hep.back() << std::endl;
        ++good;
        if (sdDD4hep.back().find("EE") != std::string::npos) {
          ++ee;
        } else if (sdDD4hep.back().find("HESil") != std::string::npos) {
          ++hesil;
        } else if (sdDD4hep.back().find("HESci") != std::string::npos) {
          ++hesci;
        }
      }
    }
    fInput2.close();
    std::cout << "Reads " << all << ":" << good << " names from " << sdFileDD4hep << " with " << ee << ":" << hesil
              << ":" << hesci << " for EE:HESilicon:HEScintillator" << std::endl;
  }
  std::ifstream fInput3(touchFileDDD);
  all = good = 0;
  if (!fInput3.good()) {
    std::cout << "Cannot open file " << touchFileDDD << std::endl;
  } else {
    while (fInput3.getline(buffer, 1000)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      ++all;
      if (items.size() > 0) {
        touchDDD.emplace_back(items[0]);
        if (((debug / 100) % 10) > 0)
          std::cout << "[" << good << "] " << touchDDD.back() << std::endl;
        ++good;
      }
    }
    fInput3.close();
    std::cout << "Reads " << all << ":" << good << " names from " << touchFileDDD << std::endl;
  }
  std::ifstream fInput4(touchFileDD4hep);
  all = good = 0;
  if (!fInput4.good()) {
    std::cout << "Cannot open file " << touchFileDD4hep << std::endl;
  } else {
    while (fInput4.getline(buffer, 1000)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      ++all;
      if (items.size() > 0) {
        touchDD4hep.emplace_back(items[0]);
        if (((debug / 100) % 10) > 0)
          std::cout << "[" << good << "] " << touchDD4hep.back() << std::endl;
        ++good;
      }
    }
    fInput4.close();
    std::cout << "Reads " << all << ":" << good << " names from " << touchFileDD4hep << std::endl;
  }
  std::ifstream fInput5(mbFile);
  all = good = 0;
  if (!fInput5.good()) {
    std::cout << "Cannot open file " << mbFile << std::endl;
  } else {
    while (fInput5.getline(buffer, 1000)) {
      std::vector<std::string> items = splitString(std::string(buffer));
      ++all;
      if (items.size() > 1) {
        mbVol.emplace_back(items[1]);
        if (((debug / 1000) % 10) > 0)
          std::cout << "[" << good << "] " << mbVol.back() << std::endl;
        ++good;
      }
    }
    fInput5.close();
    std::cout << "Reads " << all << ":" << good << " names from " << mbFile << std::endl;
  }
  all = ee = hesil = hesci = 0;
  std::vector<std::string> extraSD;
  for (const auto& sd : sdDDD) {
    if (std::find(sdDD4hep.begin(), sdDD4hep.end(), sd) == sdDD4hep.end()) {
      extraSD.emplace_back(sd);
      if (((debug / 10000) % 10) > 0)
        std::cout << "[" << all << "] " << extraSD.back() << std::endl;
      ++all;
      if (extraSD.back().find("EE") != std::string::npos) {
        ++ee;
      } else if (extraSD.back().find("HESil") != std::string::npos) {
        ++hesil;
      } else if (extraSD.back().find("HESci") != std::string::npos) {
        ++hesci;
      }
    }
  }
  std::cout << "Additional " << all << " names in " << sdFileDDD << " with " << ee << ":" << hesil << ":" << hesci
            << " for EE:HESilicon:HEScintillator" << std::endl;
  all = 0;
  std::vector<int> ddd(extraSD.size(), 0), dd4hep(extraSD.size(), 0), mb(extraSD.size(), 0);
  for (unsigned int k = 0; k < extraSD.size(); ++k) {
    bool t1 = (std::find(touchDDD.begin(), touchDDD.end(), extraSD[k]) != touchDDD.end());
    bool t2 = (std::find(touchDD4hep.begin(), touchDD4hep.end(), extraSD[k]) != touchDD4hep.end());
    bool t3 = (std::find(mbVol.begin(), mbVol.end(), extraSD[k]) != mbVol.end());
    if (t1)
      ++ddd[k];
    if (t2)
      ++dd4hep[k];
    if (t3)
      ++mb[k];
    if (t1 || t2 || t3) {
      ++all;
      std::cout << "Extra SD " << extraSD[k] << " in DDD|DD4Touch|MB File " << t1 << ":" << t2 << ":" << t3
                << std::endl;
    }
  }
  std::cout << all << " extra SD's appear in some touchable or MB file " << std::endl;
  if (((debug / 1) % 10) > 0) {
    for (unsigned int k = 0; k < extraSD.size(); ++k)
      std::cout << "[" << k << "] " << extraSD[k] << " Find count " << ddd[k] << ":" << dd4hep[k] << ":" << mb[k]
                << std::endl;
  }
  //store sorted touch file for DDD
  char fname[100];
  std::sort(touchDDD.begin(), touchDDD.end());
  sprintf(fname, "sorted%s", touchFileDDD);
  std::ofstream fOut1(fname);
  all = 0;
  for (const auto& touch : touchDDD) {
    fOut1 << " " << touch << "\n";
    ++all;
  }
  fOut1.close();
  std::cout << "Writes " << all << " Touchables in " << fname << " from " << touchFileDDD << std::endl;
  //store sorted touch file for DDD
  all = 0;
  std::sort(touchDD4hep.begin(), touchDD4hep.end());
  sprintf(fname, "sorted%s", touchFileDD4hep);
  std::ofstream fOut2(fname);
  for (const auto& touch : touchDD4hep) {
    fOut2 << " " << touch << "\n";
    ++all;
  }
  fOut2.close();
  std::cout << "Writes " << all << " Touchables in " << fname << " from " << touchFileDD4hep << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc <= 6) {
    std::cout << "Please give a minimum of 6 arguments \n"
              << "name of the SD file corresponding to DDD \n"
              << "name of the SD file corresponding to DD4hep \n"
              << "name of the Touch file corresponding to DDD \n"
              << "name of the Touch file corresponding to DD4hep \n"
              << "name of the sMnimumBias file \n"
              << "debug flag (0 for minimum printout)\n"
              << std::endl;
    return 0;
  }

  const char* sdFileDDD = argv[1];
  const char* sdFileDD4hep = argv[2];
  const char* touchFileDDD = argv[3];
  const char* touchFileDD4hep = argv[4];
  const char* mbFile = argv[5];
  const int debug = std::atoi(argv[6]);
  CompareFiles(sdFileDDD, sdFileDD4hep, touchFileDDD, touchFileDD4hep, mbFile, debug);
  return 0;
}
