#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

std::vector<std::string> splitString (const std::string& fLine) {
  std::vector<std::string> result;
  int  start(0);
  bool empty(true);
  for (unsigned i = 0; i <= fLine.size (); i++) {
    if (fLine[i] == ' ' || i == fLine.size ()) {
      if (!empty) {
        std::string item(fLine, start, i-start);
        result.push_back(item);
        empty = true;
      }
      start = i+1;
    } else {
      if (empty) empty = false;
    }
  }
  return result;
}

void makeList(char* infile) {

  std::map<std::string,int> list;
  std::ifstream fInput(infile);
  if (!fInput.good()) {
    std::cout << "Cannot open file " << infile << std::endl;
  } else {
    char buffer [1024];
    unsigned int all(0), good(0);
    while (fInput.getline(buffer, 1024)) {
      ++all;
      std::vector<std::string> items = splitString (std::string(buffer));
      if (items.size() == 6 && items[0] == "Overlap") {
	++good;
	std::map<std::string,int>::iterator itr = list.find(items[5]);
	if (itr == list.end()) {
	  list[items[5]] = 0;
	  itr = list.find(items[5]);
	}
	++(itr->second);
      }
    }
    fInput.close();
    std::cout << "Reads total of " << all << " and " << good << " good records"
              << " from " << infile << std::endl;
  }

  std::cout << "\nFinds " << list.size() << " volumes with overlaps\n\n";
  for (std::map<std::string,int>::iterator itr = list.begin();
       itr != list.end(); ++ itr)
    std::cout << "Volume " << itr->first << " Kount " << itr->second << "\n";
}
 
