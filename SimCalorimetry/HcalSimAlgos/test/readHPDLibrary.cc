#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseReader.h"
#include "SimCalorimetry/HcalSimAlgos/interface/HPDNoiseData.h"

#include <iostream>
#include "TClass.h"

int main () {
  HPDNoiseReader reader ("hpdNoiseLibrary.root");
  std::vector <std::string> names = reader.allNames ();
  for (size_t i = 0; i < names.size(); ++i) {
    HPDNoiseReader::Handle handle = reader.getHandle (names[i]);
    std::cout << "name/valid: " << names[i] << '/' << reader.valid (handle) << std::endl;
    if (reader.valid (handle)) {
      std::cout << "   rate: " << reader.rate (handle);
      std::cout << "   entries: " << reader.totalEntries (handle) << std::endl;
      for (unsigned long ievt = 0; ievt < 5; ++ievt) {
	HPDNoiseData* data;
	reader.getEntry (handle, &data);
	std::cout << "entry # " << ievt << std::endl;
	std::cout << *data << std::endl;
      }
    }
  }
  return 0;
}
