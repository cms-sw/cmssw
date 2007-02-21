// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <cmath>

// maps to compare
std::map< std::string , Int_t > theNewMapNavTypeToDetId;
std::map< std::string , Int_t > theReferenceMapNavTypeToDetId;
//

// Main
TrackerNumberingComparison( TString newFileName , TString referenceFileName , TString outputFileName) {
  // open files
  std::ifstream theNewFile;
  theNewFile.open(newFileName);
  std::ifstream theReferenceFile;
  theReferenceFile.open(referenceFileName);
  // output file
  std::ofstream theOutputFile;
  theOutputFile.open(outputFileName);
  //
  
  // filling the maps
  Int_t detid;
  detid = 0;
  std::string navType;
  float x,y,z;
  x=y=z=0;
  //
  
  //
  //  std::cout << " Reading " << newFileName << " ..." << std::endl;
  while(!theNewFile.eof()) {
    //
    theNewFile >> detid
	       >> navType
	       >> x >> y >> z;
    //
    //    std::std::cout << "New: load " << detid  << " " << navType << std::endl;
    //
    theNewMapNavTypeToDetId[navType] = detid;
  }
  //  std::cout << " ... done" << std::endl;
  //
  //  std::cout << " Reading " << referenceFileName << " ..." << std::endl;
  while(!theReferenceFile.eof()) {
    //
    theReferenceFile >> detid
	       >> navType
	       >> x >> y >> z;
    //
    //    std::std::cout << "Reference: load " << detid  << " " << navType << std::endl;
    //
    theReferenceMapNavTypeToDetId[navType] = detid;
  }
  //  std::cout << " ... done" << std::endl;
  //  
  
  // Compare the two maps
  // size
  //  std::cout << " Size of the maps: " << std::endl;
  //  std::cout << " Reference = " << theReferenceMapNavTypeToDetId.size() << std::endl;
  //  std::cout << "     New   = " << theNewMapNavTypeToDetId.size()       << std::endl;
  if(theNewMapNavTypeToDetId.size() != theReferenceMapNavTypeToDetId.size()) {
    theOutputFile << "ERROR: The size of the two maps is different" << std::endl;
    theOutputFile << " Reference = " << theReferenceMapNavTypeToDetId.size() << std::endl;
    theOutputFile << "     New   = " << theNewMapNavTypeToDetId.size()       << std::endl;
  }
  // compare the detid-navtype couples
  for( std::map< std::string , Int_t >::iterator iMap = theReferenceMapNavTypeToDetId.begin(); iMap != theReferenceMapNavTypeToDetId.end(); iMap++ ) {
    //    std::cout << " reference: " << " detid " << (*iMap).second << " navtype " << (*iMap).first << std::endl;
    //    std::cout << "   new:     " << " detid " << theNewMapNavTypeToDetId[(*iMap).first] << " navtype " << (*iMap).first << std::endl;
    Int_t referenceDetId = (*iMap).second;
    std::string navType  = (*iMap).first;
    Int_t newDetId       = theNewMapNavTypeToDetId[navType];
    if( referenceDetId != newDetId ) {
      theOutputFile << "ERROR: The detid's associated to the same navtype do not correspond" << std::endl;
      theOutputFile << " reference: " << " detid " << referenceDetId << " navtype " << navType << std::endl;
      theOutputFile << "   new:     " << " detid " << newDetId       << " navtype " << navType << std::endl;
    }
  }
  //
}
