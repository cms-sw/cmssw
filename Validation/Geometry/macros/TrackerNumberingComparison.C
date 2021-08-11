// **************** Validation Macro *******************
// * This macro compares the new and reference files   *
// * map detid,x,y,z with navtype                      *
// * for each navtype checks if detid,x,y,z correspond *
// *   written by rranieri 22 Feb 2007                 *
// *****************************************************

// include files
#include <iostream>
#include <iomanip>
#include <map>

//
Double_t tolerance = 0.0001; // 0.1 um is enough!
//

// maps to compare detid-navtype
std::map< std::string , Int_t > theNewMapNavTypeToDetId;
std::map< std::string , Int_t > theReferenceMapNavTypeToDetId;
//
// maps to compare position-navtype
std::map< std::string , Double_t > theNewMapNavTypeToGlobalPosition_X;
std::map< std::string , Double_t > theNewMapNavTypeToGlobalPosition_Y;
std::map< std::string , Double_t > theNewMapNavTypeToGlobalPosition_Z;
//
std::map< std::string , Double_t > theReferenceMapNavTypeToGlobalPosition_X;
std::map< std::string , Double_t > theReferenceMapNavTypeToGlobalPosition_Y;
std::map< std::string , Double_t > theReferenceMapNavTypeToGlobalPosition_Z;
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
  theOutputFile 
    << std::setprecision(4)
    << std::fixed;
  //
  
  // filling the maps
  Int_t detid;
  detid = 0;
  std::string navType;
  Double_t x,y,z;
  x=y=z=0.0000;
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
    theNewMapNavTypeToGlobalPosition_X[navType] = x;
    theNewMapNavTypeToGlobalPosition_Y[navType] = y;
    theNewMapNavTypeToGlobalPosition_Z[navType] = z;
    //
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
    theReferenceMapNavTypeToGlobalPosition_X[navType] = x;
    theReferenceMapNavTypeToGlobalPosition_Y[navType] = y;
    theReferenceMapNavTypeToGlobalPosition_Z[navType] = z;
    //
  }
  //  std::cout << " ... done" << std::endl;
  //  
  
  // Compare the two maps
  // size
  //  std::cout << " Size of the maps: " << std::endl;
  //  std::cout << " Reference = " << theReferenceMapNavTypeToDetId.size() << std::endl;
  //  std::cout << "     New   = " << theNewMapNavTypeToDetId.size()       << std::endl;
  if(theNewMapNavTypeToDetId.size() != theReferenceMapNavTypeToDetId.size()) {
    theOutputFile << "ERROR: The size of the two detid-navtype maps is different" << std::endl;
    theOutputFile << " Reference = " << theReferenceMapNavTypeToDetId.size()      << std::endl;
    theOutputFile << "     New   = " << theNewMapNavTypeToDetId.size()            << std::endl;
  }
  if(theNewMapNavTypeToGlobalPosition_X.size() != theReferenceMapNavTypeToGlobalPosition_X.size()) {
    theOutputFile << "ERROR: The size of the two X-navtype maps is different"  << std::endl;
    theOutputFile << " Reference = " << theReferenceMapNavTypeToGlobalPosition_X.size() << std::endl;
    theOutputFile << "     New   = " << theNewMapNavTypeToGlobalPosition_X.size()       << std::endl;
  }
  if(theNewMapNavTypeToGlobalPosition_Y.size() != theReferenceMapNavTypeToGlobalPosition_Y.size()) {
    theOutputFile << "ERROR: The size of the two Y-navtype maps is different"  << std::endl;
    theOutputFile << " Reference = " << theReferenceMapNavTypeToGlobalPosition_Y.size() << std::endl;
    theOutputFile << "     New   = " << theNewMapNavTypeToGlobalPosition_Y.size()       << std::endl;
  }
  if(theNewMapNavTypeToGlobalPosition_Z.size() != theReferenceMapNavTypeToGlobalPosition_Z.size()) {
    theOutputFile << "ERROR: The size of the two Z-navtype maps is different"  << std::endl;
    theOutputFile << " Reference = " << theReferenceMapNavTypeToGlobalPosition_Z.size() << std::endl;
    theOutputFile << "     New   = " << theNewMapNavTypeToGlobalPosition_Z.size()       << std::endl;
  }
  // compare the detid-navtype maps
  for( std::map< std::string , Int_t >::iterator iMap = theReferenceMapNavTypeToDetId.begin();
       iMap != theReferenceMapNavTypeToDetId.end(); iMap++ ) {
    //    std::cout << " reference: " << " detid " << (*iMap).second << " navtype " << (*iMap).first << std::endl;
    //    std::cout << "   new:     " << " detid " << theNewMapNavTypeToDetId[(*iMap).first] << " navtype " << (*iMap).first << std::endl;
    Int_t referenceDetId = (*iMap).second;
    std::string navType  = (*iMap).first;
    Int_t newDetId       = theNewMapNavTypeToDetId[navType];
    if( referenceDetId != newDetId ) {
      theOutputFile << "ERROR: The detid's associated to the same navtype do not correspond"   << std::endl;
      theOutputFile << " reference: " << " detid " << referenceDetId << " navtype " << navType << std::endl;
      theOutputFile << "   new:     " << " detid " << newDetId       << " navtype " << navType << std::endl;
    }
  }
  //
  // compare the position-navtype maps
  for( std::map< std::string , Int_t >::iterator iMap = theReferenceMapNavTypeToDetId.begin();
       iMap != theReferenceMapNavTypeToDetId.end(); iMap++ ) {
    //
    std::string navType = (*iMap).first;
    //
    Double_t referenceX = theReferenceMapNavTypeToGlobalPosition_X[navType];
    Double_t referenceY = theReferenceMapNavTypeToGlobalPosition_Y[navType];
    Double_t referenceZ = theReferenceMapNavTypeToGlobalPosition_Z[navType];
    Double_t newX       = theNewMapNavTypeToGlobalPosition_X[navType];
    Double_t newY       = theNewMapNavTypeToGlobalPosition_Y[navType];
    Double_t newZ       = theNewMapNavTypeToGlobalPosition_Z[navType];
    //
    /*    std::cout << " reference: "
	  << " x = "     << referenceX
	  << " y = "     << referenceY
	  << " z = "     << referenceZ
	  << " navtype " << navType
	  << std::endl;
	  std::cout << "   new:     "
	  << " x = "     << newX
	  << " y = "     << newY
	  << " z = "     << newZ
	  << " navtype " << navType
	  << std::endl;
    */
    //
    if( fabs(referenceX-newX) > tolerance
	||
	fabs(referenceY-newY) > tolerance
	||
	fabs(referenceZ-newZ) > tolerance ) {
      theOutputFile << "ERROR: The positions associated to the same navtype do not correspond" << std::endl;
      theOutputFile 
	<< " reference: "
	<< " x = "     << referenceX
	<< " y = "     << referenceY
	<< " z = "     << referenceZ
	<< " navtype " << navType
	<< std::endl;
      theOutputFile
	<< "   new:     "
	<< " x = "     << newX
	<< " y = "     << newY
	<< " z = "     << newZ
	<< " navtype " << navType
	<< std::endl;
    }
    //
  }
  //
}
