#ifndef HPDLibrary_HPDNoiseMaker_h
#define HPDLibrary_HPDNoiseMaker_h

// --------------------------------------------------------
// Engine to store HPD noise events in the library
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: BasicJet.h,v 1.11 2007/09/20 21:04:43 fedor Exp $
// --------------------------------------------------------

#include <string>
#include <vector>

class HPDNoiseData;
class HPDNoiseDataCatalog;
class TFile;
class TTree;

class HPDNoiseMaker {
 public:
  HPDNoiseMaker (const std::string& fFileName);
  ~HPDNoiseMaker ();

  /// define new HPD instance
  int addHpd (const std::string& fName, float fRate);
  /// add new HPD noise event by HPD name
  void newHpdEvent (const std::string& mName, const HPDNoiseData& mData);
  /// add new HPD noise event by HPD index
  void newHpdEvent (size_t i, const HPDNoiseData& mData);
  
 private:
  HPDNoiseMaker (const HPDNoiseMaker&);
  HPDNoiseMaker& operator=(const HPDNoiseMaker&);

  TFile* mFile;
  std::vector <TTree*> mTrees;
  std::vector <std::string> mNames;
  HPDNoiseDataCatalog* mCatalog;
};

#endif
