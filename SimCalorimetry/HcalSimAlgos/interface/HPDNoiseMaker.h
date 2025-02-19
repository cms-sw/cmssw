#ifndef HPDLibrary_HPDNoiseMaker_h
#define HPDLibrary_HPDNoiseMaker_h

// --------------------------------------------------------
// Engine to store HPD noise events in the library
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseMaker.h,v 1.3 2008/07/21 18:30:03 tyetkin Exp $
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
  int addHpd (const std::string& fName);
  /// set noise rate for the instance by using different processes
  void setRate (const std::string& fName, float fDischargeRate, float fIonFeedbackFirstPeakRate, float fIonFeedbackSecondPeakRate, float fElectronEmissionRate);
  /// add new HPD noise event by HPD name
  void newHpdEvent (const std::string& mName, const HPDNoiseData& mData);
  /// add new HPD noise event by HPD index
  void newHpdEvent (size_t i, const HPDNoiseData& mData);
  /// get number of stored events by HPD index
  unsigned long totalEntries (const std::string& mName) const;
  
 private:
  HPDNoiseMaker (const HPDNoiseMaker&);
  HPDNoiseMaker& operator=(const HPDNoiseMaker&);

  TFile* mFile;
  std::vector <TTree*> mTrees;
  std::vector <std::string> mNames;
  HPDNoiseDataCatalog* mCatalog;
};

#endif
