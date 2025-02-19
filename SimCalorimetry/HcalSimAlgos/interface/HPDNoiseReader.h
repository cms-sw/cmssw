#ifndef HPDLibrary_HPDNoiseReader_h
#define HPDLibrary_HPDNoiseReader_h

// --------------------------------------------------------
// Engine to read HPD noise events from the library
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseReader.h,v 1.3 2008/07/21 18:30:03 tyetkin Exp $
// --------------------------------------------------------

#include <string>
#include <vector>

class TFile;
class TTree;
class HPDNoiseData;

class HPDNoiseReader {
 public:
  typedef int Handle;
  HPDNoiseReader (const std::string& fFileName);
  ~HPDNoiseReader ();

  /// all HPD instances in the library
  std::vector<std::string> allNames () const;
  /// get handle to access data for one HPD instance
  Handle getHandle (const std::string& fName);
  /// test if handle is valid
  bool valid (Handle fHandle) const {return fHandle >= 0;}
  /// discharge rate for the instance
  float dischargeRate (Handle fHandle) const;
  /// ionfeedback rate for the instance
  float ionFeedbackFirstPeakRate (Handle fHandle) const;
  float ionFeedbackSecondPeakRate (Handle fHandle) const;
  /// ithermal/field emission rate for the instance
  float emissionRate (Handle fHandle) const;
  /// total number of noise events for the HPD instance in the library
  unsigned long totalEntries (Handle fHandle) const;
  /// retrive one entry from the sequentially
  void getEntry (Handle fHandle, HPDNoiseData** fData);
  /// retrive one entry from the library directly
  void getEntry (Handle fHandle, unsigned long fEntry, HPDNoiseData** fData);
  
 private:
  HPDNoiseReader (const HPDNoiseReader&); // no copy
  HPDNoiseReader& operator=(const HPDNoiseReader&); // no copy
  void grabEntry (Handle fHandle, unsigned long fEntry);

  TFile* mFile;
  std::vector <TTree*> mTrees;
  std::vector<float> mDischargeRate;//HPD discharge rate
  std::vector<float> mIonFeedbackFirstPeakRate;//HPD ion feedback rate
  std::vector<float> mIonFeedbackSecondPeakRate;//HPD ion feedback rate
  std::vector<float> mElectronEmissionRate;//HPD thermal electron emission rate
  std::vector <size_t> mCurrentEntry;
  HPDNoiseData* mBuffer;
};

#endif
