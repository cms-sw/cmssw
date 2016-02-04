#ifndef HPDLibrary_HPDNoiseDataCatalog_h
#define HPDLibrary_HPDNoiseDataCatalog_h

// --------------------------------------------------------
// Object to store HPD instance name and noise rate for the instance
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseDataCatalog.h,v 1.4 2008/08/04 22:07:08 fedor Exp $
// --------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>

class HPDNoiseDataCatalog {
 public:
  HPDNoiseDataCatalog () {}
  virtual ~HPDNoiseDataCatalog ();
  
  /// add new HPD instance to the catalog
  void addHpd (const std::string& fName, float fDischargeRate, float fIonFeedbackFirstPeakRate, float fIonFeedbackSecondPeakRate, float fElectronEmissionRate){ 
         mHpdName.push_back (fName), mDischargeRate.push_back (fDischargeRate), 
         mIonFeedbackFirstPeakRate.push_back(fIonFeedbackFirstPeakRate),
         mIonFeedbackSecondPeakRate.push_back(fIonFeedbackSecondPeakRate),
	 mElectronEmissionRate.push_back(fElectronEmissionRate);
       }
  /// total number 
  size_t size () const {return mDischargeRate.size();}
  /// all HPD instance names
  const std::vector<std::string>& allNames () const {return mHpdName;}
  /// get noise rate for the HPD instance
  float getDischargeRate (size_t i) const {return (i < mDischargeRate.size()) ? mDischargeRate[i] : 0.;}
  /// get ion feedback noise rate for the HPD instance
  float getIonFeedbackFirstPeakRate (size_t i) const {return (i < mIonFeedbackFirstPeakRate.size()) ? mIonFeedbackFirstPeakRate[i] : 0.;}
  float getIonFeedbackSecondPeakRate (size_t i) const {return (i < mIonFeedbackSecondPeakRate.size()) ? mIonFeedbackSecondPeakRate[i] : 0.;}
  /// get thermal electron emission noise rate for the HPD instance
  float getElectronEmissionRate (size_t i) const {return (i < mElectronEmissionRate.size()) ? mElectronEmissionRate[i] : 0.;}
  /// get name of the instance
  const std::string& getName (size_t i) const {return mHpdName[i];}
  /// set discharge/IonFeedback/Electron emission noise rates
  void setRate (const std::string& fName, float fDischargeRate, float fIonFeedbackFirstPeakRate, float fIonFeedbackSecondPeakRate, float fElectronEmissionRate);
  /// class name
  static const char* className () {return "HPDNoiseDataCatalog";}
  /// object name
  static const char* objectName () {return "catalog";}
 private:
  std::vector<std::string> mHpdName;
  std::vector<float> mDischargeRate;//HPD discharge rate
  std::vector<float> mIonFeedbackFirstPeakRate;//HPD ion feedback rate
  std::vector<float> mIonFeedbackSecondPeakRate;//HPD ion feedback rate
  std::vector<float> mElectronEmissionRate;//HPD thermal electron emission rate

};

/// printout
std::ostream& operator<< (std::ostream&, const HPDNoiseDataCatalog&);

#endif
