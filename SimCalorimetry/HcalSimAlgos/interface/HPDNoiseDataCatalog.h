#ifndef HPDLibrary_HPDNoiseDataCatalog_h
#define HPDLibrary_HPDNoiseDataCatalog_h

// --------------------------------------------------------
// Object to store HPD instance name and noise rate for the instance
// Project: HPD noise library
// Author: F.Ratnikov UMd, Jan. 15, 2008
// $Id: HPDNoiseDataCatalog.h,v 1.1 2008/01/16 02:12:38 fedor Exp $
// --------------------------------------------------------

#include <iostream>
#include <vector>
#include <string>

#include "TObject.h"

class HPDNoiseDataCatalog : public TObject {
 public:
  HPDNoiseDataCatalog () {}
  virtual ~HPDNoiseDataCatalog ();
  
  /// add new HPD instance to the catalog
  void addHpd (const std::string& fName, float fRate) {mHpdName.push_back (fName), mRate.push_back (fRate);}
  /// total number of instances
  size_t size () const {return mRate.size();}
  /// all HPD instance names
  const std::vector<std::string>& allNames () const {return mHpdName;}
  /// get noise rate for the HPD instance
  float getRate (size_t i) const {return (i < mRate.size()) ? mRate[i] : 0.;}
  /// get name of the instance
  const std::string& getName (size_t i) const {return mHpdName[i];}
  /// set noise rate
  void setRate (const std::string& fName, float fRate);
 private:
  std::vector<std::string> mHpdName;
  std::vector<float> mRate;

  ClassDef(HPDNoiseDataCatalog,1)
};

/// printout
std::ostream& operator<< (std::ostream&, const HPDNoiseDataCatalog&);

#endif
