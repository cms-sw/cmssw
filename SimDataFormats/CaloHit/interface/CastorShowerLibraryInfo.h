#ifndef CastorShowerLibraryInfo_h
#define CastorShowerLibraryInfo_h

#include "TROOT.h"
#include "Rtypes.h"
#include "TObject.h"
#include "TClass.h"
#include "TDictionary.h"

#include <vector>
#include <string>
class SLBin: public TObject {
      public:
             SLBin() {};
             ~SLBin() override {};
// Setters
             void Clear(Option_t * option = "") override {NEvts=NBins=NEvtPerBin=0;Bins.clear();};
             void setNEvts(unsigned int n)      {NEvts = n;};
             void setNBins(unsigned int n)      {NBins = n;};
             void setNEvtPerBin(unsigned int n) {NEvtPerBin=n;};
             void setBin(double val)            {Bins.push_back(val);};
             void setBin(const std::vector<double>& b) {Bins=b;};
// getters
             unsigned int getNEvts()            { return NEvts;};
             unsigned int getNBins()            { return NBins;};
             unsigned int getNEvtPerBin()       { return NEvtPerBin;};
             double               getBin(int i) { return Bins.at(i);};
             std::vector<double>& getBin()      { return Bins;};
      private:
             unsigned int        NEvts;
             unsigned int        NBins;
             unsigned int        NEvtPerBin;
             std::vector<double> Bins;
    ClassDefOverride(SLBin,2);
};

class CastorShowerLibraryInfo : public TObject {
  
  public:
  
    CastorShowerLibraryInfo();
    ~CastorShowerLibraryInfo() override;

    void Clear(Option_t * option = "") override;
    
    // Data members
    SLBin Energy;
    SLBin Eta;
    SLBin Phi;

    ClassDefOverride(CastorShowerLibraryInfo,2);
    
  };

#endif
