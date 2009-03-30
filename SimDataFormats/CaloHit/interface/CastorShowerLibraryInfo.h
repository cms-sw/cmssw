#ifndef CastorShowerLibraryInfo_h
#define CastorShowerLibraryInfo_h

#include "TROOT.h"
#include "Rtypes.h"
#include "TObject.h"
#include "TClass.h"
#include "TDictionary.h"

#include <vector>
#include <string>

  class CastorShowerLibraryInfo : public TObject {
  
  public:
  
    CastorShowerLibraryInfo();
    ~CastorShowerLibraryInfo();
    
    void Clear();
    
    // Data members
    unsigned int             NEv;
    unsigned int         NEnBins;
    unsigned int       NEvPerBin;
    std::vector<double> Energies;
    
    // Setters
    void setNEv(unsigned int n)               { NEv = n; };
    void setNEnBins(unsigned int n)           { NEnBins = n; };
    void setNEvPerBin(unsigned int n)         { NEvPerBin = n; };
    void setEnergies(double en)               { Energies.push_back(en); };
    void setEnergies(std::vector<double> en)  { Energies=en; };
    
    // Accessors
    unsigned int getNEv()              { return NEv; };
    unsigned int getNEnBins()          { return NEnBins; };
    unsigned int getNEvPerBin()        { return NEvPerBin; };
    double getEnergies(int i)          { return Energies[i]; };
    std::vector<double> getEnergies()  { return Energies; };
    
    ClassDef(CastorShowerLibraryInfo,1)
    
  };

#endif
