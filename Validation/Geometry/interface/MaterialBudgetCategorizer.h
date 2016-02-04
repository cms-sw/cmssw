///////////////////////////////////////////////////////////////////////////////
// File : MaterialBudgetCategorizer.h
// Author: T.Boccali  11.12.01
// Description:
// Modifications:
///////////////////////////////////////////////////////////////////////////////

#ifndef MaterialBudgetCategorizer_h
#define MaterialBudgetCategorizer_h 1

#include<string>
#include<map>
#include<vector>

class  MaterialBudgetCategorizer {

 public:
  MaterialBudgetCategorizer();
  
  int volume(std::string s){return theVolumeMap[s];}
  int material(std::string s){return theMaterialMap[s];}
  // rr
  std::vector<float> x0fraction(std::string s){return theX0Map[s];}
  std::vector<float> l0fraction(std::string s){return theL0Map[s];}
  // rr
 private:
  void buildMaps();
  void buildCategoryMap(std::string theMaterialFileName, std::map<std::string,std::vector<float> >& theMap);
  std::map<std::string,int> theVolumeMap, theMaterialMap;
  // rr
  std::map<std::string,std::vector<float> > theX0Map;
  std::map<std::string,std::vector<float> > theL0Map;
  // rr
};

#endif

    
