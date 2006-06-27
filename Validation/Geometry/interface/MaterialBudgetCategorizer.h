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
  // rr
 private:
  void buildMaps();
  std::map<std::string,int> theVolumeMap, theMaterialMap;
  // rr
  std::map<std::string,std::vector<float> > theX0Map;
  // rr
};

#endif

    
