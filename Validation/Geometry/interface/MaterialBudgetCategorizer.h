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

class  MaterialBudgetCategorizer {

 public:
  MaterialBudgetCategorizer();

  int volume(std::string s){return theVolumeMap[s];}
  int material(std::string s){return theMaterialMap[s];}
 private:
  void buildMaps();
  std::map<std::string,int> theVolumeMap, theMaterialMap;
};

#endif

    
