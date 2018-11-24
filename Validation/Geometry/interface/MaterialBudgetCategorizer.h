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
  MaterialBudgetCategorizer(std::string mode);
  
  int volume(std::string s){return theVolumeMap[s];}
  int material(std::string s){return theMaterialMap[s];}
  // rr
  const std::vector<float> & x0fraction(std::string s){return theX0Map[s];}
  const std::vector<float> & l0fraction(std::string s){return theL0Map[s];}
  // rr
  //HGCal
  const std::vector<float> & HGCalx0fraction(std::string s){return theHGCalX0Map[s];}
  const std::vector<float> & HGCall0fraction(std::string s){return theHGCalL0Map[s];}

 private:
  void buildMaps();
  void buildCategoryMap(std::string theMaterialFileName, std::map<std::string,std::vector<float> >& theMap);
  void buildHGCalCategoryMap(std::string theMaterialFileName, std::map<std::string,std::vector<float> >& theMap);
  std::map<std::string,int> theVolumeMap, theMaterialMap;
  // rr
  std::map<std::string,std::vector<float> > theX0Map;
  std::map<std::string,std::vector<float> > theL0Map;
  // rr
  //HGCal
  std::map<std::string,std::vector<float> > theHGCalX0Map;
  std::map<std::string,std::vector<float> > theHGCalL0Map;

};

#endif

    
