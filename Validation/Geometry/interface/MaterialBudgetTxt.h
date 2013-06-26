#ifndef MaterialBudgetTxt_h
#define MaterialBudgetTxt_h 1

#include <fstream>

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"


class MaterialBudgetTxt : public MaterialBudgetFormat
{
 public:
  MaterialBudgetTxt( MaterialBudgetData* data, const std::string& fileName );   
  virtual ~MaterialBudgetTxt();

  virtual void fillStartTrack();
  virtual void fillPerStep();
  virtual void fillEndTrack();

 private:
  std::ofstream* theFile;

};

#endif
