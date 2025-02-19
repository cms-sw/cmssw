#ifndef MaterialBudgetFormat_h
#define MaterialBudgetFormat_h 1

#include <string>

class MaterialBudgetData;


class MaterialBudgetFormat {
public:

  MaterialBudgetFormat( MaterialBudgetData* data );   
  virtual ~MaterialBudgetFormat(){ }

  virtual void fillStartTrack() = 0;
  virtual void fillPerStep() = 0;
  virtual void fillEndTrack() = 0;
  
 protected:
  MaterialBudgetData* theData;
  std::string theFileName;
};

#endif
