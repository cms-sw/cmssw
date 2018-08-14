#ifndef MaterialBudgetFormat_h
#define MaterialBudgetFormat_h 1

#include <string>
#include <memory>

class MaterialBudgetData;


class MaterialBudgetFormat {
public:

  MaterialBudgetFormat( std::shared_ptr<MaterialBudgetData> data );   
  virtual ~MaterialBudgetFormat(){ }

  virtual void fillStartTrack() = 0;
  virtual void fillPerStep() = 0;
  virtual void fillEndTrack() = 0;
  
 protected:
  std::shared_ptr<MaterialBudgetData> theData;
  std::string theFileName;
};

#endif
