#ifndef MaterialBudgetFormat_h
#define MaterialBudgetFormat_h 1

#include <string>
#include <memory>

class MaterialBudgetData;

class MaterialBudgetFormat {

 public:
  MaterialBudgetFormat( std::shared_ptr<MaterialBudgetData> data );
  virtual ~MaterialBudgetFormat() {}

  virtual void fillStartTrack() {}
  virtual void fillPerStep() {}
  virtual void fillEndTrack() {}
  virtual void endOfRun() {}
  
 protected:
  std::shared_ptr<MaterialBudgetData> theData;
  std::string theFileName;

};

#endif
