#ifndef MaterialBudgetTrackerHistos_h
#define MaterialBudgetTrackerHistos_h 1

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

class MaterialBudgetTrackerHistos : public MaterialBudgetFormat
{
public:
  
  MaterialBudgetTrackerHistos( std::shared_ptr<MaterialBudgetData> data, 
			       std::shared_ptr<TestHistoMgr> mgr,
			       const std::string& fileName );   
  ~MaterialBudgetTrackerHistos() override{ hend(); }
  
  void fillStartTrack() override;
  void fillPerStep() override;
  void fillEndTrack() override;
  
private:
  
  virtual void book(); 
  virtual void hend(); 
  
  
private:
  int MAXNUMBERSTEPS;
  double* theDmb;
  double* theX;
  double* theY;
  double* theZ;
  double* theVoluId;
  double* theMateId;

  std::shared_ptr<TestHistoMgr> hmgr;

};


#endif
