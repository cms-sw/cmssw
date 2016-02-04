#ifndef MaterialBudgetTrackerHistos_h
#define MaterialBudgetTrackerHistos_h 1

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

class MaterialBudgetTrackerHistos : public MaterialBudgetFormat
{
public:
  
  MaterialBudgetTrackerHistos( MaterialBudgetData* data, 
			       TestHistoMgr* mgr,
			       const std::string& fileName );   
  virtual ~MaterialBudgetTrackerHistos(){ hend(); }
  
  virtual void fillStartTrack();
  virtual void fillPerStep();
  virtual void fillEndTrack();
  
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

  TestHistoMgr* hmgr;

};


#endif
