#ifndef MaterialBudgetHistos_h
#define MaterialBudgetHistos_h 1

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

class MaterialBudgetHistos : public MaterialBudgetFormat
{
public:

  MaterialBudgetHistos( MaterialBudgetData* data, 
			TestHistoMgr* mgr, 
			const std::string& fileName );   
  ~MaterialBudgetHistos() override{ hend(); }

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

  TestHistoMgr* hmgr;

};


#endif
