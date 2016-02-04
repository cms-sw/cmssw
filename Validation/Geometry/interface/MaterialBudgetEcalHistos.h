#ifndef MaterialBudgetEcalHistos_h
#define MaterialBudgetEcalHistos_h 1

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

#include <string>

class MaterialBudgetEcalHistos : public MaterialBudgetFormat
{
public:
  
  MaterialBudgetEcalHistos( MaterialBudgetData* data, 
			    TestHistoMgr* mgr,
			    const std::string& fileName );   
  virtual ~MaterialBudgetEcalHistos(){ hend(); }
  
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
