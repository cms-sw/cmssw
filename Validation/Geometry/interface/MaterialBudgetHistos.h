#ifndef MaterialBudgetHistos_h
#define MaterialBudgetHistos_h 1

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

class MaterialBudgetHistos : public MaterialBudgetFormat
{
public:

  MaterialBudgetHistos( std::shared_ptr<MaterialBudgetData> data, 
			std::shared_ptr<TestHistoMgr> mgr, 
			const std::string& fileName );   
  ~MaterialBudgetHistos() override { }

  void fillStartTrack() override;
  void fillPerStep() override;
  void fillEndTrack() override;
  void endOfRun() override;
  
private:

  virtual void book();   

  double* theDmb;
  double* theX;
  double* theY;
  double* theZ;
  double* theVoluId;
  double* theMateId;

  std::shared_ptr<TestHistoMgr> hmgr;

};


#endif
