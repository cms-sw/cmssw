#ifndef MaterialBudgetHGCalHistos_h
#define MaterialBudgetHGCalHistos_h 1

#include "Validation/Geometry/interface/MaterialBudgetFormat.h"
#include "Validation/Geometry/interface/TestHistoMgr.h"

class MaterialBudgetHGCalHistos : public MaterialBudgetFormat {
public:
  MaterialBudgetHGCalHistos(std::shared_ptr<MaterialBudgetData> data,
                            std::shared_ptr<TestHistoMgr> mgr,
                            const std::string& fileName);
  MaterialBudgetHGCalHistos(std::shared_ptr<MaterialBudgetData> data,
                            std::shared_ptr<TestHistoMgr> mgr,
                            const std::string& fileName,
                            double minZ,
                            double maxZ,
                            int nintZ,
                            double rMin,
                            double rMax,
                            int nrbin,
                            double etaMin,
                            double etaMax,
                            int netabin,
                            double phiMin,
                            double phiMax,
                            int nphibin,
                            double RMin,
                            double RMax,
                            int nRbin);
  ~MaterialBudgetHGCalHistos() override {}
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

  double zMin_, zMax_;
  int nzbin_;
  double rMin_, rMax_;
  int nrbin_;
  double etaMin_, etaMax_;
  int netabin_;
  double phiMin_, phiMax_;
  int nphibin_;
  double RMin_, RMax_;
  int nRbin_;
};

#endif
