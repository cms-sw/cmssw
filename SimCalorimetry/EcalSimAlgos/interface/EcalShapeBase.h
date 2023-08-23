#ifndef EcalSimAlgos_EcalShapeBase_h
#define EcalSimAlgos_EcalShapeBase_h

#include <vector>
//#include<stdexcept>
#include "SimCalorimetry/CaloSimAlgos/interface/CaloVShape.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalSimPulseShape.h"
#include "CondFormats/DataRecord/interface/EcalSimPulseShapeRcd.h"

#include <iostream>
#include <fstream>
/**
   \class EcalShape
   \brief  shaper for Ecal
   
   The constructor has been updated on CMSSW10X (2018) as to expect a bool, 
   set this to true if DB conditions will be used to retrive the shape array 
   and sampling time from the DB or false if the Phase I hard coded arrays
   with 1 ns sampling should be used. -- K. Theofilatos
*/
class EcalShapeBase : public CaloVShape {
public:
  typedef std::vector<double> DVec;

  EcalShapeBase(bool);

  ~EcalShapeBase() override;

  double operator()(double aTime) const override;

  double timeOfThr() const;
  double timeOfMax() const;
  double timeToRise() const override;

  double threshold() const;

  double derivative(double time) const;  // appears to not be used anywhere

  void m_shape_print(const char* fileName) const;
  void setEventSetup(const edm::EventSetup& evtSetup, bool normalize = true);

protected:
  unsigned int timeIndex(double aTime) const;

  void buildMe(const edm::EventSetup* = nullptr, bool normalize = true);

  virtual void fillShape(float& time_interval,
                         double& m_thresh,
                         EcalShapeBase::DVec& aVec,
                         const edm::EventSetup* es) const = 0;
  bool m_useDBShape;

private:
  unsigned int m_firstIndexOverThreshold;
  double m_firstTimeOverThreshold;
  unsigned int m_indexOfMax;
  double m_timeOfMax;
  double m_thresh;
  unsigned int m_kNBinsPerNSec;
  DVec m_shape;
  DVec m_deriv;

  unsigned int m_arraySize;
  unsigned int m_denseArraySize;
  double m_qNSecPerBin;
};

#endif
