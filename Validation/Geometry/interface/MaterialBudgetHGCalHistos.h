#ifndef Validation_Geometry_MaterialBudgetHGCalHistos_h
#define Validation_Geometry_MaterialBudgetHGCalHistos_h 

#include "DetectorDescription/Core/interface/DDFilteredView.h"
#include "DetectorDescription/Core/interface/DDsvalues.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "G4Step.hh"
#include "G4Track.hh"

#include <TH1F.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TProfile2D.h>

#include <string>
#include <vector>

class MaterialBudgetHGCalHistos
{
 public:
  
  MaterialBudgetHGCalHistos( const edm::ParameterSet & );
  virtual ~MaterialBudgetHGCalHistos( void ) {}
  
  void fillBeginJob( const DDCompactView & );
  void fillStartTrack( const G4Track* );
  void fillPerStep( const G4Step * );
  void fillEndTrack( void );
  
 private:
  
  void book( void ); 
  void fillHisto( int );
  std::vector<std::string> getNames( DDFilteredView& );
  bool isItHGC( const std::string & );
  
  static const int         maxSet = 5;
  std::vector<std::string> sensitives, hgcNames, sensitiveHGC;
  bool                     m_fillHistos;
  bool                     m_printSum;
  int                      binEta, binPhi;
  double                   maxEta, etaLow, etaHigh;
  std::vector<std::string> matList;
  std::vector<double>      stepLength, radLength, intLength;
  TH1F                     *me400[maxSet], *me800[maxSet];
  TH2F                     *me1200[maxSet];
  TProfile                 *me100[maxSet], *me200[maxSet], *me300[maxSet];
  TProfile                 *me500[maxSet], *me600[maxSet], *me700[maxSet];
  TProfile2D               *me900[maxSet], *me1000[maxSet],*me1100[maxSet];
  int                      m_id;
  int layer, steps;
  double                   radLen, intLen, stepLen;
  double                   eta, phi;
};


#endif
