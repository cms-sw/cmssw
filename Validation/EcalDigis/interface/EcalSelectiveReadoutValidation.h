#ifndef EcalSelectiveReadoutValidation_H
#define EcalSelectiveReadoutValidation_H

/*
 * \file EcalSelectiveReadoutValidation.h
 *
 * $Date: 2006/10/26 08:30:31 $
 * $Revision: 1.9 $
 *
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

// 
// #include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


// #include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "DataFormats/EcalDetId/interface/EcalScDetId.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"

#include <string>

class DaqMonitorBEInterface;
class MonitorElement;
class EBDetId;
class EEDetId;

class EcalSelectiveReadoutValidation: public edm::EDAnalyzer{

public:

  /// Constructor
  EcalSelectiveReadoutValidation(const edm::ParameterSet& ps);

  /// Destructor
  ~EcalSelectiveReadoutValidation();

protected:

  /// Analyze
  void analyze(const edm::Event& e, const edm::EventSetup& c);

  // BeginJob
  void beginJob(const edm::EventSetup& c);

  // EndJob
  void endJob(void);

private:
  enum subdet_t {EB, EE};

  template<class T, class U>
  void anaDigi(const T& frame, const U& srFlagColl);
  
  void anaDigiInit();

  double getEcalEventSize(double nReadXtals) const{
    return getDccOverhead(EB)*nEbDccs+getDccOverhead(EE)*nEeDccs
      + nReadXtals*getBytesPerCrystal()
      + (nEeRus+nEbRus)*8;
  }
  
  double getEbEventSize(double nReadXtals) const{
    return getDccOverhead(EB)*nEbDccs + nReadXtals*getBytesPerCrystal()
      + nEbRus*8;
  }
  
  double getEeEventSize(double nReadXtals) const{
    return getDccOverhead(EE)*nEeDccs + nReadXtals*getBytesPerCrystal()
      + nEeRus*8;
  }
  
  double getDccOverhead(subdet_t subdet) const{
    //  return (subdet==EB?34:25)*8;
    return (subdet==EB?33:51)*8;
  }

  double getBytesPerCrystal() const{
    return 3*8;
  }
  
  double getDccEventSize(int iDcc0, double nReadXtals) const{
    subdet_t subdet;
  if(iDcc0<9 || iDcc0>=45){
    subdet = EE;
  } else{
    subdet = EB;
  }
  return getDccOverhead(subdet)+nReadXtals*getBytesPerCrystal()
    + getRuCount(iDcc0)*8;
  }

  int getRuCount(int iDcc0) const;

  unsigned dccNum(const DetId& xtalId) const;

  int iEta2cIndex(int iEta) const{
    return (iEta<0)?iEta+85:iEta+84;
  }

  int iPhi2cIndex(int iPhi) const{
    return iPhi-1;
  }

  int iXY2cIndex(int iX) const{
    return iX-1;
  }

  int cIndex2iXY(int iX0) const{
    return iX0+1;
  }
  int cIndex2iEta(int i) const{
    return (i<85)?i-85:i-84;
  }

  int cIndex2iPhi(int i) const {
    return i+1;
  }
  
  EcalScDetId superCrystalOf(const EEDetId& xtalId) const;
  
  EcalTrigTowerDetId readOutUnitOf(const EBDetId& xtalId) const;
  
  EcalScDetId readOutUnitOf(const EEDetId& xtalId) const;
  
private:
  bool verbose_;
  
  DaqMonitorBEInterface* dbe_;
  
  std::string outputFile_;

  edm::InputTag ebDigiCollection_;
  edm::InputTag eeDigiCollection_;
  edm::InputTag ebSrFlagCollection_;
  edm::InputTag eeSrFlagCollection_;
  
  MonitorElement* meDccVol_;
  MonitorElement* meVol_;
  MonitorElement* meVolB_;
  MonitorElement* meVolE_;
  MonitorElement* meVolBLI_;
  MonitorElement* meVolELI_;
  MonitorElement* meVolLI_;
  MonitorElement* meVolBHI_;
  MonitorElement* meVolEHI_;
  MonitorElement* meVolHI_;

  const EcalTrigTowerConstituentsMap * triggerTowerMap_;

  static const unsigned nDccs = 54;
  
  int nEb_;
  int nEe_;
  int nEeLI_;
  int nEeHI_;
  int nEbLI_;
  int nEbHI_;
  int nPerDcc_[nDccs];
  
  /// number of bytes in 1 kByte:
  static const int kByte_ = 1024;

  static const int nEbDccs = 36;
  static const int nEeDccs = 18;
  static const int nEbRus = 36*68;
  static const int nEeRus = 2*(34+32+33+33+32+34+33+34+33);
};

#endif //EcalSelectiveReadoutValidation_H not defined

