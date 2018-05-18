#ifndef EcalEBTrigPrimClusterAlgo_h
#define EcalEBTrigPrimClusterAlgo_h
/** \class EcalEBTrigPrimClusterAlgo
\author N. Marinelli - Univ. of Notre Dame
 * transforming the existing algo (Run I-II) forPhase II 
 * While the new digitization is not yet implemented, we use the old Digis to make TP per crystal
 *
 ************************************************************/
#include <iostream>
#include <vector>

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "DataFormats/EcalDetId/interface/EcalTriggerElectronicsId.h"
#include "DataFormats/Common/interface/SortedCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"


#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixLinearizer.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixAmplitudeFilter.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixPeakFinder.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixStripFormatEB.h" 
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalFenixTcpFormatCluster.h"
#include "SimCalorimetry/EcalEBTrigPrimAlgos/interface/EcalEBTrigPrimBaseAlgo.h"

#include <map>
#include <utility>



class EcalEBClusterTriggerPrimitiveSample;
class CaloSubdetectorGeometry;
class EBDataFrame;
class SimpleCaloHit;

 
class EcalEBTrigPrimClusterAlgo : public  EcalEBTrigPrimBaseAlgo
{  
 public:
  
  explicit EcalEBTrigPrimClusterAlgo(const edm::EventSetup & setup, int nSamples, int binofmax, bool tcpFormat, bool barrelOnly, bool debug, bool famos);

  virtual ~EcalEBTrigPrimClusterAlgo();

  void run(const edm::EventSetup &, const EBDigiCollection *col, EcalEBClusterTrigPrimDigiCollection & result, 
	   EcalEBClusterTrigPrimDigiCollection & resultTcp, 
	   int dEta, int dPhi,
           double hitNoiseCut,
	   double etCutOnSeed);


  std::vector<uint16_t> makeCluster  (std::vector<std::vector<SimpleCaloHit> >& hitCollection, EcalEBClusterTrigPrimDigiCollection & result, int dEta, int dPhi, double hitNoiseCut,double etCutOnSeed );

 private:
  
  EcalFenixTcpFormatCluster *fenixTcpFormatClu_;
  EcalFenixTcpFormatCluster *getFormatter() const {return fenixTcpFormatClu_;}

  std::vector<std::vector<float> > clusters_out_;

};


class SimpleCaloHit   {

 public:

  

  SimpleCaloHit (int et) {  etInADC_=et;}
  bool stale=false; // Hits become stale once used in clustering algorithm to prevent overlap in clusters
  EBDetId id() const {return id_;}
  void setId(const EBDetId id)  { id_=id;}
  GlobalVector position() {return position_;}
  void setPosition( GlobalVector pos ) {position_=pos;}
  void setEtInADC(int etInADC) { etInADC_= etInADC;}
  int  etInADC() const {return etInADC_;}
  float etInGeV() const {return etInADC_*0.125;} // very temporary. Assumes the old Et_Sat=128 GEV and the 10 bits Et
  float energy() {return etInGeV()/sin(position().theta());}
  


  int dieta(SimpleCaloHit& other) const
  {
    
    if (id().ieta() * other.id().ieta() > 0)
      return id().ieta()-other.id().ieta();
    return  id().ieta()-other.id().ieta()-1;
  };
  inline float dphi(SimpleCaloHit& other) {return reco::deltaPhi(static_cast<float>(position().phi()), static_cast<float>(other.position().phi()));};

  int diphi(SimpleCaloHit& other) const
  {
    // Logic from EBDetId::distancePhi() without the abs()
    int PI = 180;
    int result = id().iphi() - other.id().iphi();
    while (result > PI) result -= 2*PI;
    while (result <= -PI) result += 2*PI;
    return result;
  };
  
  bool operator==(SimpleCaloHit& other) 
  {
    if ( id() == other.id() &&
	 position() == other.position() &&
	 energy() == other.energy()
	 ) return true;
    
    return false;
  };
  



 private:
  EBDetId id_;
  GlobalVector position_; // As opposed to GlobalPoint, so we can add them (for weighted average)
  uint16_t etInADC_;
  
  
};





#endif
