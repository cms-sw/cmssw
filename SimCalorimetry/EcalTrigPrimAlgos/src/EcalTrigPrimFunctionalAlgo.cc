#include <string>
#include <algorithm>
#include <numeric>

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/CaloTopology/interface/EcalBarrelTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/Navigation/interface/EcalBarrelNavigator.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalTrigPrimFunctionalAlgo.h"
#include "SimCalorimetry/EcalTrigPrimAlgos/interface/EcalFenixLinearizer.h"

#include "DataFormats/EcalDigi/interface/EcalTriggerPrimitiveSample.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"
#include "DataFormats/EcalDetId/interface/EcalTrigTowerDetId.h"


//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::EcalTrigPrimFunctionalAlgo(const edm::EventSetup & setup)
{
  edm::ESHandle<CaloGeometry> theGeometry;
  setup.get<IdealGeometryRecord>().get( theGeometry );
  ebTopology_ = new EcalBarrelTopology(theGeometry);
  ebstrip_=new EcalBarrelFenixStrip(ebTopology_);
  //UB FIXME: configurables
//   static SimpleConfigurable<float> thresh(0.0,"EcalTrigPrim:Threshold");
//   threshold=thresh.value();
//   static SimpleConfigurable<bool> coherence(false,"EcalTrigPrim:CoherenceTest");
  //   // coherence tests
//   bool cohtest=coherence.value();
//   if (cohtest) cTest_=new ETPCoherenceTest();
//   else cTest_=NULL;

}

//----------------------------------------------------------------------

EcalTrigPrimFunctionalAlgo::~EcalTrigPrimFunctionalAlgo() 
{
    delete ebstrip_;
    delete ebTopology_;
}

//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::run(const EBDigiCollection* ebdcol, EcalTrigPrimDigiCollection &  result) {
  
  sumBarrel_.clear();
 
  int nhitsb(0);
  
//   static SimpleConfigurable<float> ratBL(0.8,"ECALBarrel:FGLowEnRatioTh");
//   static SimpleConfigurable<float> ratBH(0.9,"ECALBarrel:FGHighEnRatioTh");
//   static SimpleConfigurable<float> enBL(5.0,"ECALBarrel:FGLowEnTh");
//   static SimpleConfigurable<float> enBH(25.0,"ECALBarrel:FGHighEnTh");
  
//   //SimpleConfigurable<float> ratEL(0.8,"ECALEndcap:FGLowEnRatioTh");
//   //SimpleConfigurable<float> ratEH(0.9,"ECALEndcap:FGHighEnRatioTh");
//   static SimpleConfigurable<float> enEL(5.0,"ECALEndcap:FGLowEnTh");
//   static SimpleConfigurable<float> enEH(25.0,"ECALEndcap:FGHighEnTh");


  
  for(unsigned int i = 0; i < ebdcol->size() ; ++i) {
    const EBDetId & myid=(*ebdcol)[i].id();
    const EcalTrigTowerDetId coarser= myid.tower();
    if(coarser.null())  
      {
	std::cout << "Cell " << myid << " has trivial coarser granularity (probably EFRY corner, not in this tower map; hit ignored)" << std::endl;
	continue;
      }	
	
   nhitsb++;
    fillBarrel(coarser,(*ebdcol)[i]);
  }// loop over all CaloDataFrames
  std::cout << "[EcalTrigPrimFunctionalAlgo] (found " << nhitsb << " frames in " 
  	    << sumBarrel_.size() << " EBRY towers  "  << std::endl;

  
  //   Barrel

    SUMV::const_iterator it = sumBarrel_.begin(); 
    SUMV::const_iterator e = sumBarrel_.end(); 

    int itow=0;
    // loop over all trigger towers
    for(;it!=e;it++) 
      {
        itow++;
	const EcalTrigTowerDetId & thisTower =(*it).first;
	// loop over all strips assigned to this trigger tower
	std::vector<std::vector<int> > striptp;
        std::vector<std::vector<EBDataFrame> > dbgdf;
	for(unsigned int i = 0; i < min(it->second.size(),size_t(ecal_barrel_strips_per_trigger_tower)) ; ++i) 
           {
// 	    // 	    // inventing an id for the strip
// 	    //             // temporary, as long as CaloBase ESTR not yet exists
// 	    //             unsigned int stripnr=stripbasenumber+i+1;
// 	    //             printf("null, stripnr %d\n",stripnr);fflush(stdout);
// 	    //             PCellID pc(stripnr);
// 	    //             printf("nullnull, stripnr %d\n",stripnr);fflush(stdout);
// 	    // 	    //            CellID tpid= CellID(pc);
// 	    //            CellID tpid; // dummy since no corresponding caloBase for strips

	    std::vector<int> tp;
 	    std::vector<EBDataFrame> df=it->second[i];
// 	    // here ebstrip should be configured
// 	    //should become  ebstrip.process(df,tp); with stripnr coded in CellID of the EcalTrigPrim
	    if (df.size()>0) {
	      tp=ebstrip_->process(df,i);
	      striptp.push_back(tp);

 	    }
	   }


	EcalTriggerPrimitiveDigi tptow(thisTower);

	ebtcp_.process(striptp,tptow);
	result.push_back(tptow);
	
      }
}

//----------------------------------------------------------------------

//----------------------------------------------------------------------

void EcalTrigPrimFunctionalAlgo::fillBarrel(const EcalTrigTowerDetId & coarser, const EBDataFrame& samples) 
{
  // here we store a vector of CaloDataFrames for each strip into a vector belonging to the corresponding tower
 
  //  int stripnr = createStripNr( samples.getMyCell());
  int stripnr = createStripNr( samples.id());
  SUMV::iterator it= sumBarrel_.find(coarser);

  if(it==sumBarrel_.end()) 
    {
      for (int i=0;i<ecal_barrel_strips_per_trigger_tower;i++ ) {
 	std::vector<EBDataFrame>  truc;
         sumBarrel_[coarser].push_back(truc);
       } 
    }
  (sumBarrel_[coarser])[stripnr-1].push_back(samples);
    
}

int EcalTrigPrimFunctionalAlgo::createStripNr(const EBDetId &cryst) {
  //  Calculates the stripnr in the barrel .
  //  The stripnumber is found by counting how many times one can go to the east
  // before hitting the east border, starting from the given crystal

       int stripnr =ecal_barrel_strips_per_trigger_tower+1;
       EBDetId tmp=cryst;
       EcalBarrelNavigator nav(tmp,ebTopology_);
       uint32_t  towernumber = EcalTrigTowerDetId(tmp.tower()).rawId();
       while ( towernumber== EcalTrigTowerDetId(tmp.tower()).rawId()) {
  	--stripnr;
	tmp=nav.east();
        if (tmp.null()) {
          break; //no cell east of this one
	}
      }
      return  stripnr;
}




