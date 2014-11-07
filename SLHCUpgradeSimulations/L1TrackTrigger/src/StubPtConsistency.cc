#include <iostream> 
#include <memory>
//#include <cmath>

#include "SLHCUpgradeSimulations/L1TrackTrigger/interface/StubPtConsistency.h"

namespace StubPtConsistency {

  float getConsistency(TTTrack< Ref_PixelDigi_ > aTrack, const StackedTrackerGeometry* theStackedGeometry, double mMagneticFieldStrength, int nPar) {
    
    double m_ptconsist = 0.0;

    if ( !(nPar==4 || nPar==5)) {
      std::cerr << "Not a valid nPar option!" << std::endl;
      return m_ptconsist;
    }


    // ----------------------------------------------------------------------------------------------------------    
    // define stub 1/pt RMS for different regions 
    // ----------------------------------------------------------------------------------------------------------    
    
    const int nREGION = 3; //barrel, outer disks (dR>60cm), inner disks (dR<60cm)
    const int nLAYER = 6; 
    
    double rms_stub[nREGION][nLAYER] = {
      {0.12, 0.12, 0.10, 0.07, 0.06, 0.06}, //barrel
      {0.08, 0.09, 0.09, 0.10, 0.10, 1.0},  //endcap, outer part of disks, dR>60cm (no layer6)
      {0.21, 0.23, 0.22, 0.21, 0.21, 1.0}   //endcap, inner part of disks, dR<60cm (no layer6)
    };
    

    // ----------------------------------------------------------------------------------------------------------    
    // get charged track pt
    // ---------------------------------------------------------------------------------------------------------- 

    double this_trk_pt = aTrack.getMomentum(nPar).perp();
  
    double this_trk_ptsign = 1.0;
    if (aTrack.getRInv(nPar) < 0) this_trk_ptsign = -1.0;
    
    this_trk_pt = this_trk_pt*this_trk_ptsign;
  
  
    // ----------------------------------------------------------------------------------------------------------    
    // loop over stubs
    // ----------------------------------------------------------------------------------------------------------    

    std::vector< edm::Ref< edmNew::DetSetVector< TTStub< Ref_PixelDigi_ > >, TTStub< Ref_PixelDigi_ > > > theStubRefs = aTrack.getStubRefs();

    for (unsigned int iStub=0; iStub<theStubRefs.size(); iStub++) {
      
      // figure out stub position
      StackedTrackerDetId thisDetId( theStubRefs.at(iStub)->getDetId() );
      GlobalPoint posStub = theStackedGeometry->findGlobalPosition( &(*theStubRefs.at(iStub)) );
      
      float dR = fabs(posStub.x()*posStub.x() + posStub.y()*posStub.y());
      int iRegion = 0;
      if (thisDetId.isEndcap() && (dR > 60.0) ) iRegion = 1;
      else if (thisDetId.isEndcap()) iRegion = 2;
      
      int iLayer = 0;
      if (thisDetId.isBarrel()) iLayer = thisDetId.iLayer()-1;
      else iLayer = thisDetId.iDisk()-1;
      
      
      // read stub invPt RMS
      double this_stub_rms = rms_stub[iRegion][iLayer];
      
      
      // read signed stub pt 
      double this_stub_pt = (double)theStackedGeometry->findRoughPt(mMagneticFieldStrength,&(*theStubRefs.at(iStub)));
      
      double this_stub_ptsign = 1.0;
      double trigBend = theStubRefs.at(iStub)->getTriggerBend();
      
      if (thisDetId.isEndcap() && (posStub.z() > 0)) { 
	if (trigBend>0) this_stub_ptsign = (-1)*this_stub_ptsign;
      }
      else {
	if (trigBend<0) this_stub_ptsign = (-1)*this_stub_ptsign;
      }
      
      this_stub_pt = this_stub_pt*this_stub_ptsign;
      
      if ( this_stub_pt == 0 ) {
	std::cerr << "Could not calculate track-stub pt consistency!" << std::endl;
	m_ptconsist = 0.0;
	break;
      }
      
      // add to variable
      m_ptconsist += (1.0/this_trk_pt - 1.0/this_stub_pt)*(1.0/this_trk_pt - 1.0/this_stub_pt)/(this_stub_rms*this_stub_rms);
      
    }// end loop over stubs
    // ----------------------------------------------------------------------------------------------------------
        
    // return value
    return m_ptconsist;
    
  }

}
