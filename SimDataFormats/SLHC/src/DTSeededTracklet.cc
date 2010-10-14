#include <algorithm>
#include <vector>

//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTL1SimOperation.h"
//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTSeededTracklet.h"
#include "SimDataFormats/SLHC/interface/DTSeededTracklet.h"
#include "SLHCUpgradeSimulations/L1Trigger/interface/DTParameters.h"   


// static member init ************************************************

size_t  DTSeededTracklet::_DTSeededTrackletsCollectionSize = 0; 

//********************************************************************


void DTSeededTracklet::_init() {
  
  for(int l=0; l<StackedLayersInUseTotal; l++){	
    _stub_x[l] = NAN;
    _stub_y[l] = NAN;
    _stub_z[l] = NAN;
    _stub_rho[l]= NAN;
    _stub_phi[l] = NAN;    
    _stub_theta[l] = NAN;
    _stub_direction[l] = GlobalVector();
    _flagMatch[l] = false;
  }

  Stubs_5_3_0 = DTStubMatchPt(std::string("Stubs-5-3-0"));
  Stubs_5_1_0 = DTStubMatchPt(std::string("Stubs-5-1-0"));
  Stubs_3_2_0 = DTStubMatchPt(std::string("Stubs-3-2-0")); 
  Stubs_3_1_0 = DTStubMatchPt(std::string("Stubs-3-1-0")); 
  Stubs_5_3_V = DTStubMatchPt(std::string("Stubs-5-3-V")); 
  Stubs_5_0_V = DTStubMatchPt(std::string("Stubs-5-0-V")); 
  Stubs_3_0_V = DTStubMatchPt(std::string("Stubs-3-0-V"));
  Mu_5_0   = DTStubMatchPt(std::string("Mu-5-0"));
  Mu_3_0   = DTStubMatchPt(std::string("Mu-3-0"));
  Mu_2_0   = DTStubMatchPt(std::string("Mu-2-0"));
  Mu_1_0   = DTStubMatchPt(std::string("Mu-1-0"));
  Mu_5_V   = DTStubMatchPt(std::string("Mu-5-V"));
  Mu_3_V   = DTStubMatchPt(std::string("Mu-3-V"));
  Mu_2_V   = DTStubMatchPt(std::string("Mu-3-V"));
  Mu_1_V   = DTStubMatchPt(std::string("Mu-1-V"));
  Mu_0_V   = DTStubMatchPt(std::string("Mu-0-V"));
  IMu_5_0   = DTStubMatchPt(std::string("IMu-5-0"));
  IMu_3_0   = DTStubMatchPt(std::string("IMu-3-0"));
  IMu_2_0   = DTStubMatchPt(std::string("IMu-2-0"));
  IMu_1_0   = DTStubMatchPt(std::string("IMu-1-0"));
  IMu_5_V   = DTStubMatchPt(std::string("IMu-5-V"));
  IMu_3_V   = DTStubMatchPt(std::string("IMu-3-V"));
  IMu_2_V   = DTStubMatchPt(std::string("IMu-3-V"));
  IMu_1_V   = DTStubMatchPt(std::string("IMu-1-V"));
  IMu_0_V   = DTStubMatchPt(std::string("IMu-0-V"));
  mu_5_0  = DTStubMatchPt(std::string("mu-5-0"));
  mu_3_0  = DTStubMatchPt(std::string("mu-3-0"));
  mu_2_0  = DTStubMatchPt(std::string("mu-2-0"));
  mu_1_0  = DTStubMatchPt(std::string("mu-1-0"));
  mu_5_V  = DTStubMatchPt(std::string("mu-5-V"));
  mu_3_V  = DTStubMatchPt(std::string("mu-3-V"));
  mu_2_V  = DTStubMatchPt(std::string("mu-2-V"));
  mu_1_V  = DTStubMatchPt(std::string("mu-1-V"));
  mu_0_V  = DTStubMatchPt(std::string("mu-0-V"));
  only_Mu_V =  DTStubMatchPt(std::string("only-Mu-V"));

  _theAllStubsPtVariety = PtVariety();
  _theDTmuAndStubsPtVariety = PtVariety();
  _theDTMuAndStubsPtVariety = PtVariety();
  _theDTIMuAndStubsPtVariety = PtVariety();


}



// copy constructor
DTSeededTracklet::DTSeededTracklet(const DTSeededTracklet& t) {
  _theDTTracklet = DTTracklet(t.theDTTracklet());
  _theStubTracklet = StubTracklet(t.theStubTracklet());
  _theCoreStubTracklet = StubTracklet(t.theCoreStubTracklet());
  _theDTTracklet_size = _theDTTracklet.size();
  _theStubTracklet_size = _theStubTracklet.size();
  _theCoreStubTracklet_size = _theCoreStubTracklet.size();
  _DTSeededTrackletsCollectionSize = t.getDTSeededTrackletsCollectionSize();
  _theAllStubsPtVariety = t._theAllStubsPtVariety;
  _theDTmuAndStubsPtVariety = t._theDTmuAndStubsPtVariety;
  _theDTMuAndStubsPtVariety = t._theDTMuAndStubsPtVariety;
  _theDTIMuAndStubsPtVariety = t._theDTIMuAndStubsPtVariety;

  Stubs_5_3_0 = DTStubMatchPt(t.Stubs_5_3_0);
  Stubs_5_1_0 = DTStubMatchPt(t.Stubs_5_1_0);
  Stubs_3_2_0 = DTStubMatchPt(t.Stubs_3_2_0); 
  Stubs_3_1_0 = DTStubMatchPt(t.Stubs_3_1_0); 
  Stubs_5_3_V = DTStubMatchPt(t.Stubs_5_3_V); 
  Stubs_5_0_V = DTStubMatchPt(t.Stubs_5_0_V); 
  Stubs_3_0_V = DTStubMatchPt(t.Stubs_3_0_V);
  Mu_5_0   = DTStubMatchPt(t.Mu_5_0);
  Mu_3_0   = DTStubMatchPt(t.Mu_3_0);
  Mu_2_0   = DTStubMatchPt(t.Mu_2_0);
  Mu_1_0   = DTStubMatchPt(t.Mu_1_0);
  Mu_5_V   = DTStubMatchPt(t.Mu_5_V);
  Mu_3_V   = DTStubMatchPt(t.Mu_3_V);
  Mu_2_V   = DTStubMatchPt(t.Mu_2_V);
  Mu_1_V   = DTStubMatchPt(t.Mu_1_V);
  Mu_0_V   = DTStubMatchPt(t.Mu_0_V);
  IMu_5_0   = DTStubMatchPt(t.IMu_5_0);
  IMu_3_0   = DTStubMatchPt(t.IMu_3_0);
  IMu_2_0   = DTStubMatchPt(t.IMu_2_0);
  IMu_1_0   = DTStubMatchPt(t.IMu_1_0);
  IMu_5_V   = DTStubMatchPt(t.IMu_5_V);
  IMu_3_V   = DTStubMatchPt(t.IMu_3_V);
  IMu_2_V   = DTStubMatchPt(t.IMu_2_V);
  IMu_1_V   = DTStubMatchPt(t.IMu_1_V);
  IMu_0_V   = DTStubMatchPt(t.IMu_0_V);
  mu_5_0   = DTStubMatchPt(t.mu_5_0);
  mu_3_0   = DTStubMatchPt(t.mu_3_0);
  mu_2_0   = DTStubMatchPt(t.mu_2_0);
  mu_1_0   = DTStubMatchPt(t.mu_1_0);
  mu_5_V   = DTStubMatchPt(t.mu_5_V);
  mu_2_V   = DTStubMatchPt(t.mu_2_V);
  mu_1_V   = DTStubMatchPt(t.mu_1_V);
  mu_0_V   = DTStubMatchPt(t.mu_0_V);
  only_Mu_V = DTStubMatchPt(t.only_Mu_V);
}



// Pt ***********************************************************************

void DTSeededTracklet::setPt(const edm::ParameterSet& pSet) {
  StubTracklet::const_iterator ist;
  for(ist = _theCoreStubTracklet.begin(); ist != _theCoreStubTracklet.end(); ist++) {
    int l =  tracker_lay_Id_to_our( (*ist)->layer() );
    _stub_x[l]         = (*ist)->x();
    _stub_y[l]         = (*ist)->y();
    _stub_z[l]         = (*ist)->z();
    _stub_phi[l]       = (*ist)->phi();
    _stub_theta[l]     = (*ist)->theta();
    _stub_rho[l]       = (*ist)->rho();
    _stub_direction[l] = (*ist)->direction();
    _flagMatch[l]      = true;
  }
  vector<string> labels = 
    pSet.getUntrackedParameter<std::vector<std::string> >("labels");
  for(size_t s=0; s<labels.size(); s++) {
    DTStubMatchPt* aPt = new DTStubMatchPt();
    if((labels[s]) == string("only-Mu-V")) {
      float rB = (0.5 * Erre* Erre)/
	((theDTTracklet()[0])->rhoDT()*fabs((theDTTracklet()[0])->bendingDT()));
      only_Mu_V = DTStubMatchPt((theDTTracklet()[0])->station(),
				pSet, 
				(theDTTracklet()[0])->bendingDT(), 
				rB);
    }
    else if((labels[s])[0] == 'm') {
      aPt = new DTStubMatchPt(labels[s], 
			      (theDTTracklet()[0])->station(),
			      pSet, 
			      (theDTTracklet()[0])->xerre(), 
			      (theDTTracklet()[0])->yerre(), 
			      _stub_x, _stub_y, _flagMatch); 
      _theDTmuAndStubsPtVariety.push_back(*aPt); 
    }
    else if((labels[s])[0] == 'M') {
      aPt = new DTStubMatchPt(labels[s], 
			      (theDTTracklet()[0])->station(),
			      pSet, 
			      (theDTTracklet()[0])->Xerre(), 
			      (theDTTracklet()[0])->Yerre(), 
			      _stub_x, _stub_y, _flagMatch); 
      _theDTMuAndStubsPtVariety.push_back(*aPt); 
    }
    else if((labels[s])[0] == 'I') {
      aPt = new DTStubMatchPt(labels[s], 
			      (theDTTracklet()[0])->station(),
			      pSet,
			      (theDTTracklet()[0])->XerreI(), 
			      (theDTTracklet()[0])->YerreI(), 
			      _stub_x, _stub_y, _flagMatch);
      _theDTIMuAndStubsPtVariety.push_back(*aPt); 
    }
    else if((labels[s])[0] == 'S') {
      aPt = new DTStubMatchPt(labels[s], (theDTTracklet()[0])->station(),
			      pSet, _stub_x, _stub_y, _flagMatch);
      _theAllStubsPtVariety.push_back(*aPt);
    }

    if(labels[s] == std::string("Stubs-5-3-0")) 
      Stubs_5_3_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-5-1-0")) 
      Stubs_5_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-3-2-0")) 
      Stubs_3_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-3-1-0")) 
      Stubs_3_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-5-3-V")) 
      Stubs_5_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-5-0-V")) 
      Stubs_5_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Stubs-3-0-V")) 
      Stubs_3_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-5-0")) 
      Mu_5_0 = DTStubMatchPt(*aPt);  
    else if(labels[s] == std::string("Mu-3-0")) 
      Mu_3_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-2-0")) 
      Mu_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-1-0")) 
      Mu_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-5-V")) 
      Mu_5_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-3-V")) 
      Mu_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-2-V")) 
      Mu_2_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-1-V")) 
      Mu_1_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("Mu-0-V")) 
      Mu_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-5-0")) 
      IMu_5_0 = DTStubMatchPt(*aPt);  
    else if(labels[s] == std::string("IMu-3-0")) 
      IMu_3_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-2-0")) 
      IMu_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-1-0")) 
      IMu_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-5-V")) 
      IMu_5_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-3-V")) 
      IMu_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-2-V")) 
      IMu_2_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-1-V")) 
      IMu_1_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("IMu-0-V")) 
      IMu_0_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-5-0")) 
      mu_5_0 = DTStubMatchPt(*aPt);  
    else if(labels[s] == std::string("mu-3-0")) 
      mu_3_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-2-0")) 
      mu_2_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-1-0")) 
      mu_1_0 = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-5-V")) 
      mu_5_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-3-V")) 
      mu_3_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-2-V")) 
      mu_2_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-1-V")) 
      mu_1_V = DTStubMatchPt(*aPt);
    else if(labels[s] == std::string("mu-0-V")) 
      mu_0_V = DTStubMatchPt(*aPt);
  }
}





const DTStubMatchPt& DTSeededTracklet::getPtOBJ(std::string const label) const 
{
  for(PtVariety::const_iterator it = _theAllStubsPtVariety.begin(); 
      it != _theAllStubsPtVariety.end(); it++) 
    if( (*it).label() == label)
      return (*it);
  for(PtVariety::const_iterator it = _theDTmuAndStubsPtVariety.begin(); 
      it != _theDTmuAndStubsPtVariety.end(); it++) 
    if( (*it).label() == label)
      return (*it);
  for(PtVariety::const_iterator it = _theDTMuAndStubsPtVariety.begin(); 
      it != _theDTMuAndStubsPtVariety.end(); it++) 
    if( (*it).label() == label)
      return (*it);
  for(PtVariety::const_iterator it = _theDTIMuAndStubsPtVariety.begin(); 
      it != _theDTIMuAndStubsPtVariety.end(); it++) 
    if( (*it).label() == label)
      return (*it);
  DTStubMatchPt* null = new DTStubMatchPt();
  return *null;
}





float const DTSeededTracklet::Pt(std::string const label) const
{
  for(PtVariety::const_iterator it = _theAllStubsPtVariety.begin(); 
      it != _theAllStubsPtVariety.end(); it++) 
    {
      if( (*it).label() == label) {
	return (*it).Pt();
      }
    }
  for(PtVariety::const_iterator it = _theDTmuAndStubsPtVariety.begin(); 
      it != _theDTmuAndStubsPtVariety.end(); it++) 
    {
      if( (*it).label() == label) {
	return (*it).Pt();
      }
    }
  for(PtVariety::const_iterator it = _theDTMuAndStubsPtVariety.begin(); 
      it != _theDTMuAndStubsPtVariety.end(); it++) 
    {
      if( (*it).label() == label) {
	return (*it).Pt();
      }
    }
  for(PtVariety::const_iterator it = _theDTIMuAndStubsPtVariety.begin(); 
      it != _theDTIMuAndStubsPtVariety.end(); it++) 
    {
      if( (*it).label() == label) {
	return (*it).Pt();
      }
    }
  return only_Mu_V.Pt();
  //  return NAN;
}

