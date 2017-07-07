#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"



#include "RecoTracker/TkNavigation/interface/TkMSParameterization.h"
#include "RecoTracker/Record/interface/TkMSParameterizationRecord.h"


class TkMSParameterizationTest final : public edm::EDAnalyzer {
public:
   explicit TkMSParameterizationTest(const edm::ParameterSet&){}
private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
};


void
TkMSParameterizationTest::analyze(const edm::Event&, const edm::EventSetup& iSetup) {
  using namespace tkMSParameterization;
  edm::ESHandle<TkMSParameterization> paramH;
  iSetup.get<TkMSParameterizationRecord>().get(paramH);

  auto const & msParam = *paramH;


 // test MSParam
  { 
    std::cout << "\n\nProduced MSParam" << std::endl;
    unsigned short f,t;
    for (auto const & e : msParam()) {
      std::tie(f,t) = unpackLID(e.first);
      std::cout << "from/to " << f << "->" << t << std::endl;
      for(auto const & d : e.second()) {
	std::cout << d().size() << ' ';
      }
      std::cout << std::endl;
    }
    
    int lst[] = {0,1,29};
    for (auto st : lst)
      for (int il=1; il<5;++il) {
	int ll = st+il;
	if (st==0 && il==5) ll=29;
	std::cout << "from/to " << st << "->" << ll <<std::endl;
	auto const & d = *msParam.fromTo(st,ll);
	int lb=0; 
	for(auto const & ld : d()) {
	  lb++;
	  if (ld().empty()) continue;
	  std::cout << lb << ": ";
	  for(auto const & e : ld()) std::cout << e << ' ';
	  std::cout<<std::endl;
	} 
      }
  }
  







};

//define this as a plug-in
DEFINE_FWK_MODULE(TkMSParameterizationTest);
