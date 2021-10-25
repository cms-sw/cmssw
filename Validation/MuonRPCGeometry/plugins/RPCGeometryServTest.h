/**
 *
 *  \author M. Maggi - INFN Bari
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <map>

class RPCGeometry;
class MuonGeometryRecord;
class RPCGeometryServTest : public edm::one::EDAnalyzer<> {
public:
  RPCGeometryServTest(const edm::ParameterSet& pset);

  ~RPCGeometryServTest() override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const std::string& myName() { return myName_; }

private:
  const int dashedLineWidth_;
  const std::string dashedLine_;
  const std::string myName_;
  std::map<int, std::pair<double, double> > barzranges;
  std::map<int, std::pair<double, double> > forRranges;
  std::map<int, std::pair<double, double> > bacRranges;
  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeomToken_;
};
