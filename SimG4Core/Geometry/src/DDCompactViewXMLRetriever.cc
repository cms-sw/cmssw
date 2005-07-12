#include "SimG4Core/Geometry/interface/DDCompactViewXMLRetriever.h"

#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDParser/interface/DDLParser.h"
#include "DetectorDescription/DDCore/interface/DDCompactView.h"
#include "DetectorDescription/DDCore/interface/DDSpecifics.h"
#include "DetectorDescription/DDParser/interface/DDLConfiguration.h"
#include "DetectorDescription/DDAlgorithm/src/AlgoInit.h"

#include <memory>

using namespace edm::eventsetup;

DDCompactViewXMLRetriever::DDCompactViewXMLRetriever(const edm::ParameterSet & p) 
{
    std::string configfile = p.getParameter<std::string>("SimConfiguration");
    AlgoInit();
    DDLParser * parser = DDLParser::instance();
    DDLConfiguration cf;
    cf.readConfig(configfile);
    int result = parser->parse(cf);
    if (result != 0) 
      throw DDException("DDD-Parser: parsing failed!");
    setWhatProduced(this);
    findingRecord<PerfectGeometryRecord>();
}

DDCompactViewXMLRetriever::~DDCompactViewXMLRetriever() {}

const DDCompactView *
DDCompactViewXMLRetriever::produce(const PerfectGeometryRecord &)
{ return new DDCompactView(); }

void DDCompactViewXMLRetriever::setIntervalFor(const EventSetupRecordKey &,
					       const edm::Timestamp &, 
					       edm::ValidityInterval & oValidity)
{
   edm::ValidityInterval infinity(edm::Timestamp(1), edm::Timestamp::endOfTime());
   oValidity = infinity;
}

