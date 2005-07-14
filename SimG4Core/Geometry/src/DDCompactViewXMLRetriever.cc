#include "SimG4Core/Geometry/interface/DDCompactViewXMLRetriever.h"

#include "DetectorDescription/DDBase/interface/DDdebug.h"
#include "DetectorDescription/DDParser/interface/DDLParser.h"
#include "DetectorDescription/DDCore/interface/DDCompactView.h"
#include "DetectorDescription/DDCore/interface/DDSpecifics.h"
#include "DetectorDescription/DDParser/interface/DDLConfiguration.h"
#include "DetectorDescription/DDAlgorithm/src/AlgoInit.h"

#include <memory>

using namespace edm::eventsetup;

DDCompactViewXMLRetriever::DDCompactViewXMLRetriever(std::string & configfile) 
{
    DDLParser * parser = DDLParser::instance();
    AlgoInit();
    DDLConfiguration cf;
    int result1 = cf.readConfig(configfile);
    int result2 = parser->parse(cf);
    std::cout << " and parsed " << std::endl;    
    if (result2 != 0) 
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

