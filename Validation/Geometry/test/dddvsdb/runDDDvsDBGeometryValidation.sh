#! /bin/tcsh

cmsenv

echo " START Geometry Validation"

# $1 is the Global Tag
# $2 is the scenario
# $3 is "round" to round values in comparisons  to 0 if < |1.e7|.
# Omit this option to show differences down to |1.e-23|.

set roundFlag = ''
if ($#argv == 0) then
    set gtag="auto:run1_mc"
    set geometry="GeometryExtended"
else if($#argv == 1) then
    set gtag=`echo ${1}`
    set geometry="GeometryExtended"
else if ($#argv == 2) then
    set gtag=`echo ${1}`
    set geometry=`echo ${2}`
else if ($#argv == 3) then 
    set gtag=`echo ${1}`
    set geometry=`echo ${2}`
    set roundFlag = `echo ${3}`
endif
echo GlobalTag = ${gtag}
echo geometry = ${geometry}
echo roundFlag = ${roundFlag}

set tolerance = '1.0e-7'
# If rounding enabled, tolerance for numerical comparisons. Absolute values less than this are set to 0.

#global tag gtag is assumed to be of the form GeometryWORD such as GeometryExtended or GeometryIdeal
#as of 3.4.X loaded objects in the DB, these correspond to condlabels Extended, Ideal, etc...
# Run 2 Extended condlabel corresponds to GeometryExtended2015 scenario. 
set condlabel = `(echo $geometry | sed -e '{s/Geometry//g}' -e '{s/Plan//g}' -e '{s/[0-9]*//g}')`
echo ${condlabel} " geometry label from db"

set workArea = `(echo $geometry)`
mkdir ${workArea}
cd ${workArea}
set myDir=`pwd`
echo $myDir

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/writehelpers/geometryxmlwriter.py .
echo $geometry
sed -i "{s/GeometryExtended/${geometry}/}" geometryxmlwriter.py >  GeometryValidation.log
cmsRun geometryxmlwriter.py >>  GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrywriter.py .
# cp $CMSSW_BASE/src/CondTools/Geometry/test/geometrywriter.py .
sed -i "{s/GeometryExtended/${geometry}/}" geometrywriter.py >>  GeometryValidation.log
sed -i "{s/geTagXX.xml/geSingleBigFile.xml/g}" geometrywriter.py >>  GeometryValidation.log
cmsRun geometrywriter.py >>  GeometryValidation.log
if ( -e myfile.db ) then
    echo "The local DB file is present" | tee -a GeometryValidation.log
else
    echo "ERROR the local DB file is not present" | tee -a GeometryValidation.log
    exit
endif

echo "Start compare the content of GT and the local DB" | tee -a GeometryValidation.log

# (MEC:1) The following two tests with the diff below them actually make
# sure that the Global Tag (GT) and Local DB XML file blobs are fine... 
# meaning that the full simulation geometry source is fine (XML blob)
# as well as the reco geometries.
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrytest_local.py .
sed -i "{/process.GlobalTag.globaltag/d}" geometrytest_local.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" geometrytest_local.py >> GeometryValidation.log

cmsRun geometrytest_local.py > outLocalDB.log
if ( -s outLocalDB.log ) then
    echo "Local DB access run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of Local DB access test is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrytest_db.py .
sed -i "{/process.GlobalTag.globaltag/d}" geometrytest_db.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" geometrytest_db.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" geometrytest_db.py >> GeometryValidation.log 
cmsRun geometrytest_db.py > outGTDB.log
if ( -s outGTDB.log ) then
    echo "GT DB access run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of GT DB access test is empty" | tee -a GeometryValidation.log
    exit
endif

diff outLocalDB.log outGTDB.log > logDiffLocalvsGT.log
if ( -s logDiffLocalvsGT.log ) then
    echo "WARNING THE CONTENT OF GLOBAL TAG MAY BE DIFFERENT WITH RESPECT TO THE LOCAL DB FILE" | tee -a GeometryValidation.log
    cp $CMSSW_BASE/src/Validation/Geometry/test/dddvsdb/sortXML.sh .
    cp $CMSSW_BASE/src/Validation/Geometry/test/dddvsdb/sortCompositeMaterials.py .
    ./sortXML.sh outLocalDB.log localdb.xml
    ./sortXML.sh outGTDB.log gtdb.xml
    diff localdb.xml gtdb.xml > logDiffLocXMLvsGTXML.log
    sort  localdb.xml > localdb.sort
    sort gtdb.xml > gtdb.sort
    diff localdb.sort gtdb.sort > logDiffLocvsGTSort.log
    echo Examine logDiffLocXMLvsGTXML.log to see the differences in the local and GT XML files. | tee -a GeometryValidation.log
    echo Examine logDiffLocvsGTSort.log to see the differences in sorted content of the local and GT XML files. | tee -a GeometryValidation.log
    echo The two XML files may have real differences, or they may have identical content that is simply re-arranged. | tee -a GeometryValidation.log
    echo Examining these log files can help you determine whether the XML files have significant differences. | tee -a GeometryValidation.log
endif

echo "End compare the content of GT and the local DB" | tee -a GeometryValidation.log

echo "Start Tracker RECO geometry validation" | tee -a GeometryValidation.log

mkdir tkdb
mkdir tkdblocal
mkdir tkddd

cp myfile.db tkdblocal

cd tkdb
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/python/testTrackerModuleInfoDB_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" testTrackerModuleInfoDB_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testTrackerModuleInfoDB_cfg.py >> ../GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testTrackerModuleInfoDB_cfg.py >> ../GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testTrackerModuleInfoDB_cfg.py >> GeometryValidation.log
endif
cmsRun testTrackerModuleInfoDB_cfg.py >> ../GeometryValidation.log
mv testTrackerModuleInfoDB_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from DB run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from DB is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../tkdblocal
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/python/trackerModuleInfoLocalDB_cfg.py .
# cp $CMSSW_BASE/src/Geometry/TrackerGeometryBuilder/test/python/trackerModuleInfoLocalDB_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" trackerModuleInfoLocalDB_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" trackerModuleInfoLocalDB_cfg.py >> ../GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" trackerModuleInfoLocalDB_cfg.py >> ../GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" trackerModuleInfoLocalDB_cfg.py >> GeometryValidation.log
endif
cmsRun trackerModuleInfoLocalDB_cfg.py >> ../GeometryValidation.log
mv trackerModuleInfoLocalDB_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from Local DB run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from Local DB is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../tkddd
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/python/testTrackerModuleInfoDDD_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testTrackerModuleInfoDDD_cfg.py >>  ../GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testTrackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testTrackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testTrackerModuleInfoDDD_cfg.py >> GeometryValidation.log
endif
cmsRun testTrackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log
mv testTrackerModuleInfoDDD_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from DDD run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from DDD is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../
rm -f tkdblocal/myfile.db
diff -r tkdb/ tkddd/ > logTkDiffGTvsDDD.log
if ( -s logTkDiffGTvsDDD.log ) then
    echo "WARNING THE TRACKER RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log
endif

diff -r tkdblocal/ tkddd/ > logTkDiffLocalvsDDD.log
if ( -s logTkDiffLocalvsDDD.log ) then
    echo "WARNING THE TRACKER RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff -r tkdb/ tkdblocal/ > logTkDiffGTvsLocal.log
if ( -s logTkDiffGTvsLocal.log ) then
    echo "WARNING THE TRACKER RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End Tracker RECO geometry validation" | tee -a GeometryValidation.log

echo "Start DT RECO geometry validation" | tee -a GeometryValidation.log

# cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromDB_cfg.py .  
cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometryFromDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testDTGeometryFromDB_cfg.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testDTGeometryFromDB_cfg.py >> GeometryValidation.log
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testDTGeometryFromDB_cfg.py >> GeometryValidation.log
endif
cmsRun testDTGeometryFromDB_cfg.py > outDB_DT.log
if ( -s outDB_DT.log ) then
    echo "DT test from DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from DB is empty" | tee -a GeometryValidation.log
    exit
endif

# cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromLocalDB_cfg.py .  
cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromLocalDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometryFromLocalDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testDTGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testDTGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testDTGeometryFromLocalDB_cfg.py >> GeometryValidation.log
endif
cmsRun testDTGeometryFromLocalDB_cfg.py > outLocalDB_DT.log
if ( -s outDB_DT.log ) then
    echo "DT test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

# cp $CMSSW_BASE/src/Geometry/DTGeometry/test/testDTGeometry_cfg.py .
cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometry_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testDTGeometry_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometry_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testDTGeometry_cfg.py >> GeometryValidation.log 
if ( "${roundFlag}" == round ) then                                                               
  sed -i "/tolerance/s/1.0e-23/${tolerance}/" testDTGeometry_cfg.py >> GeometryValidation.log
endif
cmsRun testDTGeometry_cfg.py > outDDD_DT.log
if ( -s outDDD_DT.log ) then
    echo "DT test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outDB_DT.log outDDD_DT.log > logDTDiffGTvsDDD.log
if ( -s logDTDiffGTvsDDD.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outLocalDB_DT.log outDDD_DT.log > logDTDiffLocalvsDDD.log
if ( -s logDTDiffLocalvsDDD.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outDB_DT.log outLocalDB_DT.log > logDTDiffGTvsLocal.log
if ( -s logDTDiffGTvsLocal.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND  LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End DT RECO geometry validation" | tee -a GeometryValidation.log

echo "Start CSC RECO geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testCSCGeometryFromDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testCSCGeometryFromDB_cfg.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testCSCGeometryFromDB_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometryFromDB_cfg.py > outDB_CSC.log
if ( -s outDB_CSC.log ) then
    echo "CSC test from GT DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from GT DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromLocalDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testCSCGeometryFromLocalDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testCSCGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testCSCGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometryFromLocalDB_cfg.py > outLocalDB_CSC.log
if ( -s outLocalDB_CSC.log ) then
    echo "CSC test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/CSCGeometry/test/testCSCGeometry_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testCSCGeometry_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testCSCGeometry_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testCSCGeometry_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometry_cfg.py > outDDD_CSC.log
if ( -s outDDD_CSC.log ) then
    echo "CSC test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outDB_CSC.log outDDD_CSC.log > logCSCDiffGTvsDDD.log
if ( -s logCSCDiffGTvsDDD.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outLocalDB_CSC.log outDDD_CSC.log > logCSCDiffLocalvsDDD.log
if ( -s logCSCDiffLocalvsDDD.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outLocalDB_CSC.log outDB_CSC.log > logCSCDiffLocalvsGT.log
if ( -s logCSCDiffLocalvsGT.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End CSC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start RPC RECO geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometryFromDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testRPCGeometryFromDB_cfg.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testRPCGeometryFromDB_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometryFromDB_cfg.py > outDB_RPC.log
if ( -s outDB_RPC.log ) then
    echo "RPC test from GT DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from GT DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromLocalDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometryFromLocalDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testRPCGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testRPCGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometryFromLocalDB_cfg.py > outLocalDB_RPC.log
if ( -s outLocalDB_RPC.log ) then
    echo "RPC test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometry_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testRPCGeometry_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometry_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testRPCGeometry_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometry_cfg.py > outDDD_RPC.log
if ( -s outDDD_RPC.log ) then
    echo "RPC test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outDB_RPC.log outDDD_RPC.log > logRPCDiffGTvsDDD.log
if ( -s logRPCDiffGTvsDDD.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outLocalDB_RPC.log outDDD_RPC.log > logRPCDiffLocalvsDDD.log
if ( -s logRPCDiffLocalvsDDD.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outLocalDB_RPC.log outDB_RPC.log > logRPCDiffLocalvsDB.log
if ( -s logRPCDiffLocalvsDB.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End RPC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start CALO RECO geometry validation" | tee -a GeometryValidation.log

cp myfile.db $CMSSW_BASE/src/Geometry/CaloEventSetup/test/
cd $CMSSW_BASE/src/Geometry/CaloEventSetup/
cd data
wget -i download.url
cd ../test
source setup.scr >> ${myDir}/GeometryValidation.log
sed -i "{s/GeometryExtended/${geometry}/}" runTestCaloGeometryDDD_cfg.py >> ${myDir}/GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" runTestCaloGeometryDDD_cfg.py >> ${myDir}/GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" runTestCaloGeometryDDD_cfg.py >> ${myDir}/GeometryValidation.log 
cmsRun runTestCaloGeometryDDD_cfg.py > GeometryCaloValidationDDD.log
if ( -s GeometryCaloValidationDDD.log ) then
    echo "CALO test from DDD run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO test from DDD is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif

sed -i "{/process.GlobalTag.globaltag/d}" runTestCaloGeometryDB_cfg.py >> ${myDir}/GeometryValidation.log
sed -i "s/auto:startup/${gtag}/" runTestCaloGeometryDB_cfg.py >> ${myDir}/GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" runTestCaloGeometryDB_cfg.py >> ${myDir}/GeometryValidation.log 
cmsRun runTestCaloGeometryDB_cfg.py > GeometryCaloValidationDB.log
if ( -s GeometryCaloValidationDB.log ) then
    echo "CALO test from GT DB run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO test from GT DB is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif

sed -i "{/process.GlobalTag.globaltag/d}" runTestCaloGeometryLocalDB_cfg.py >> ${myDir}/GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" runTestCaloGeometryLocalDB_cfg.py >> ${myDir}/GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" runTestCaloGeometryLocalDB_cfg.py >> ${myDir}/GeometryValidation.log 
cmsRun runTestCaloGeometryLocalDB_cfg.py > GeometryCaloValidationLocal.log
if ( -s GeometryCaloValidationLocal.log ) then
    echo "CALO Local test from Local DB run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO test from Local DB is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif
cd ${myDir}

less $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDDD.log >> GeometryValidation.log
less $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDB.log >> GeometryValidation.log
less $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log >> GeometryValidation.log

grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDDD.log > CALODDDError.log 
grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDB.log > CALODBError.log 
grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log > CALOLocalError.log 

rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDDD.log
rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationDB.log
rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log
source $CMSSW_BASE/src/Geometry/CaloEventSetup/test/clean.scr

if ( -s CALODDDError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN DDD AND REF" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      

if ( -s CALODBError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN GT DB AND REF" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      

if ( -s CALOLocalError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN LOCAL DB AND REF" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      
                                                                                              
echo "End CALO RECO geometry validation" | tee -a GeometryValidation.log

echo "Start Simulation geometry validation" | tee -a GeometryValidation.log

# (MEC:2) see (MEC:1) Since the global tag versus the local database
# blobs have been verified, it is possible to argue that 
# there is really no reason to check those two blobs using this method.
# However, in this test, the actual DDD is built and dumped for each 
# of standard (STD, i.e. the list of smaller xml files), the "BIG" XML
# File (BDB, i.e. the one prepped to become a blob), the local database
# file blob (LocDB, after Big is loaded into the local database), and
# the file blob that comes from the global tag that was provided to the 
# script (GTDB, could be same or older version).

# Old version. Is it obsolete?
# echo "Here I am " > readXML.expected
# echo "Top Most LogicalPart =cms:OCMS " >> readXML.expected
# echo " mat=materials:Air" >> readXML.expected
# echo " solid=cms:OCMS   Polycone_rrz: 0 6.28319 -450000 0 1000 -27000 0 1000 -27000 0 17500 27000 0 17500 27000 0 1000 450000 0 1000 " >> readXML.expected
# echo "After the GeoHistory in the output file dumpGeoHistoryOnRead you will see x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33" >> readXML.expected
# echo "finished" >> readXML.expected

if ( ${geometry} == GeometryExtended2021 ) then                                                               
  cat > readXML.expected <<END_OF_TEXT  
Here I am 
Top Most LogicalPart =cms:OCMS 
 mat=materials:Air
 solid=cms:OCMS   Box:  xhalf[cm]=10100 yhalf[cm]=10100 zhalf[cm]=45000
After the GeoHistory in the output file dumpGeoHistoryOnRead you will see x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33
finished
END_OF_TEXT

else

  cat > readXML.expected <<END_OF_TEXT  
Here I am 
Top Most LogicalPart =cms:OCMS 
 mat=materials:Air
 solid=cms:OCMS   Polycone_rrz:  startPhi[deg]=0 dPhi[deg]=360 Sizes[cm]=-45000 0 100 -2700 0 100 -2700 0 1750 2700 0 1750 2700 0 100 45000 0 100 
After the GeoHistory in the output file dumpGeoHistoryOnRead you will see x, y, z, r11, r12, r13, r21, r22, r23, r31, r32, r33
finished
END_OF_TEXT

endif

cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/readExtendedAndDump.py .
sed -i "{s/GeometryExtended/${geometry}/}" readExtendedAndDump.py >>  GeometryValidation.log
cmsRun readExtendedAndDump.py > readXMLAndDump.log

cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testReadXMLFromGTDB.py .
# cp $CMSSW_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testReadXMLFromGTDB.py .
sed -i "{/process.GlobalTag.globaltag/d}" testReadXMLFromGTDB.py >> GeometryValidation.log
sed -i "{/process.XMLFromDBSource.label/d}" testReadXMLFromGTDB.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testReadXMLFromGTDB.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('${condlabel}')" testReadXMLFromGTDB.py >> GeometryValidation.log
cmsRun testReadXMLFromGTDB.py > readXMLfromGTDB.log

cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testReadXMLFromDB.py .
# cp $CMSSW_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testReadXMLFromDB.py .
sed -i "{/process.GlobalTag.globaltag/d}" testReadXMLFromDB.py >> GeometryValidation.log
sed -i "{/process.XMLFromDBSource.label/d}" testReadXMLFromDB.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\from Configuration.AlCa.GlobalTag import GlobalTag\nprocess.GlobalTag = GlobalTag(process.GlobalTag, '${gtag}', '')" testReadXMLFromDB.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.XMLFromDBSource.label = cms.string('')" testReadXMLFromDB.py >> GeometryValidation.log
cmsRun testReadXMLFromDB.py > readXMLfromLocDB.log

cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/readBigXMLAndDump.py .
sed -i "{/geomXMLFiles = cms.vstring('GeometryReaders\/XMLIdealGeometryESSource\/test\/fred.xml'),/d}" readBigXMLAndDump.py >> GeometryValidation.log
sed -i "/XMLIdealGeometryESSource/ a\\t\tgeomXMLFiles=cms.vstring('${workArea}\/geSingleBigFile.xml')," readBigXMLAndDump.py >>  GeometryValidation.log
cmsRun readBigXMLAndDump.py > readBigXMLAndDump.log

diff readXMLAndDump.log readXML.expected > diffreadXMLSTD.log
diff readXMLfromGTDB.log readXML.expected > diffreadXMLGTDB.log
diff readXMLfromLocDB.log readXML.expected > diffreadXMLLocDB.log
diff readBigXMLAndDump.log readXML.expected > diffreadXMLBDB.log

if ( -s diffreadXMLSTD.log ) then
    echo "ERROR THE MULTI-XML FILE GEOMETRY WAS NOT DUMPED PROPERLY." | tee -a GeometryValidation.log
    exit
else
    echo "GeometryFile dump from multiple XML files done."
endif

if ( -s diffreadXMLGTDB.log ) then
    echo "ERROR THE GLOBAL TAG DATABASE GEOMETRY WAS NOT DUMPED PROPERLY." | tee -a GeometryValidation.log
    exit
else
    echo "GeometryFile dump from global tag database done."
endif

if ( -s diffreadXMLLocDB.log ) then
    echo "ERROR THE LOCAL DATABASE GEOMETRY WAS NOT DUMPED PROPERLY." | tee -a GeometryValidation.log
    exit
else
    echo "GeometryFile dump from local database done."
endif

if ( -s diffreadXMLBDB.log ) then
    echo "ERROR THE BIG SINGLE XML FILE WAS NOT DUMPED PROPERLY." | tee -a GeometryValidation.log
    exit
else
    echo "GeometryFile dump from big single XML file done."
endif

#    ,dumpFile1 = cms.string("workarea/xml/dumpSTD")
#    ,dumpFile2 = cms.string("workarea/db/dumpBDB")
#dumpBDB                            dumpGTDB
#dumpLocDB                          dumpSTD
#>>> processing event # run: 1 lumi: 1 event: 1 time 1
#>>> processed 1 events

echo ">>> processing event # run: 1 lumi: 1 event: 1 time 1" >compDDdumperrors.expected
echo ">>> processed 1 events" >>compDDdumperrors.expected

cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testCompareDumpFiles.py .
sed -i "{/dumpFile1 /d}" testCompareDumpFiles.py
sed -i "{/dumpFile2 /d}" testCompareDumpFiles.py
sed -i "/TestCompareDDDumpFiles/ a\dumpFile1=cms.string\('./dumpSTD'\)\, dumpFile2=cms.string\('./dumpBDB'\)," testCompareDumpFiles.py
cmsRun testCompareDumpFiles.py > tcdfSTDvsBDB.log


if (-s tcdfSTDvsBDB.log || -s diffcompSTDvsBDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN STD XML AND BIG SINGLE XML." | tee -a GeometryValidation.log
endif

rm testCompareDumpFiles.py
cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testCompareDumpFiles.py .
sed -i "{/dumpFile1 /d}" testCompareDumpFiles.py
sed -i "{/dumpFile2 /d}" testCompareDumpFiles.py
sed -i "/TestCompareDDDumpFiles/ a\dumpFile1=cms.string\('./dumpSTD'\)\, dumpFile2=cms.string\('./dumpLocDB'\)," testCompareDumpFiles.py
cmsRun testCompareDumpFiles.py > tcdfSTDvsLocDB.log

diff compDDdumperrors.log compDDdumperrors.expected > diffcompSTDvsLocDB.log
if (-s tcdfSTDvsLocDB.log || -s diffcompSTDvsLocDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN STD XML AND LOCAL DATABASE BLOB." | tee -a GeometryValidation.log
endif

rm testCompareDumpFiles.py
cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testCompareDumpFiles.py .
sed -i "{/dumpFile1 /d}" testCompareDumpFiles.py
sed -i "{/dumpFile2 /d}" testCompareDumpFiles.py
sed -i "/TestCompareDDDumpFiles/ a\dumpFile1=cms.string\('./dumpSTD'\)\, dumpFile2=cms.string\('./dumpGTDB'\)," testCompareDumpFiles.py
cmsRun testCompareDumpFiles.py > tcdfSTDvsGTDB.log

diff compDDdumperrors.log compDDdumperrors.expected > diffcompSTDvsGTDB.log
if (-s tcdfSTDvsGTDB.log || -s diffcompSTDvsGTDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN STD XML AND GLOBALTAG DATABASE BLOB." | tee -a GeometryValidation.log
endif

rm testCompareDumpFiles.py
cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testCompareDumpFiles.py .
sed -i "{/dumpFile1 /d}" testCompareDumpFiles.py
sed -i "{/dumpFile2 /d}" testCompareDumpFiles.py
sed -i "/TestCompareDDDumpFiles/ a\dumpFile1=cms.string\('./dumpBDB'\)\, dumpFile2=cms.string\('./dumpLocDB'\)," testCompareDumpFiles.py
cmsRun testCompareDumpFiles.py > tcdfBDBvsLocDB.log

diff compDDdumperrors.log compDDdumperrors.expected > diffcompBDBvsLocDB.log
if (-s tcdfBDBvsLocDB.log || -s diffcompBDBvsLocDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN SINGLE BIG XML FILE AND LOCAL DATABASE BLOB." | tee -a GeometryValidation.log
endif

rm testCompareDumpFiles.py
cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testCompareDumpFiles.py .
sed -i "{/dumpFile1 /d}" testCompareDumpFiles.py
sed -i "{/dumpFile2 /d}" testCompareDumpFiles.py
sed -i "/TestCompareDDDumpFiles/ a\dumpFile1=cms.string\('./dumpBDB'\)\, dumpFile2=cms.string\('./dumpGTDB'\)," testCompareDumpFiles.py
cmsRun testCompareDumpFiles.py > tcdfBDBvsGTDB.log

diff compDDdumperrors.log compDDdumperrors.expected > diffcompBDBvsGTDB.log
if (-s tcdfBDBvsGTDB.log || -s diffcompBDBvsGTDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN SINGLE BIG XML FILE AND GLOBALTAG DATABASE BLOB."  | tee -a GeometryValidation.log
endif

rm testCompareDumpFiles.py
cp $CMSSW_RELEASE_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/testCompareDumpFiles.py .
sed -i "{/dumpFile1 /d}" testCompareDumpFiles.py
sed -i "{/dumpFile2 /d}" testCompareDumpFiles.py
sed -i "/TestCompareDDDumpFiles/ a\dumpFile1=cms.string\('./dumpLocDB'\)\, dumpFile2=cms.string\('./dumpGTDB'\)," testCompareDumpFiles.py
cmsRun testCompareDumpFiles.py > tcdfLocDBvsGTDB.log

diff compDDdumperrors.log compDDdumperrors.expected > diffcompLocDBvsGTDB.log
if (-s tcdfLocDBvsGTDB.log || -s diffcompLocDBvsGTDB.log ) then
    echo "WARNING THE GEOMETRYFILE IS DIFFERENT BETWEEN LOCAL AND GLOBALTAG DATABASE BLOBS."  | tee -a GeometryValidation.log
endif

echo "End Simulation geometry validation" | tee -a GeometryValidation.log
