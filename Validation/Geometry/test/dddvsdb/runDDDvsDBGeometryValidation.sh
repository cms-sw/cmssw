#! /bin/tcsh

echo " START Geometry Validation"

if ($#argv == 0) then
    set gtag="MC_31X_V8::All"
    set geometry="GeometryIdeal"
else if($#argv == 1) then
    set gtag=`echo ${1}`
    set geometry="GeometryIdeal"
else
    set gtag=`echo ${1}`
    set geometry=`echo ${2}`
endif

cmsenv
mkdir workArea
cd workArea
set myDir=`pwd`
source $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/blob_preparation.txt > GeometryValidation.log
cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometryxmlwriter.py .
sed -i "{s/GeometryExtended/${geometry}/}" geometryxmlwriter.py >>  GeometryValidation.log
cmsRun geometryxmlwriter.py >>  GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrywriter.py .
sed -i "{s/GeometryExtended/${geometry}/}" geometrywriter.py >>  GeometryValidation.log
cmsRun geometrywriter.py >>  GeometryValidation.log
if ( -e myfile.db ) then
    echo "The local DB file is present" | tee -a GeometryValidation.log
else
    echo "ERROR the local DB file is not present" | tee -a GeometryValidation.log
    exit
endif

echo "Start compare the content of GT and the local DB" | tee -a GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrytest_local.py .
cmsRun geometrytest_local.py > outLocalDB.log
if ( -s outLocalDB.log ) then
    echo "Local DB access run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of Local DB access test is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/CondTools/Geometry/test/geometrytest_db.py .
sed -i "{/process.GlobalTag.globaltag/d}" geometrytest_db.py >> GeometryValidation.log 
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" geometrytest_db.py >> GeometryValidation.log 
cmsRun geometrytest_db.py > outGTDB.log
if ( -s outGTDB.log ) then
    echo "GT DB access run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of GT DB access test is empty" | tee -a GeometryValidation.log
    exit
endif

diff outLocalDB.log outGTDB.log > logLocalvsGTDiff.log
if ( -s logLocalvsGTDiff.log ) then
    echo "WARNING THE CONTENT OF GLOBAL TAG IS DIFFERENT WHIT RESPECT TO THE LOCAL DB FILE" | tee -a GeometryValidation.log
endif

echo "End compare the content of GT and the local DB" | tee -a GeometryValidation.log

echo "Start Tracker RECO geometry validation" | tee -a GeometryValidation.log

mkdir tkdb
mkdir tkdblocal
mkdir tkddd

cp myfile.db tkdblocal

cd tkdb
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/trackerModuleInfoDB_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" trackerModuleInfoDB_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" trackerModuleInfoDB_cfg.py >> ../GeometryValidation.log 
cmsRun trackerModuleInfoDB_cfg.py >> ../GeometryValidation.log
mv trackerModuleInfoDB_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from DB run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from DB is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../tkdblocal
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/trackerModuleInfoLocalDB_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" trackerModuleInfoLocalDB_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" trackerModuleInfoLocalDB_cfg.py >> ../GeometryValidation.log 
cmsRun trackerModuleInfoLocalDB_cfg.py >> ../GeometryValidation.log
mv trackerModuleInfoLocalDB_cfg.py ../
if ( -s ModuleInfo.log ) then
    echo "TK test from Local DB run ok" | tee -a ../GeometryValidation.log
else
    echo "ERROR the output of TK test from Local DB is empty" | tee -a ../GeometryValidation.log
    exit
endif

cd ../tkddd
cp $CMSSW_RELEASE_BASE/src/Geometry/TrackerGeometryBuilder/test/trackerModuleInfoDDD_cfg.py .
sed -i "{/process.GlobalTag.globaltag/d}" trackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" trackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log 
cmsRun trackerModuleInfoDDD_cfg.py >> ../GeometryValidation.log
mv trackerModuleInfoDDD_cfg.py ../
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

cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometryFromDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testDTGeometryFromDB_cfg.py >> GeometryValidation.log 
cmsRun testDTGeometryFromDB_cfg.py > outDB_DT.log
if ( -s outDB_DT.log ) then
    echo "DT test from DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometryFromLocalDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometryFromLocalDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testDTGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
cmsRun testDTGeometryFromLocalDB_cfg.py > outLocalDB_DT.log
if ( -s outDB_DT.log ) then
    echo "DT test from Local DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from Local DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/DTGeometry/test/testDTGeometry_cfg.py .
sed -i "{s/GeometryExtended/${geometry}/}" testDTGeometry_cfg.py >>  GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" testDTGeometry_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testDTGeometry_cfg.py >> GeometryValidation.log 
cmsRun testDTGeometry_cfg.py > outDDD_DT.log
if ( -s outDDD_DT.log ) then
    echo "DT test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of DT test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outDDD_DT.log outDB_DT.log > logDTDiffGTvsDDD.log
if ( -s logDTDiffGTvsDDD.log ) then
    echo "WARNING THE DT RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for DTGeom' outDDD_DT.log outLocalDB_DT.log > logDTDiffLocalvsDDD.log
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
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testCSCGeometryFromDB_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometryFromDB_cfg.py > outDB_CSC.log
if ( -s outDB_CSC.log ) then
    echo "CSC test from GT DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from GT DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/CSCGeometry/test/testCSCGeometryFromLocalDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testCSCGeometryFromLocalDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testCSCGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
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
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testCSCGeometry_cfg.py >> GeometryValidation.log 
cmsRun testCSCGeometry_cfg.py > outDDD_CSC.log
if ( -s outDDD_CSC.log ) then
    echo "CSC test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of CSC test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outDDD_CSC.log outDB_CSC.log > logCSCDiffGTvsDDD.log
if ( -s logCSCDiffGTvsDDD.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outDDD_CSC.log outLocalDB_CSC.log > logCSCDiffLocalvsDDD.log
if ( -s logCSCDiffLocalvsDDD.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND Local DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for CSCGeom' outDB_CSC.log outLocalDB_CSC.log > logCSCDiffLocalvsGT.log
if ( -s logCSCDiffLocalvsGT.log ) then
    echo "WARNING THE CSC RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End CSC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start RPC RECO geometry validation" | tee -a GeometryValidation.log

cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometryFromDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testRPCGeometryFromDB_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometryFromDB_cfg.py > outDB_RPC.log
if ( -s outDB_RPC.log ) then
    echo "RPC test from GT DB run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from GT DB is empty" | tee -a GeometryValidation.log
    exit
endif

cp $CMSSW_RELEASE_BASE/src/Geometry/RPCGeometry/test/testRPCGeometryFromLocalDB_cfg.py .  
sed -i "{/process.GlobalTag.globaltag/d}" testRPCGeometryFromLocalDB_cfg.py >> GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testRPCGeometryFromLocalDB_cfg.py >> GeometryValidation.log 
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
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" testRPCGeometry_cfg.py >> GeometryValidation.log 
cmsRun testRPCGeometry_cfg.py > outDDD_RPC.log
if ( -s outDDD_RPC.log ) then
    echo "RPC test from DDD run ok" | tee -a GeometryValidation.log
else
    echo "ERROR the output of RPC test from DDD is empty" | tee -a GeometryValidation.log
    exit
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outDDD_RPC.log outDB_RPC.log > logRPCDiffGTvsDDD.log
if ( -s logRPCDiffGTvsDDD.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outDDD_RPC.log outLocalDB_RPC.log > logRPCDiffLocalvsDDD.log
if ( -s logRPCDiffLocalvsDDD.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN DDD AND LOCAL DB" | tee -a GeometryValidation.log
endif

diff --ignore-matching-lines='Geometry node for RPCGeom' outDB_RPC.log outLocalDB_RPC.log > logRPCDiffLocalvsDB.log
if ( -s logRPCDiffLocalvsDB.log ) then
    echo "WARNING THE RPC RECO GEOMETRY IS DIFFERENT BETWEEN GT DB AND LOCAL DB" | tee -a GeometryValidation.log
endif

echo "End RPC RECO geometry validation" | tee -a GeometryValidation.log

echo "Start CALO RECO geometry validation" | tee -a GeometryValidation.log

addpkg Geometry/CaloEventSetup
cp myfile.db $CMSSW_BASE/src/Geometry/CaloEventSetup/test/
cd $CMSSW_BASE/src/Geometry/CaloEventSetup/
cd data
wget -i download.url
cd ../test
source setup.scr >> ${myDir}/GeometryValidation.log
sed -i "{/process.GlobalTag.globaltag/d}" runTestCaloGeometryXMLDB_cfg.py >> ${myDir}/GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" runTestCaloGeometryXMLDB_cfg.py >> ${myDir}/GeometryValidation.log 
cmsRun runTestCaloGeometryXMLDB_cfg.py > GeometryCaloValidation.log
if ( -s GeometryCaloValidation.log ) then
    echo "CALO test run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO test is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif

sed -i "{/process.GlobalTag.globaltag/d}" runTestCaloGeometryXMLLocalDB_cfg.py >> ${myDir}/GeometryValidation.log
sed -i "/FrontierConditions_GlobalTag_cff/ a\process.GlobalTag.globaltag = '${gtag}'" runTestCaloGeometryXMLLocalDB_cfg.py >> ${myDir}/GeometryValidation.log 
cmsRun runTestCaloGeometryXMLLocalDB_cfg.py > GeometryCaloValidationLocal.log
if ( -s GeometryCaloValidationLocal.log ) then
    echo "CALO Local test run ok" | tee -a ${myDir}/GeometryValidation.log
else
    echo "ERROR the output of CALO Local test is empty" | tee -a ${myDir}/GeometryValidation.log
    exit
endif
cd ${myDir}

less $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidation.log | tee -a GeometryValidation.log
less $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log | tee -a GeometryValidation.log

grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidation.log > CALOError.log 
grep 'BIG DISAGREEMENT FOUND' $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log > CALOLocalError.log 

rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidation.log
rm -f $CMSSW_BASE/src/Geometry/CaloEventSetup/test/GeometryCaloValidationLocal.log

if ( -s CALOError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN DDD AND GT DB" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      

if ( -s CALOLocalError.log ) then                                                               
    echo "WARNING THE CALO GEOMETRY IS DIFFERENT BETWEEN DDD AND LOCAL DB" | tee -a GeometryValidation.log                                                                                  
endif                                                                                                      
                                                                                              
echo "End CALO RECO geometry validation" | tee -a GeometryValidation.log

#echo "Start Simulation geometry validation" | tee -a GeometryValidation.log

#addpkg DetectorDescription/Schema
#addpkg GeometryReaders/XMLIdealGeometryESSource
#cd $CMSSW_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/
#source runXMLBigFileToDBAndBackValidation.sh ${geometry} > GeometryXMLValidation.log
#cd ${myDir}
#less $CMSSW_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/GeometryXMLValidation.log | tee -a GeometryValidation.log
#rm -f $CMSSW_BASE/src/GeometryReaders/XMLIdealGeometryESSource/test/GeometryXMLValidation.log

#echo "End Simulation geometry validation" | tee -a GeometryValidation.log
