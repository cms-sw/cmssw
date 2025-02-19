#! /bin/csh

eval `scramv1 runtime -csh`

pushd ../
scramv1 b
popd

echo "===========> Validating Offline Primary Vertex Reco with 10-muon samples......."
echo "===========> Validating reconstruction....."

rm -f simpleVertexAnalyzer.root
cmsRun produceAndAnalyzePrimaryVertex.cfg
setenv REFFILE "../data/simpleVertexAnalyzer_10muons.root"
setenv CURFILE "simpleVertexAnalyzer.root"
root -b -p -q DoCompare.C\(\"_10muons\"\)

echo "===========> Validating Pool I/O....."

rm -f simplePersistentVertexAnalyzer.root
cmsRun analyzePersistentPrimaryVertex.cfg
setenv REFFILE "simpleVertexAnalyzer.root"
setenv CURFILE "simplePersistentVertexAnalyzer.root"
root -b -p -q DoCompare.C\(\"persistent_10muons\"\)
