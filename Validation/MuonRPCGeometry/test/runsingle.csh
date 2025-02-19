#! /bin/csh

if ($#argv != 4) then
  echo "Usage: $0 uniqueNumber scrtipDir script storeFname"
  exit 231
endif

set unum = $1
set scriptDir = $2
set script = $3
set cfile = $4

cd $scriptDir
eval `scramv1 runtime -csh`
cd -

cp $scriptDir/$script .

echo process.source.fileNames = cms.untracked.vstring\(\"$cfile\"\) >> $script
echo 'process.maxEvents = cms.untracked.PSet(input = cms.untracked.int32(-1))' >> $script



cmsRun $script

set fout = mu.$unum.txt
mv muons.txt $fout
gzip -9 $fout
mv *.gz $scriptDir/out

#detailedInfo.txt
#gzip -9 detailedInfo.txt
#mv detailedInfo.txt.gz $scriptDir/out/detailedInfo_$unum.txt.gz







