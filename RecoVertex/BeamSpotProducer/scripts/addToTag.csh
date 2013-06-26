#! /bin/csh 

######################
set fileName=$1
set inFile=/afs/cern.ch/cms/CAF/CMSALCA/ALCA_PROMPT/"$fileName"
#####################

echo "fileName is " $fileName
echo "inFile is " $inFile

ls $inFile

if(-e $inFile) then
#
   cmscond_export_iov -d sqlite_file:MergedOffline_BeamSpotObject_ByLumi_v21_2.db \
   -s sqlite_file:$inFile \
   -i BeamSpotObject_ByLumi \
   -t BeamSpotObjects_2009_LumiBased_SigmaZ_v21_offline \
   -l sqlite_file:log.db
#
else
#
   cmscond_export_iov -d sqlite_file:MergedOffline_BeamSpotObject_ByLumi_v21_2.db \
   -s sqlite_file:$fileName \
   -i BeamSpotObject_ByLumi \
   -t BeamSpotObjects_2009_LumiBased_SigmaZ_v21_offline \
   -l sqlite_file:log.db
#
endif
#
#
#
