#! /bin/csh

set RefRelease=$1

echo Copying and removing reference histograms for release $RefRelease

tar cvzf ${RefRelease}.tgz /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles/${RefRelease}
rm -r /afs/cern.ch/cms/performance/tracker/activities/validation/ReferenceFiles/${RefRelease}
rfcp ${RefRelease}.tgz /castor/cern.ch/cms/Validation/TrackerValidation
