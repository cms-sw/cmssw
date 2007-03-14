#!/bin/bash 

cmsRun $PKGDIR/test/${1}.cfg 2>&1 | gzip > $OUTDIR/${1}.log.gz
RETVAL=$?

\mv -f *.root $OUTDIR/

exit $RETVAL
