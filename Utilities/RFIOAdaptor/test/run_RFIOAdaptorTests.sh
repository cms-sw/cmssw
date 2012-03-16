#!/bin/bash

function die { echo Failure $1: status $2 ; exit $2 ; }

pushd ${LOCAL_TMP_DIR}

  domain=`hostname -d`
  echo domain is $domain

  echo $domain | grep cern

  if [ $? = "1" ]
  then
    echo This test only works at cern. This machine is not at cern.
    echo Skipping test.
    exit 0
  fi
  filepostfix=`date +.%Y%M%d%H%m-${SCRAM_ARCH}-$$`
  ${LOCAL_TOP_DIR}/test/${SCRAM_ARCH}/test_RFIOAdaptor_put ${filepostfix} || die "test_RFIOAdaptor_put" $?
  castorfile=/castor/cern.ch/`whoami | sed 's|^\(.\)|user/\1/\1|'`/rfiotestput${filepostfix}
  stager_rm -M ${castorfile}                         
  rfrm ${castorfile}

popd
exit 0
