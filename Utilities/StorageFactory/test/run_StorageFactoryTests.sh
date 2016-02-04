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

  echo " "
  echo Running test_StorageFactory_Rfio -----------------------------------------------------------
  ${LOCAL_TOP_DIR}/test/${SCRAM_ARCH}/test_StorageFactory_Rfio || die "test_StorageFactory_Rfio" $?

  echo " "
  echo Running test_StorageFactory_Write ----------------------------------------------------------
  ${LOCAL_TOP_DIR}/test/${SCRAM_ARCH}/test_StorageFactory_Write || die "test_StorageFactory_Write" $?

popd
exit 0
