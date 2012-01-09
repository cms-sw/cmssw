#!/bin/csh

# This script is used to store generated histograms
# in their archives, currently in the web space.

#============= Configuration =================
# This script behavior is tuned by few unix variables and command-line
# arguments. You can use Oval as execution manager, whose configuration is
# within OvalFile. Oval will set the relevant variables for you.
# If you prefer to set the variables and run this script directly,
# here they are :
#
# $1 : eventual first command-line argument, immediatly duplicated into STORE_ENV,
#   is the name of the current context, used to build some default value
#   for STORE_FILE.
# $2 : eventual second command-line argument, immediatly duplicated into STORE_OUTPUT_FILE,
#   is the default base name of the files containing the histograms ; it is
#   also used to build some default value for STORE_FILE.
#
# STORE_RELEASE : chosen name for the new release to validate ; used in web pages
#   and used to build the path where the web pages will be stored.
#
# STORE_FILE : complete path of the ROOT file containing the histograms.
#   If not set, a default value is computed, based on 1st command line argument,
#   2nd command line argument and current directory.
# 
# STORE_LOGS : complete path of the associated log files.
# 
#=========================================================================================


#============== Core config ==================

setenv STORE_ENV $1
setenv STORE_OUTPUT_FILE $2
setenv STORE_ORIGINAL_DIR $cwd

#============== Prepare output directory ==================

if ( "${STORE_WEB}" == "" ) then
  echo "STORE_WEB must be defined"
  exit 1
else
  echo "STORE_WEB = ${STORE_WEB}"
endif

if ( "${STORE_RELEASE}" == "" ) then
  echo "STORE_RELEASE must be defined"
  exit 2
else
  echo "STORE_RELEASE = ${STORE_RELEASE}"
endif

setenv OUTPUT_DIR "$STORE_WEB/$STORE_RELEASE/Electrons/data"

if (! -d $OUTPUT_DIR) then
  mkdir -p $OUTPUT_DIR
endif

#============== Find data files ==================

if ( ${?STORE_FILE} == "0" ) setenv STORE_FILE ""
if ( ${?STORE_LOGS} == "0" ) setenv STORE_LOGS ""

if ( ${STORE_FILE} == "" ) then
  if ( -r "${STORE_ORIGINAL_DIR}/cmsRun.${STORE_ENV}.olog.${STORE_OUTPUT_FILE}" ) then
    setenv STORE_FILE "${STORE_ORIGINAL_DIR}/cmsRun.${STORE_ENV}.olog.${STORE_OUTPUT_FILE}"
    setenv STORE_LOGS "${STORE_ORIGINAL_DIR}/*.${STORE_ENV}.olog"
  endif
endif

if ( ${STORE_FILE} == "" ) then
  echo "Do not know which file to copy !"
  exit 3
endif

if ( -r "${STORE_FILE}" ) then
  echo "STORE_FILE = ${STORE_FILE}"
else
  echo "${STORE_FILE} is unreadable !"
  exit 4
endif
  
#============== Copy ==================

echo cp $STORE_FILE $OUTPUT_DIR
cp -f $STORE_FILE $OUTPUT_DIR
  
if ( "${STORE_LOGS}" != "" ) then
  echo cp ${STORE_LOGS} $OUTPUT_DIR
  cp -f ${STORE_LOGS} $OUTPUT_DIR
  gzip -f $OUTPUT_DIR/*.olog
endif

