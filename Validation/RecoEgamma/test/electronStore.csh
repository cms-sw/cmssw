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
#   $1 : when present, its value is duplicated into STORE_FILE.
#   $... : when present, other arguments are duplicated into STORE_LOGS.
#
# Mandatory variables :
#
#   STORE_WEB : base directory where all the files are stored.
#   STORE_RELEASE : chosen name for the new release to validate ; used
#     in web pages and used to build the path where the web pages will be stored.
#
# Optional variables, eventually overloaded by command-line arguments :
#
#   STORE_FILE : path of the ROOT file containing the histograms.
#   STORE_LOGS : path of the associated log files (expected to end with .olog),
#     which will also be compressed.
# 
#=========================================================================================


#============== Comand-line arguments ==================

#if ( "$1" == "-f" ) then
#  setenv STORE_FORCE "yes"
#  shift
#else
  setenv STORE_FORCE "no"
#endif

if ( "$1" != "" ) then
  setenv STORE_FILE "$1"
  shift
endif

if ( "$*" != "" ) then
  setenv STORE_LOGS "$*"
endif

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
  echo "Do not know which file to copy !"
  exit 3
endif

if ( -r "${STORE_FILE}" ) then
  echo "STORE_FILE = ${STORE_FILE}"
else
  echo "${STORE_FILE} is unreadable !"
  exit 4
endif
  
if ( "${STORE_LOGS}" != "" ) then
  echo "STORE_LOGS = ${STORE_LOGS}"
endif
  
#============== Check not already done ==================

if ( ${STORE_FORCE} == "no" && -f "${OUTPUT_DIR}/${STORE_FILE}" ) then
  echo "ERROR: ${STORE_FILE} ALREADY STORED IN ${OUTPUT_DIR} !"
  exit 5
endif

  
#============== Copy ==================

echo cp $STORE_FILE $OUTPUT_DIR
cp -f $STORE_FILE $OUTPUT_DIR
  
if ( "${STORE_LOGS}" != "" ) then
  echo cp ${STORE_LOGS} $OUTPUT_DIR
  cp -f ${STORE_LOGS} $OUTPUT_DIR
  echo "cd $OUTPUT_DIR && gzip -f *.olog"
  cd $OUTPUT_DIR && gzip -f *.olog
endif

