################################################################################
# RelMon: a tool for automatic Release Comparison                              
# https://twiki.cern.ch/twiki/bin/view/CMSPublic/RelMon
#
# This is a code manipulation of a component of the DQMCompare tool of 
# Marco Rovere and Luca Malgeri.
#
# $Author: dpiparo $
# $Date: 2012/06/12 12:25:27 $
# $Revision: 1.1 $
#
#                                                                              
# Danilo Piparo CERN - danilo.piparo@cern.ch                                   
#                                                                              
################################################################################

from os import getenv
from os.path import exists
from httplib import HTTPSConnection
from urllib2  import AbstractHTTPHandler
#-------------------------------------------------------------------------------  

class X509CertAuth(HTTPSConnection):
  '''Class to authenticate via Grid Certificate'''
  def __init__(self, host, *args, **kwargs):
    key_file = None
    cert_file = None

    x509_path = getenv("X509_USER_PROXY", None)
    if x509_path and exists(x509_path):
      key_file = cert_file = x509_path

    if not key_file:
      x509_path = getenv("X509_USER_KEY", None)
      if x509_path and exists(x509_path):
        key_file = x509_path

    if not cert_file:
      x509_path = getenv("X509_USER_CERT", None)
      if x509_path and exists(x509_path):
        cert_file = x509_path

    if not key_file:
      x509_path = getenv("HOME") + "/.globus/userkey.pem"
      if exists(x509_path):
        key_file = x509_path

    if not cert_file:
      x509_path = getenv("HOME") + "/.globus/usercert.pem"
      if exists(x509_path):
        cert_file = x509_path

    if not key_file or not exists(key_file):
      print >>stderr, "No certificate private key file found"
      exit(1)

    if not cert_file or not exists(cert_file):
      print >>stderr, "No certificate public key file found"
      exit(1)

    #print "Using SSL private key", key_file
    #print "Using SSL public key", cert_file
    
    HTTPSConnection.__init__(self, 
                              host,
                              key_file = key_file,
                              cert_file = cert_file,
                              **kwargs)

#-----------------------------------------------------------------------------

class X509CertOpen(AbstractHTTPHandler):
  def default_open(self, req):
    return self.do_open(X509CertAuth, req)