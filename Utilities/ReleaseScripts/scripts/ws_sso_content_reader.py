#!/usr/bin/env python3
###Description: The tool reads cern web services behind SSO using user certificates
from __future__ import print_function
import os, urllib, urllib2, httplib, cookielib, sys, HTMLParser, re
from optparse import OptionParser

def getFile(path):
  npath = os.path.expanduser(path)
  while os.path.islink(npath):
    path = os.readlink(npath)
    if path[0] != "/": path = os.path.join(os.path.dirname(npath),path)
    npath = path
  return npath

class HTTPSClientAuthHandler(urllib2.HTTPSHandler):  
  def __init__(self, key, cert):  
    urllib2.HTTPSHandler.__init__(self)  
    self.key = getFile(key)  
    self.cert = getFile(cert) 

  def https_open(self, req):  
    return self.do_open(self.getConnection, req)  

  def getConnection(self, host, timeout=300):  
    return httplib.HTTPSConnection(host, key_file=self.key, cert_file=self.cert)

def _getResponse(opener, url, post_data=None, debug=False):
  response = opener.open(url, post_data)
  if debug:
    sys.stderr.write("Code: %s\n" % response.code)
    sys.stderr.write("Headers: %s\n" % response.headers)
    sys.stderr.write("Msg: %s\n" % response.msg)
    sys.stderr.write("Url: %s\n" % response.url)
  return response

def getResponseContent(opener, url, post_data=None, debug=False):
  return _getResponse(opener, url, post_data, debug).read()

def getResponseURL(opener, url, post_data=None, debug=False):
  return urllib2.unquote(_getResponse(opener, url, post_data, debug).url)

def getParentURL(url):
  items = url.split("/")
  return '%s//%s/%s/' % (items[0],items[2],items[3])

def getSSOCookie(opener, target_url, cookie, debug=False):
  opener.addheaders = [('User-agent', 'curl-sso-certificate/0.0.2')] #in sync with cern-get-sso-cookie tool
  url = getResponseURL(opener, getParentURL(target_url), debug=debug)
  content = getResponseContent(opener, url, debug=debug)
  ret = re.search('<form .+? action="(.+?)">', content)
  if ret == None:
    raise Exception("error: The page doesn't have the form with adfs url, check 'User-agent' header")
  url = urllib2.unquote(ret.group(1))
  h = HTMLParser.HTMLParser()
  post_data_local = ''
  for match in re.finditer('input type="hidden" name="([^"]*)" value="([^"]*)"', content):
    post_data_local += "&%s=%s" % (match.group(1), urllib.quote(h.unescape(match.group(2))))
    is_link_found = True
  
  if not is_link_found:
    raise Exception("error: The page doesn't have the form with security attributes, check 'User-agent' header")
  post_data_local = post_data_local[1:] #remove first &
  getResponseContent(opener, url, post_data_local, debug)

def getContent(target_url, cert_path, key_path, post_data=None, debug=False, adfslogin=None):
  opener = urllib2.build_opener(urllib2.HTTPSHandler())
  if adfslogin:
    opener.addheaders = [('Adfs-Login', adfslogin)] #local version of tc test
  
  #try to access the url first
  try:
    content = getResponseContent(opener, target_url, post_data, debug)
    if not 'Sign in with your CERN account' in content:
      return content
  except Exception:
    if debug:
      sys.stderr.write("The request has an error, will try to create a new cookie\n")

  cookie = cookielib.CookieJar()
  opener = urllib2.build_opener(urllib2.HTTPCookieProcessor(cookie), HTTPSClientAuthHandler(key_path, cert_path))  #will use private key and ceritifcate
  if debug:
    sys.stderr.write("The return page is sso login page, will request cookie.")
  hasCookie = False
  # if the access gave an exception, try to get a cookie
  try:
    getSSOCookie(opener, target_url, cookie, debug)
    hasCookie = True 
    result = getResponseContent(opener, target_url, post_data, debug)
  except Exception as e:
    result = ""
    print(sys.stderr.write("ERROR:"+str(e)))
  if hasCookie:
    burl = getParentURL(target_url)
    try:
      _getResponse(opener, burl+"signOut").read()
      _getResponse(opener, "https://login.cern.ch/adfs/ls/?wa=wsignout1.0").read()
    except:
      sys.stderr.write("Error, could not logout correctly from server") 
  return result

def checkRequiredArguments(opts, parser):
  missing_options = []
  for option in parser.option_list:
    if re.match(r'^\[REQUIRED\]', option.help) and eval('opts. %s' % option.dest) == None:
      missing_options.extend(option._long_opts)
    if len(missing_options) > 0:
      parser.error('Missing REQUIRED parameters: %s' % str(missing_options))    

if __name__ == "__main__":
  parser = OptionParser(usage="%prog [-d(ebug)] -o(ut) COOKIE_FILENAME -c(cert) CERN-PEM -k(ey) CERT-KEY -u(rl) URL") 
  parser.add_option("-d", "--debug", dest="debug", help="Enable pycurl debugging. Prints to data and headers to stderr.", action="store_true", default=False)
  parser.add_option("-p", "--postdata", dest="postdata", help="Data to be sent as post request", action="store", default=None)
  parser.add_option("-c", "--cert", dest="cert_path", help="[REQUIRED] Absolute path to cert file.", action="store")
  parser.add_option("-k", "--key", dest="key_path", help="[REQUIRED] Absolute path to key file.", action="store")
  parser.add_option("-u", "--url", dest="url", help="[REQUIRED] Url to a service behind the SSO", action="store")
  (opts, args) = parser.parse_args()
  checkRequiredArguments(opts, parser)
  content = getContent(opts.url, opts.cert_path, opts.key_path, opts.postdata, opts.debug)
  print(content)
