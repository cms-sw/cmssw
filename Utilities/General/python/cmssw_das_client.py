import time
import os
from json import loads, dumps
from types import GeneratorType
import subprocess

#Copied from das_client.py

def convert_time(val):
    "Convert given timestamp into human readable format"
    if  isinstance(val, int) or isinstance(val, float):
        return time.strftime('%d/%b/%Y_%H:%M:%S_GMT', time.gmtime(val))
    return val

def size_format(uinput, ibase=0):
    """
    Format file size utility, it converts file size into KB, MB, GB, TB, PB units
    """
    if  not ibase:
        return uinput
    try:
        num = float(uinput)
    except Exception as _exc:
        return uinput
    if  ibase == 2.: # power of 2
        base  = 1024.
        xlist = ['', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']
    else: # default base is 10
        base  = 1000.
        xlist = ['', 'KB', 'MB', 'GB', 'TB', 'PB']
    for xxx in xlist:
        if  num < base:
            return "%3.1f%s" % (num, xxx)
        num /= base

def extract_value(row, key, base=10):
    """Generator which extracts row[key] value"""
    if  isinstance(row, dict) and key in row:
        if  key == 'creation_time':
            row = convert_time(row[key])
        elif  key == 'size':
            row = size_format(row[key], base)
        else:
            row = row[key]
        yield row
    if  isinstance(row, list) or isinstance(row, GeneratorType):
        for item in row:
            for vvv in extract_value(item, key, base):
                yield vvv

def get_value(data, filters, base=10):
  """Filter data from a row for given list of filters"""
  for ftr in filters:
    if  ftr.find('>') != -1 or ftr.find('<') != -1 or ftr.find('=') != -1:
      continue
    row = dict(data)
    values = []
    keys = ftr.split('.')
    for key in keys:
      val = [v for v in extract_value(row, key, base)]
      if  key == keys[-1]: # we collect all values at last key
        values += [dumps(i) for i in val]
      else:
        row = val
      if  len(values) == 1:
        yield values[0]
      else:
        yield values

def get_data(query, limit=None, threshold=None, idx=None, host=None, cmd=None):
  cmd_opts = "--format=json"
  if threshold is not None: cmd_opts += " --threshold=%s" % threshold
  if limit     is not None: cmd_opts += " --limit=%s"     % limit
  if idx       is not None: cmd_opts += " --idx=%s"       % idx
  if host      is not None: cmd_opts += " --host=%s"      % host
  if not cmd:
    cmd = "das_client"
    for path in os.getenv('PATH').split(':'):
      if  os.path.isfile(os.path.join(path, 'dasgoclient')):
        cmd = "dasgoclient"
        break

  p = subprocess.Popen("%s %s --query '%s'" % (cmd, cmd_opts, query),shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  stdout, stderr = p.communicate()
  if not p.returncode: return loads(stdout)
  return {'status' : 'error', 'reason' : stdout}
