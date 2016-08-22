from commands import getstatusoutput
from json import loads
from das_client import get_value as das_get_value

def get_value(data, filters, base=10):
  return das_get_value(data, filters, base)

def get_data(query, limit=None, threshold=None, idx=None, host=None):
  cmd_opts = "--format=json"
  if threshold is not None: cmd_opts += " --threshold=%s" % threshold
  if limit     is not None: cmd_opts += " --limit=%s"     % limit
  if idx       is not None: cmd_opts += " --idx=%s"       % idx
  if host      is not None: cmd_opts += " --host=%s"      % host
  err, out = getstatusoutput("das_client %s --query '%s'" % (cmd_opts, query))
  if not err: return loads(out)
  return {'status' : 'error', 'reason' : out}
