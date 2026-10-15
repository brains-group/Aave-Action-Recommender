"""Run one command with an atomic status/heartbeat and durable log."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--status-dir',type=Path,required=True)
    p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args();command=a.command
    if command and command[0]=='--':command=command[1:]
    if not command:p.error('command required')
    out=a.status_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    guard=(out/'driver.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    record=dict(command=command,started_epoch=time.time(),pid=os.getpid(),state='running')
    def save():
        record['heartbeat_epoch']=time.time()
        tmp=out/'status.tmp';tmp.write_text(json.dumps(record,indent=2));tmp.replace(out/'status.json')
    with (out/'driver.log').open('a') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        record['child_pid']=child.pid
        try:
            while child.poll() is None:
                save();time.sleep(5)
            record.update(state='done' if child.returncode==0 else 'failed',exit_code=child.returncode)
        except BaseException:
            child.terminate();child.wait();record.update(state='interrupted',exit_code=child.returncode)
            raise
        finally:save()
    raise SystemExit(child.returncode)

if __name__=='__main__':main()
