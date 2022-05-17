# this script runs the commands to setup the watchdog to run on boot
# make sure to execute this script in the current directory
import subprocess
import os
from crontab import CronTab

cwd = os.getcwd()

print("making launcher file executable...")
make_executable = subprocess.run(["chmod", "+x", "watchdog.sh"])
print("The exit code was: %d" % make_executable.returncode)

print("making logs directory...")
make_logs = subprocess.run(["mkdir", "logs"])
print("The exit code was: %d" % make_logs.returncode)

commands = ["@reboot sh "+cwd+"/watchdog.sh >"+cwd+"/logs/cronlog 2>&1"]
print("adding to crontab...")
cron = CronTab(user="root")  # root users cron
for cmd in commands:
    job = cron.new(command=cmd)
    cron.write()
