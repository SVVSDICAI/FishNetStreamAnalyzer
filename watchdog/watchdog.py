import subprocess
from time import sleep

host = 'www.youtube.com'
connection = 'wlan'

def ping(host):
    ping = subprocess.Popen(
        ['ping', '-c', '4', host],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    out, error = ping.communicate()
    if error != b'':
        # the ping failed
        return 'failed'
    return 'success'

def get_network_interface(eth_or_wlan): # get the name of the network interface being used
    interfaces = subprocess.Popen(
        ['ip', 'addr'],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    out, error = interfaces.communicate()
    out = out.decode().replace('\n', ' ')
    if error != b'':
        # the command failed
        return 'could not determine the interface in use'
    delims = [str(i) + ':' for i in range(10)]
    #print(delims)
    found = False
    for word in out.split(' '):
        if found:
            #print(word)
            if eth_or_wlan == 'eth':
                if word[0] == 'e':
                    return word
            elif eth_or_wlan == 'wlan':
                if word[0] == 'w':
                    return word
            found = False
        for delim in delims:
            if delim == word: # this line is the start of the information listed for a specific interface
                found = True
                #print(word)
                break
interface = get_network_interface('eth')

def disable_interface(interface):
    disable = subprocess.Popen(
        ['sudo', 'ip', 'link', 'set', interface, 'down'],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    out, error = disable.communicate()
    if error != b'':
        # the command failed
        return error.decode()
    return 'success'

def enable_interface(interface):
    enable = subprocess.Popen(
        ['sudo', 'ip', 'link', 'set', interface, 'up'],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    )
    out, error = enable.communicate()
    if error != b'':
        # the command failed
        return 'failed'
    return 'success'

# main function
def main():
    success = False
    while not success:
        ping_res = ping(host)
        if ping_res == 'success':
            print('the site was pinged successfully, exiting')
            break
        # if the ping was not successful, restart the network interface
        interface = get_network_interface(connection)
        print('ping failed, restarting interface: ' + interface)
        sleep(10)
        print('disabling: ' + disable_interface(interface))
        sleep(10)
        print('enabling: ' + enable_interface(interface))
        sleep(10)
main()
