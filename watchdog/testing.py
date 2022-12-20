# test script to see how the subprocess lib behaves

import subprocess
from time import sleep


# this is how subprocess can be used to run a command asynchronously
def run_asynchronously(command, run_next):
    print('running asynchronously...')
    process = subprocess.Popen(command,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    ) # start the command with Popen, capturing the standard output and any possible errors
    
    print('command "' + ' '.join(command) + '" started...\n')

    for stdout_line in iter(process.stdout.readline, b''):
        print(stdout_line) # capture the output of the command and print them out
    process.stdout.close()
    return_code = process.wait() # wait for the command to finish up
    # note that the above structure could be useful
    # it can be used to wait for a command that runs continuously to fail, and restart it when it does
    # this has applications for the watchdog scripts because it can be used to make sure the stream is restarted if the connection drops

    print('\ncommand "' + ' '.join(command) + '" completed')
    out, error = process.communicate()
    if error:
        print('the command returned the following error:\n\n' + str(error) + '\n\n')

    print('now that the above command has terminated, the script can now continue...')
    run_next() # do something now that the command finished running


# this is how subprocess can be used to run a command synchronously
def run_synchronously(command, while_we_wait):
    print('running synchronously...')
    process = subprocess.Popen(command, 
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE
    ) # start the command with Popen, capturing the standard output and any possible errors

    print('command "' + ' '.join(command) + '" started...')
    print('doing other things while we wait:\n')

    while_we_wait() # do something while the command is running

    # check the status of the passed command
    status = process.poll()
    if status == 0: # command completed successfully
        print('\ncommand "' + ' '.join(command) + '" completed')
        out, error = process.communicate()
        print('the command returned the following:\n\n' + str(out) + '\n\n' + str(error))
    else: # still running, so terminate it
        process.terminate()
        print('\ncommand "' + ' '.join(command) + '" was still running and has been terminated')
        out, error = process.communicate()
        print('the command returned the following:\n\n' + str(out) + '\n\n' + str(error))



# example usage
command = ['ping', 'github.com', '-c', '4']
def func(): # example function (in the spirit of AI this script was generated with help from ChatGPT)
    n = 10 # number of correct digits to calculate pi to
    # initialize the result to 0
    result = 0
    # set the precision to the specified number of digits
    precision = 10**(-n)
    
    # iterate over the terms of the series and add them to the result
    k = 0
    while True:
        term = (1/16**k) * ((4/(8*k + 1)) - (2/(8*k + 4)) - (1/(8*k + 5)) - (1/(8*k + 6)))
        result += term
        print(result)
        # if the term is smaller than the precision, we can stop the iteration
        if abs(term) < precision:
            break
        k += 1
        sleep(0.2)
    
    print('pi to ' + str(n) + ' correct digits is ' + str(result))


run_asynchronously(command, func) # will run the command and function one at a time, waiting for the command to finish before running the function
print('\n')
run_synchronously(command, func) # will run the command and function at the same time, terminating the command if it takes longer to execute than the function


