import logging
from datetime import datetime


levelDict = {
    'info'  : logging.INFO,
    'debug' : logging.DEBUG,
    'warn'  : logging.WARN,
    'error' : logging.ERROR,
    'fatal' : logging.FATAL
}

'''
creates a logger object that records debug messages to a log file of level INFO or higher
level: a logging._Level object, of values (INFO, DEBUG, WARN, ERROR, FATAL)
dir: a string that points to a directory that stores log files
returns: a logger object
'''
def createLogger(level:str, dir:str):
    logger = logging.getLogger(logging.__name__)
    logger.setLevel(levelDict[level])

    now = str.partition(str(datetime.now()), '.')[0].replace(' ', '-').replace(':', '-') #formats the datetime to adhere to file naming conventions
    
    logFileHandler = logging.FileHandler(f'{dir}/log{now}.log') #creates the log file and writes/appends to it
    logFileHandler.setLevel(levelDict[level])
    logFileHandlerW = logging.FileHandler(f'{dir}/log{now}.log') #creates the log file and writes/appends to it
    logFileHandlerW.setLevel(levelDict['warn'])
    formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s") #creates the format of how the logger will record debug messages
    logFileHandler.setFormatter(formatter)
    logFileHandlerW.setFormatter(formatter)
    logger.addHandler(logFileHandler)
    logger.addHandler(logFileHandlerW)

    return logger