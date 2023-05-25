import random

'''
returns a seed of a specified length
seed_length: an int that represents how many digits of a seed you want
returns: an int
credits: RMPR, stack overflow. https://stackoverflow.com/questions/58468532/generate-random-seed-in-python
'''
def get_numerical_seed(seed_length:int = 9):
    random.seed() #initiate the random number generator
    min = 10**(seed_length - 1)
    max = 9*min + (min - 1)
    seed = random.randint(min, max)
    return seed