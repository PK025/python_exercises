#!/usr/bin/python
"""
@author: PK025

A game of rock-paper-scissors against AI based on Markov chain.

Optional arguments:
-n --number_of_rounds
How many rounds of the game will be played, default value is 20.
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import random as rd

def main(args):
    max_rounds = args.number_of_rounds;
    cumulative_points = 0
    legend = ['rock' , 'paper' , 'scissors' , 'total']
    w = np.c_[np.ones((3, 3))*10 , np.ones((3))*3*10]
    rules = np.matrix('0 1 -1; -1 0 1; 1 -1 0')
    res = np.zeros(max_rounds+1)
    last = rd.choice([0, 1, 2])
    
    for i in range(max_rounds):
    
        player_choice = get_player_choice()
    
        roll = rd.random()
        if(roll<(w[last, 0]/w[last, 3])):
            pred = 0
        elif(roll<(w[last, 0]/w[last, 3])+(w[last, 1]/w[last, 3])):
            pred = 1
        else:
            pred = 2
    
        answer = pred + 1
        if(answer > 2):
            answer = 0
        
    
        outcome = rules[player_choice, answer]
        
        if(outcome == 1):
            if(i>1):
                w[last, pred] += 3
                w[last, 3] += 3
        elif(outcome == -1):
            if(w[last, pred]>1 and i>1):
                w[last, pred] -= 1
                w[last, 3] -= 1
        else:
            if(i>1):
                w[last, pred] += 1
                w[last, 3] += 1
                
        last = player_choice
        cumulative_points += outcome
        res[i+1] = cumulative_points
        
        print('You: ' + legend[player_choice], end=', ')
        print('AI: '+ legend[answer], end='. ')
        if(outcome == 1):
            print('You lost!')
        elif(outcome == -1):
            print('You won!')
        else:
            print('It\'s a draw!')
        
    print('\n\nSummary:')
    print('\nWeight matrix W')
    print(w)
    m = np.transpose(w[:, 0:3]) / w[:, 3]
    m = np.transpose(m)
    print('\nTransition matrix M')
    print(m)
    
    plt.plot( res, 'b-')
    plt.grid(True)
    plt.title(u'Game results')
    plt.ylabel(u'Total points')
    plt.xlabel(u'Rounds')
    plt.show()
    
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='A game of rock-paper-scissors against AI based on Markov chain.')
    parser.add_argument('-n', '--number_of_rounds', type=int, default=20, help='How many rounds of the game will be played, default value is 20.')
    return parser.parse_args()


def get_player_choice():
    print('Your turn, pick [r]ock, [p]aper or [s]cissors:')
    
    while(True):
        player_input = input()
        if(player_input=='r'):
            return 0
        elif(player_input=='p'):
            return 1
        elif(player_input=='s'):
            return 2
        print('Wrong input, please pick [r]ock, [p]aper or [s]cissors:')


if __name__ == '__main__':
    main(parse_arguments())
