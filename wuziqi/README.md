# wuziqi board game
   This project apply deep reinforce learning to train a network to play wuziqi board game

## game rules
   There are two players in a game. Two player take term to put one piece into board. The first player who get 5 of his 
   pieces linked in a line without the pieces of other player in between wins. The line can be horizontal or vertical or
   diagonal.
   
## run the game with two random player
   In training.py, the function run_competing_agent will run the games with two reinforce learning agents. Each agent is
   consisted of a value net and a policy net. The value net is linear regression net. It will provide the value of how 
   good is an action on a specific state. The policy net is classification net. It will provide the possibilities of 
   actions on a state should be taken. The two agents run with same algorithm. They train their policy net and value net
   on each move.
    
    To run the game in bash
```bash
   python training.py

```
   