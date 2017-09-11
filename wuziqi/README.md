# wuziqi board game
  It is a deep reinforce learning project training agent to play Gomoku an abstract strategy board game. 
  Here is the link of Gomoku https://en.wikipedia.org/wiki/Gomoku

   
## run the game with two reinforce learning agents
   In go.py, the function ai_vs_human will run the games with a reinforce learning agents and human player. The agent 
   is consisted of a value net and a policy net. The value net is linear regression net. It will provide the value of how 
   good is an action on a specific state. The policy net is classification net. It will provide the possibilities of 
   actions on a state should be taken. The agent can train its own policy net and value net online.
    
    To run the game in bash
```bash
   python go.py

```
   