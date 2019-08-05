# black-ai

Playing Black Jack with Reinforcement Learning

#### Requisites:
* Python3
* Virtualenv

#### Instalation
Cloning:
```bash
$ git clone 
$ cd ai
```
Creating a virtual environment:
```bash
$ virtualenv -p python3.5 env
```

Load environment, this is an example for fish shell:
```bash
$ source env/bin/activate.fish
```
Install requirments:
```bash
$ pip install
```
#### Run the application server
Run the main file:
```bash
$ python main.py
```

#### Running the game

In the main directory go to the game carpet:
```bash
$ cd game
```

Run a simple http server with python(Not necesary the enviroment here)
```bash
$ python -m SimpleHTTPServer 8080
```

#### Try
* Once running both application we need to configure wherever the server ask
* go http://localhost:8080 in the browser
* If just appear "Epoch 1" then refresh the browser

#### Credits
* Thanks to adventuresinML for the incredible post of RL  (https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/)
* Thanks to Kusnierewicz for the game of blackjack in javascript (https://github.com/Kusnierewicz/Blackjack-game-in-JS)
* For the structure of the enviroment i used some part of Vincent Dutordoir's repo (https://github.com/vdutor/TF-rex)

