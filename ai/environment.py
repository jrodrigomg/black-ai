import numpy as np
from websocket_server import WebsocketServer
import multiprocessing
import base64, json, re, time, threading


class Action:
    STAND   =  0
    HIT     =  1

class Environment:
    """Clase de ambiente"""

    actions = {Action.HIT:'HIT', Action.STAND:'STAND'}
    duration = 0.01

    def __init__(self, host, port):
        print ("host:" + host + " port:" + str(port))
        self.queue = multiprocessing.Queue()
        self.server = WebsocketServer(port, host=host)
        self.server.set_fn_new_client(self.new_client)
        self.server.set_fn_message_received(self.new_message)
        self.game_client = None
        thread = threading.Thread(target=self.server.run_forever)
        thread.daemon = True
        thread.start()
    
    def new_client(self,client,server):
        self.game_client = client
        self.server.send_message(self.game_client, "Connected! :)")

    def new_message(self, client, server, message):
        #Por lo general se envia la accion que se quiera tomar
        data = json.loads(message)
        state, score, finished = data['state'], data['reward'], data['finished']
        # print(state, score, finished)
        self.queue.put((state, score, finished))
    
    def start_game(self):
        # game can not be started as long as the browser is not ready
        while self.game_client is None:
            time.sleep(self.duration)
        #Empezar el juego!
        self.server.send_message(self.game_client, "START")
        time.sleep(self.duration)
        return self.get_state()

    def refresh_game(self):
        #Empezar el juego!
        print("Refreshing")
        self.server.send_message(self.game_client, "START")
        time.sleep(self.duration)


    def do_action(self, action):
        print(self.actions[action])
        self.server.send_message(self.game_client, self.actions[action])
        time.sleep(self.duration)
        return self.get_state()

    def get_state(self):
        self.server.send_message(self.game_client, "STATE")
        time.sleep(self.duration)
        state, score, finished = self.queue.get()
        print("Actual hand(state):",state,"Score:",score,"Finished:",finished)
        return state, score, finished





