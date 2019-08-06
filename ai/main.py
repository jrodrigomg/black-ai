#!env/bin/python

#Author: jrodrigomg

#Especial thanks to the author of this post:
#https://adventuresinmachinelearning.com/reinforcement-learning-tutorial-python-keras/


from environment import Environment
import numpy as np
import time
from keras.models import Sequential
from keras.layers import Dense, InputLayer
import keras
import matplotlib.pylab as plt
from keras.models import load_model

len_epoch = 1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def play(env):
    epoch = 0
    loaded = False
    #Neural network
    model = Sequential()
    model.add(InputLayer(batch_input_shape=(1, 30)))
    model.add(Dense(60, activation='relu'))
    model.add(Dense(30, activation='relu'))
    model.add(Dense(2, activation='linear'))
    # adam =keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss='mse', optimizer="adam", metrics=['mae'])

    #Load a previous model
    response = input("Load previous model? y/n (no if you dont have a previous model)\n")
    if(response=="y" or response=="Y" or response=="yes"):
        model.load_weights('my_model.h5')
        weights =model.layers[1].get_weights()[0]
        print("Getting this weights in the last layer:\n",weights)
    

    #Q Learning configurations
    y = 0.8
    eps = 0.5
    if(loaded):
        eps = 0.20
    decay_factor = 0.996
    y_factor = 0.996
    r_avg_list = []
    r_10_list = []
    winnigs_seguidas = 0
    n = 200
    r_10_sum = 0

    #Setting Configurations
    response = input("How quickly? (s)")
    env.duration = float(response)
    response = input("How many iterations?(n)\n")
    n = int(response)


    #Juega indefinidamente
    for i in range(1,n):
        #Contamos cuantas veces vamos jugando
        epoch+= 1
        print("\n Epoch:", epoch)
        r_sum = 0
        #Empezamos el juego en el ambiente.
        #Obteniendo el primer estado y ver si no ha terminado ya!
        s, _, finished = env.start_game()
        while not finished: #Mientras el juego sigue en pie
            if np.random.random() < eps: #Esto se lo fumaron de algoritmos geneticos para que cambie de opcion almenos unas veces
                a = np.random.randint(0, 2) #Si le toca escoger random entonces q escojer entre hit y stand
            else:
                a = np.argmax(model.predict(np.identity(30)[s:s+1])) #Si no se quiere cambiar entonces que escoja nuestro modelo.
            #Obtenemos el estado, reward y si está finalizado en base a la accion tomada
            new_s, r, finished = env.do_action(a)
            #Ahora nuestro Q sera en base a la ecuacion de Bellman
            target = r + y * np.max(model.predict(np.identity(30)[new_s:new_s+1]))
            #Ahora necesitamos solo modificar uno de los dos resultados ya que
            #se escogio solo una de esas dos acciones
            #nos explica entonces que entonces el target lo contrenda el vector en la posicion de la accion tomada
            target_vec = model.predict(np.identity(30)[s:s+1])
            target_vec[0][a] = target

            #Ahora actualizamos el modelo con los nuevos valores
            model.fit(np.identity(30)[s:s + 1], target_vec, epochs=1, verbose=0)
            #Ahora el s actual se vuelve el que fue en base a nuestra accion anterior
            s = new_s
            #Sumamos el reward
            r_sum += r
        
        #sumamos el reward
        r_10_sum += r_sum

        #Calculando el nuevo discount factor y
        y = 1 - y_factor*(1- y)

        #Calculando el epsilon nuevo 
        eps *= decay_factor

        ##Suponemos que con mayor o igual a 0.6 es que ha ganado.
        if(r_sum >= 1):
            winnigs_seguidas+=1
        #Cada 10 juegos miramos cuantas veces gano y volvemos a resetear el numero
        if(epoch%20 == 0):
            r_avg_list.append(winnigs_seguidas)
            r_10_list.append(r_10_sum)
            r_10_sum =0
            winnigs_seguidas = 0
    
    #Miramos cuanto tiene epsilon a este punto  
    print("eps:", eps)
    #Miramos cuanto tiene y
    print("y:", y)

    #Imprimamos nuestro resultado:
    print("Result:(weights)")
    weights =model.layers[2].get_weights()[0]
    print(weights)


    # #Graficamos la suma de cada 10 juegos
    plt.plot(r_10_list)
    plt.ylabel('Reward each 20 games')
    plt.xlabel('Step each 20 games')
    plt.show()


    #Guardamos los pesos del modelo!
    
    response = input("Save model? y/n\n")
    if(response=="y" or response=="Y" or response=="yes"):
        print("Saving Model")
        model.save('my_model.h5')  

    #Graficamos
    # plt.plot(r_avg_list)
    # plt.ylabel('Winnings each 5 games')
    # plt.xlabel('Number of 5 games iterations')
    # plt.show()




def play_greedy(env):
    epoch = 0
    #Neural network
    q_table = np.zeros((30, 2))
    y = 0.8
    eps = 0.5
    lr = 0.7
    lr_factor = 0.995
    y_factor = 0.995
    decay_factor = 0.995
    winnigs_seguidas = 0
    r_avg_list = []
    r_10_list = []
    done = False
    n = 50
    while not done:

        r_10_sum = 0
        #Hacer n iteraciones
        for i in range(1,n):
            #Contamos cuantas veces vamos jugando
            epoch+= 1
            print("\n Epoch:", epoch)
            r_sum = 0
            #Empezamos el juego en el ambiente.
            #Obteniendo el primer estado y ver si no ha terminado ya!
            s, _, finished = env.start_game()
            while not finished: #Mientras el juego sigue en pie
                if np.random.random() < eps: #Esto se lo fumaron de algoritmos geneticos para que cambie de opcion almenos unas veces
                    a = np.random.randint(0, 2) #Si le toca escoger random entonces q escojer entre hit y stand
                else:
                    a = np.argmax(q_table[s, :]) #Si no se quiere cambiar entonces que escoja nuestro modelo.
                new_s, r, finished = env.do_action(a)
                #ECUACION DE Bellman , el nucleo de nuestro RL
                q_table[s, a] +=  lr * (r + y * np.max(q_table[new_s, :]) - q_table[s, a])
                #ahora s es nuestro new_s ya que es nuevo estado
                s = new_s
                #Sumamos el reward
                r_sum += r
            
            r_10_sum += r_sum
            ##Suponemos que con mayor o igual a 0.6 es que ha ganado.
            if(r_sum >= 1):
                winnigs_seguidas+=1
            
            #Cada 10 juegos miramos cuantas veces gano y volvemos a resetear el numero
            if(epoch%20 == 0):
                r_avg_list.append(winnigs_seguidas)
                r_10_list.append(r_10_sum)
                winnigs_seguidas = 0
                r_10_sum = 0
            

            #Calculando el nuevo discount factor y
            y = 1 - y_factor*(1- y)

            ##Calculando el nuevo learning rate lr
            lr *= (lr_factor) 

            #Calculando el epsilon nuevo 
            eps *= decay_factor

        #Miramos cuanto tiene epsilon a este punto        
        print("eps:", eps)
        #Miramos cuanto tiene lr
        print("lr:", lr)
        #Miramos cuanto tiene y
        print("y:", y)

        #Miramos lo que contiene nuestro table
        print("Q TABLE:\n",q_table)


        
        #TODO:
        # * Save the model (Q_TABLE)

        # Graficamos el punto
        # plt.plot(r_avg_list)
        # plt.ylabel('Winnings each 20 games')
        # plt.xlabel('Number of 20 games iterations')
        # plt.show()

        # #Graficamos la suma de cada 10 juegos
        plt.plot(r_10_list)
        plt.ylabel('Reward each 20 games')
        plt.xlabel('Step each 20 games')
        plt.show()


        ##Si deseamos correr de nuevo sin perder nuestro table.
        response = input("Run again? (y/n) \n")
        if(response=="y" or response=="Y" or response=="yes"):
            response = input("How quick? (s)\n")
            env.duration = float(response)
            response = input("How many iterations?(n)\n")
            n = int(response)
        else:
            done = True
            print("Ok bye bye")
        

def main():
    # Initialize key objects: environment, agent and preprocessor
    env = Environment("127.0.0.1", 9090)
    response = input("Wich model you want to run? \n 1. e-greedy model \n 2.- keras model \n")
    
    if(response=="2"):
        print("Running keras model")
        play(env)
    else:
        print("Runing e-greedy model")
        play_greedy(env)
    

if __name__ == '__main__':
    main()









