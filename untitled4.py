import numpy as np
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random

class DQLAgent:
    def __init__(self,env):
        #build_model
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        
        #replay
        self.gamma = 0.95 #feature reward
        self.learning_rate = 0.001 #alfa
        
        #adaptiveEpsilonGreedy
        self.epsilon = 1 # en yüksek değer gitgide azaltacağız
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        #remember
        self.memory = deque(maxlen = 1000) #1000 lik bir liste
        
        #build_model
        self.model = self.build_model()
    
    def build_model(self):
        #neural network for deep q learning
        model = Sequential()
        model.add(Dense(48,input_dim = self.state_size,activation = "tanh"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse",optimizer = Adam(learning_rate = self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done):
        #storage
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state):
        #acting explore or exploit
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample()
        else:
            # büyük olanın indexini alma
            act_values = self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self,batch_size):
        #training 
        # memoryde en az 16tane state,action,reward,next_state,done varsa replay uygulanabilir
        if len(self.memory)<batch_size:
            return
        minibatch = random.sample(self.memory,batch_size) 
        #selfmemory içinden batch size kadar sample al
        
        for state,action,reward,next_state,done in minibatch:
            if done:
                target = reward
            else:
                target = reward+self.gamma*np.amax(self.model.predict(next_state)[0])
            train_target = self.model.predict(state)
            train_target[0][action] = target
            self.model.fit(state,train_target,verbose = 0)
            
    def adaptiveEGreedy(self):
        if self.epsilon>self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    
    
def preprocess_state(state, state_size):
    # Handle tuple and flatten to a single list
    if isinstance(state, tuple):
        state = [item for sublist in state for item in sublist]
    
    # Convert to NumPy array and reshape
    state = np.array(state, dtype=np.float32).reshape([1, state_size])
    return state
    
if __name__ == "__main__":
    # initialize env and agent
    env = gym.make("CartPole-v1")
    agent = DQLAgent(env)
    batch_size = 16 
    episodes = 1
    for e in range(episodes):
        # initialize environment
        state = env.reset()
        state = preprocess_state(state, agent.state_size)
        #state = (state[0].reshape(1, -1), state[1])
        
        time = 0
        while True:
            
            #act
            action = agent.act(state) # select an action
            
            #step
            next_state,reward,done,_,_ = env.step(action)
            #next_state = (next_state[0].reshape(1, -1), next_state[1])
            next_state = preprocess_state(next_state, agent.state_size)
            
            #remember/storage
            agent.remember(state,action,reward,next_state,done)
            
            #update state
            state = next_state
            
            #replay
            agent.replay(batch_size)
            
            #adjust epsilon
            agent.adaptiveEGreedy()
            
            time +=1
            if done: #15dereceden fazla eğilme veya çubuk ekrandan çıkma
                print("Episode: {} , time: {}".format(e,time))
                break
#%%
import gym
import time

# Ortamı render moduyla başlat
env = gym.make("CartPole-v1", render_mode="human")

# Diğer kodlarınız
trained_model = agent
state = env.reset()
state = preprocess_state(state, agent.state_size)
time_t = 0

while True:
    env.render()
    action = trained_model.act(state)
    next_state, reward, done, _, _ = env.step(action)
    next_state = preprocess_state(next_state, agent.state_size)
    state = next_state
    time_t += 1
    print(time_t)
    time.sleep(0.4)
    if done:
        break
print("Done")


