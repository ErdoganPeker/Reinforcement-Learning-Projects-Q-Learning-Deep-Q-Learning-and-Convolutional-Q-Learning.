import pygame
import random
import time
#sprite = ekranda hareket eden objeler
import numpy as np
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random




#window size
WIDTH = 360
HEIGHT = 360
FPS = 30 # how fast games

#colors
WHITE = (255,255,255)
BLACK = (0,0,0)
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)


class Player(pygame.sprite.Sprite):
    # sprite for the player
    def __init__(self,):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((20,20)) #20ye 20lik bir şeyo oluşturma
        self.image.fill(BLUE) #maviye boya
        self.rect = self.image.get_rect()#sınırları belirliyor
        self.radius = 10
        pygame.draw.circle(self.image,RED,self.rect.center,self.radius)
        self.rect.centerx = WIDTH/2 
        self.rect.bottom = HEIGHT-1
        self.x_speed = 0
        
    def update(self,action):
       self.x_speed = 0
       key_state =pygame.key.get_pressed()
       
       if key_state[pygame.K_LEFT] or action == 0:
           self.x_speed = -4
       elif key_state[pygame.K_RIGHT] or action == 1:
           self.x_speed = 4
       else:
           self.x_speed = 0
       self.rect.x += self.x_speed
       if self.rect.right >WIDTH:
           self.rect.right = WIDTH
       if self.rect.left <0:
           self.rect.left = 0
           
    def getCoordinates(self):
        return (self.rect.x,self.rect.y)


class Enemy(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.Surface((10,10))
        self.image.fill(RED)
        self.rect = self.image.get_rect()
        self.radius = 5
        pygame.draw.circle(self.image,WHITE,self.rect.center,self.radius)
        self.rect.x = random.randrange(0,WIDTH-(self.rect.width))
        self.rect.y = random.randrange(2,6)
        self.x_speed = 0
        self.y_speed = 3
        
    def update(self):
       self.rect.x += self.x_speed
       self.rect.y += self.y_speed
       if self.rect.top > HEIGHT + 10:
           self.rect.x = random.randrange(0,WIDTH-(self.rect.width))
           self.rect.y = random.randrange(2,6)
           self.x_speed = 0
           self.y_speed = 3
           
    def getCoordinates(self):
        return (self.rect.x,self.rect.y)
    
class DQLAgent:
    def __init__(self):
        self.state_size = 4 #distance (playerx-m1x),(playery-m1y),(playerx-m2x),(playery-m2y)
        self.action_size = 3 #right,left,none(no move)
        
        #replay
        self.gamma = 0.95 #feature reward
        self.learning_rate = 0.001 #alfa
        
        #adaptiveEpsilonGreedy
        self.epsilon = 1 # en yüksek değer gitgide azaltacağız
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        #remember
        self.memory = deque(maxlen = 10000) #1000 lik bir liste
        
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
        state = np.array(state)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size) # env.action_space.sample()
        
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # en yüksek değeri döndür
    
    
    def replay(self,batch_size):
        #training 
        # memoryde en az 16tane state,action,reward,next_state,done varsa replay uygulanabilir
        if len(self.memory)<batch_size:
            return
        minibatch = random.sample(self.memory,batch_size) 
        #selfmemory içinden batch size kadar sample al
        
        for state,action,reward,next_state,done in minibatch:
            state = np.array(state)
            next_state = np.array(next_state)
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
    
    
class Env(pygame.sprite.Sprite):
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        self.batch_size = 520
        self.reward = 0
        self.done = False
        self.total_reward = 0
        self.agent = DQLAgent()
        
    def find_distance(self,a,b):
        distance = a-b
        return distance
    
    def step(self,action):
        state_list = []
        
        #update
        self.player.update(action)
        self.enemy.update()
        
        #get coordinate 
        next_player_state = self.player.getCoordinates()
        next_m1_state = self.m1.getCoordinates()
        next_m2_state = self.m2.getCoordinates()
        
        #find distance
        state_list.append(self.find_distance(next_player_state[0],next_m1_state[0]))
        state_list.append(self.find_distance(next_player_state[1],next_m1_state[1]))
        state_list.append(self.find_distance(next_player_state[0],next_m2_state[0]))
        state_list.append(self.find_distance(next_player_state[1],next_m2_state[1]))
        return [state_list]
        
    #reset
    def initialStates(self):
        pygame.sprite.Sprite.__init__(self)
        self.all_sprite = pygame.sprite.Group()
        self.enemy = pygame.sprite.Group()
        self.player = Player()
        self.all_sprite.add(self.player)
        self.m1 = Enemy()
        self.m2 = Enemy()
        self.all_sprite.add(self.m1)
        self.all_sprite.add(self.m2)
        self.enemy.add(self.m1)
        self.enemy.add(self.m2)
        
        self.reward = 0
        self.done = False
        self.total_reward = 0
        
        state_list = []
        #get coordinate
        player_state = self.player.getCoordinates()
        m1_state = self.m1.getCoordinates()
        m2_state = self.m2.getCoordinates()
        
        #find distance
        state_list.append(self.find_distance(player_state[0],m1_state[0]))
        state_list.append(self.find_distance(player_state[1],m1_state[1]))
        state_list.append(self.find_distance(player_state[0],m2_state[0]))
        state_list.append(self.find_distance(player_state[1],m2_state[1]))
        return [state_list]
        
        
    def run(self):
        #game loop
        state = self.initialStates()
        running = True
        while running:
            self.reward = 2
            # keep loop running at right speed
            clock.tick(FPS)
            #process input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
            #update
            action = self.agent.act(state)
            next_state = self.step(action)
            self.total_reward += self.reward
            
            hits = pygame.sprite.spritecollide(self.player,self.enemy,False,pygame.sprite.collide_circle)
            if hits:
                self.reward = -150
                self.total_reward += self.reward
                self.done = True
                running = False
                print("Total reward: ",self.total_reward)
                
            self.agent.remember(state, action, self.reward, next_state, self.done)
            #draw and render(show)
            state = next_state
            self.agent.replay(self.batch_size)
            self.agent.adaptiveEGreedy()
            
            screen.fill(GREEN)
            self.all_sprite.draw(screen)
            #after drawing flip display
            pygame.display.flip()
            
        pygame.quit()
    
    
    
    
#initialize pygame and create window
if __name__ == "__main__":
    env = Env()
    liste = []
    t =0
    while True:
        t+=1
        print("Episode : ",t)
        liste.append(env.total_reward)
        pygame.init()
        screen = pygame.display.set_mode((WIDTH,HEIGHT))
        pygame.display.set_caption("RL Game")
        clock = pygame.time.Clock()
        env.run()




    
    
    
    
    
    
    